"""All 12 core training scripts must refuse a v4 (unfiltered) HDF5 and accept a v5 one.

v5 is the canonical sample: the zero-tolerance static no-arbitrage midpoint filter
drops the rows the BS decoder cannot reach at any sigma_hat (call ~2.2%, put ~0.8%).
The tensor *schema* is unchanged, so `schema_version` stays "v4" and a second,
independent attr `dataset_version = "v5"` identifies the sample. That is why the
old schema-only gate is no longer sufficient: a v4 file satisfies it.

Gate enforced by every one of the 12 (all must hold, all abort via SystemExit
before any data loads):
    dataset_version == "v5"
    quote_filter == "static_bounds_midpoint"  and  quote_filter_tolerance == 0.0
    schema_version == "v4"
    cp_flag matches the script's option_type
    export_t_basis == "settlement";  export_min_maturity_days == 1.0
    `split_id` and the script's own branch_key exist as datasets
The 4 vix_history scripts additionally gate the VIX source series + CSV sha256
(see test_vix_provenance_gate.py). The other 8 deliberately do not -- they never
read `vix_history`, so it would be a false constraint.

Run: pytest tests/test_v5_gate_all_scripts.py  or  python tests/test_v5_gate_all_scripts.py
"""
import contextlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "wrds_data_2020-2025"

BRANCHES = ["vol_surface", "vol_surface_don", "spot_history", "spot_history_don",
            "vix_history", "vix_history_don"]
SCRIPTS = [(f"{side}/train_{b}.py", side) for side in ("call", "put") for b in BRANCHES]

# The v4 smoke files are real, unfiltered v4 -- now the headline refusal case,
# and (attrs-only) the base for synthesizing a v5-looking file.
SMOKE_V4 = {"call": "smoke_v4/deeponet_tensors_call_v4_smoke.h5",
            "put": "smoke_v4/deeponet_tensors_put_v4_smoke.h5"}
V3 = {"call": "deeponet_tensors_call.h5", "put": "deeponet_tensors_put.h5"}
OTHER = {"call": "put", "put": "call"}

# What promotes a v4 file to the canonical v5 sample.
V5_ATTRS = {
    "dataset_version": "v5",
    "quote_filter": "static_bounds_midpoint",
    "quote_filter_tolerance": 0.0,
    "quote_filter_bounds": "call: max(0, M e^{-qT} - e^{-rT}) <= V/K <= M e^{-qT}",
}


@contextlib.contextmanager
def _synth_h5(side, *, attrs=None, drop=(), v5=True):
    """A throwaway HDF5: real attrs from the v4 smoke file, + v5 attrs, + overrides.

    Never touches a real file -- it copies attrs (not data) into a temp file with
    stub datasets, so a single gate can be broken in isolation.
    """
    src = DATA / SMOKE_V4[side]
    if not src.exists():
        pytest.skip(f"{SMOKE_V4[side]} not present")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "synth_v5.h5"
        with h5py.File(src, "r") as fin, h5py.File(out, "w") as fout:
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            for k, v in (V5_ATTRS if v5 else {}).items():
                fout.attrs[k] = v
            for k, v in (attrs or {}).items():
                if v is None:
                    fout.attrs.pop(k, None)
                else:
                    fout.attrs[k] = v
            for name in ("split_id", "branch_u", "spot_history", "vix_history"):
                if name not in drop:
                    fout.create_dataset(name, data=np.zeros(4, dtype="i4"))
        yield out


def _load(rel):
    p = ROOT / "train_model_v3" / rel
    spec = importlib.util.spec_from_file_location(f"g_{rel.replace('/', '_')[:-3]}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_all_twelve_scripts_exist():
    missing = [r for r, _ in SCRIPTS if not (ROOT / "train_model_v3" / r).is_file()]
    assert not missing, f"missing training scripts: {missing}"


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_default_h5_and_results_are_v5(rel, side):
    """A bare invocation must reach neither older data nor any older run's weights."""
    cfg = _load(rel).get_config()
    assert cfg["h5_path"].endswith(f"_{side}_v5.h5"), f"{rel}: h5 default {cfg['h5_path']}"
    assert cfg["results_dir"].endswith("_v5"), f"{rel}: results default {cfg['results_dir']}"


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_refuses_v4_hdf5(rel, side):
    """THE headline regression: the unfiltered v4 sample is noncanonical for training.

    It passes `schema_version == "v4"`, so only the dataset_version gate stops it.
    """
    h5 = DATA / SMOKE_V4[side]
    if not h5.exists():
        pytest.skip(f"{SMOKE_V4[side]} not present")
    m = _load(rel)
    cfg = m.get_config()
    cfg["h5_path"] = str(h5)
    with pytest.raises(SystemExit) as e:
        m.add_provenance(cfg)
    assert "dataset_version" in str(e.value)


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_refuses_v3_hdf5(rel, side):
    h5 = DATA / V3[side]
    if not h5.exists():
        pytest.skip(f"{V3[side]} not present")
    m = _load(rel)
    cfg = m.get_config()
    cfg["h5_path"] = str(h5)
    with pytest.raises(SystemExit) as e:
        m.add_provenance(cfg)
    assert "schema_version" in str(e.value)


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_accepts_v5_hdf5(rel, side):
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side) as h5:
        cfg["h5_path"] = str(h5)
        m.add_provenance(cfg)
    assert cfg["h5_sha256"] and len(cfg["h5_sha256"]) == 64
    assert cfg["dataset_version"] == "v5"
    assert cfg["quote_filter"] == "static_bounds_midpoint"
    assert cfg["quote_filter_tolerance"] == 0.0
    # provenance is recorded under one of the two documented keys
    prov = cfg.get("data_provenance") or cfg.get("vix_provenance")
    assert prov, f"{rel}: no provenance recorded"


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_refuses_missing_file(rel, side):
    m = _load(rel)
    cfg = m.get_config()
    cfg["h5_path"] = str(DATA / "does_not_exist_v5.h5")
    with pytest.raises(SystemExit):
        m.add_provenance(cfg)


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_refuses_opposite_option_type(rel, side):
    """call and put files are structurally identical -- only cp_flag separates them."""
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(OTHER[side]) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert "cp_flag" in str(e.value)


@pytest.mark.parametrize("rel,side", SCRIPTS)
@pytest.mark.parametrize("bad", ["", "static_bounds_bid_ask", None])
def test_refuses_wrong_quote_filter(rel, side, bad):
    """Wrong or missing quote_filter -- a differently-filtered sample is not v5."""
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side, attrs={"quote_filter": bad}) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert "quote_filter" in str(e.value)


@pytest.mark.parametrize("rel,side", SCRIPTS)
@pytest.mark.parametrize("bad", [0.01, None])
def test_refuses_wrong_quote_filter_tolerance(rel, side, bad):
    """Only the approved zero-tolerance filter is canonical."""
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side, attrs={"quote_filter_tolerance": bad}) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert "quote_filter_tolerance" in str(e.value)


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_refuses_wrong_min_maturity(rel, side):
    """The canonical paper rule is T > 1 day; any other export threshold aborts."""
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side, attrs={"export_min_maturity_days": 0.0}) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert "export_min_maturity_days" in str(e.value)


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_refuses_wrong_t_basis(rel, side):
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side, attrs={"export_t_basis": "calendar"}) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert "export_t_basis" in str(e.value)


@pytest.mark.parametrize("rel,side", SCRIPTS)
def test_refuses_missing_branch_dataset(rel, side):
    """Each script must check for its own branch input, not a hardcoded name."""
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side, drop=(cfg["branch_key"],)) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert cfg["branch_key"] in str(e.value)


if __name__ == "__main__":
    ok = True
    try:
        test_all_twelve_scripts_exist()
        print("PASS all 12 scripts present")
    except AssertionError as exc:
        print(f"FAIL {exc}")
        ok = False
    for rel, side in SCRIPTS:
        cases = [(f, ()) for f in (
            test_default_h5_and_results_are_v5, test_refuses_v4_hdf5, test_refuses_v3_hdf5,
            test_accepts_v5_hdf5, test_refuses_missing_file, test_refuses_opposite_option_type,
            test_refuses_wrong_min_maturity, test_refuses_wrong_t_basis,
            test_refuses_missing_branch_dataset)]
        cases += [(test_refuses_wrong_quote_filter, (b,)) for b in ("", "static_bounds_bid_ask", None)]
        cases += [(test_refuses_wrong_quote_filter_tolerance, (b,)) for b in (0.01, None)]
        for fn, extra in cases:
            try:
                fn(rel, side, *extra)
                print(f"PASS {fn.__name__:40s} {rel} {extra if extra else ''}")
            except Exception as exc:  # noqa: BLE001 - direct-run reporting
                print(f"FAIL {fn.__name__:40s} {rel} {extra if extra else ''}: {exc}")
                ok = False
    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
