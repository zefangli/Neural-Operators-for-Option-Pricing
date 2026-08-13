"""The four vix_history training scripts must REFUSE a non-VIX (VVIX / v3) HDF5.

Regression test for the 2026-08-12 finding: the scripts defaulted to the v3 HDF5
(which carries VVIX in `vix_history`) while unconditionally stamping the output
config "source = CBOE SPX VIX", so a bare invocation would train on VVIX and
label it VIX -- silently re-creating the exact mislabel the migration undoes.

This file covers the VIX-only gate (source series + CSV sha256), which the other
8 scripts deliberately do not carry. The sample/schema gate shared by all 12
lives in test_v5_gate_all_scripts.py.

Run: pytest tests/test_vix_provenance_gate.py   or   python tests/test_vix_provenance_gate.py
"""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v5_gate_all_scripts import _synth_h5  # noqa: E402  (shared synth helper)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "wrds_data_2020-2025"

# (script, side, v3/VVIX HDF5 it must refuse)
CASES = [
    ("call/train_vix_history.py",     "call", "deeponet_tensors_call.h5"),
    ("call/train_vix_history_don.py", "call", "deeponet_tensors_call.h5"),
    ("put/train_vix_history.py",      "put",  "deeponet_tensors_put.h5"),
    ("put/train_vix_history_don.py",  "put",  "deeponet_tensors_put.h5"),
]


def _load(rel):
    p = ROOT / "train_model_v3" / rel
    spec = importlib.util.spec_from_file_location(f"m_{rel.replace('/', '_')[:-3]}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("rel,side,bad", CASES)
def test_default_h5_is_v5(rel, side, bad):
    """A bare run must not reach the v3 (VVIX) file, nor the unfiltered v4 one."""
    default = _load(rel).get_config()["h5_path"]
    assert default.endswith("_v5.h5"), f"{rel} defaults to {default}"


@pytest.mark.parametrize("rel,side,bad", CASES)
def test_refuses_v3_vvix_h5(rel, side, bad):
    """The v3 HDF5 has no schema_version and must be rejected before training."""
    h5 = DATA / bad
    if not h5.exists():
        pytest.skip(f"{bad} not present")
    m = _load(rel)
    cfg = m.get_config()
    cfg["h5_path"] = str(h5)
    with pytest.raises(SystemExit) as e:
        m.add_provenance(cfg)
    assert "schema_version" in str(e.value)


@pytest.mark.parametrize("rel,side,bad", CASES)
def test_refuses_wrong_vix_series(rel, side, bad):
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side, attrs={"phase1_vix_source_series": "CBOE VVIX"}) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert "vix source series" in str(e.value)


@pytest.mark.parametrize("rel,side,bad", CASES)
def test_refuses_wrong_vix_csv_hash(rel, side, bad):
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side, attrs={"phase1_vix_sha256": "0" * 64}) as h5:
        cfg["h5_path"] = str(h5)
        with pytest.raises(SystemExit) as e:
            m.add_provenance(cfg)
    assert "sha256" in str(e.value)


@pytest.mark.parametrize("rel,side,bad", CASES)
def test_accepts_v5_and_reads_provenance(rel, side, bad):
    """A genuine v5 file passes, and provenance is READ from it, not asserted."""
    m = _load(rel)
    cfg = m.get_config()
    with _synth_h5(side) as h5:
        cfg["h5_path"] = str(h5)
        m.add_provenance(cfg)
    prov = cfg["vix_provenance"]
    assert "CBOE VIX" in prov["source_series"]
    assert prov["schema_version"] == "v4"          # tensor schema, unchanged by v5
    assert cfg["dataset_version"] == "v5"          # sample definition
    assert prov["vix_csv_sha256"] == m.EXPECTED_VIX_SHA256
    assert cfg["h5_sha256"] and len(cfg["h5_sha256"]) == 64


if __name__ == "__main__":
    ok = True
    for rel, side, bad in CASES:
        for fn in (test_default_h5_is_v5, test_refuses_v3_vvix_h5,
                   test_refuses_wrong_vix_series, test_refuses_wrong_vix_csv_hash,
                   test_accepts_v5_and_reads_provenance):
            try:
                fn(rel, side, bad)
                print(f"PASS {fn.__name__:38s} {rel}")
            except Exception as exc:  # noqa: BLE001 - direct-run reporting
                print(f"FAIL {fn.__name__:38s} {rel}: {exc}")
                ok = False
    print("\nALL GATE TESTS PASS" if ok else "\nGATE TESTS FAILED")
    sys.exit(0 if ok else 1)
