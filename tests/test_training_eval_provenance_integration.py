"""The config a training run writes must be readable by analysis/eval_to_json.py.

Regression: training wrote `dataset_version` only at the TOP LEVEL of config.json
while `config_dataset_version()` read only the NESTED data_/vix_provenance block,
so it always returned "". Nothing noticed because run_provenance() also identifies
the sample by manifest hash. The hash is unavailable when the HDF5 no longer exists
at the recorded path -- then the config is the only source, and it read empty.

This test crosses the seam directly: real `add_provenance()` on the real v5 smoke
HDF5 -> real `config_dataset_version()` on the resulting dict. No manifest lookup,
no run_provenance() pipeline. One data_provenance script and one vix_provenance
script, since the two families write different block names.

Run: pytest tests/test_training_eval_provenance_integration.py
  or python tests/test_training_eval_provenance_integration.py
"""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SMOKE_V5 = ROOT / "wrds_data_2020-2025" / "smoke_v5" / "deeponet_tensors_call_v5_smoke.h5"

sys.path.insert(0, str(ROOT / "analysis"))
import eval_to_json  # noqa: E402

# (script, provenance block key) -- one of each family.
SCRIPTS = [("call/train_vol_surface.py", "data_provenance"),
           ("call/train_vix_history.py", "vix_provenance")]


def _load(rel):
    p = ROOT / "train_model_v3" / rel
    spec = importlib.util.spec_from_file_location(f"p_{rel.replace('/', '_')[:-3]}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _real_cfg(rel):
    if not SMOKE_V5.exists():
        pytest.skip(f"{SMOKE_V5.name} not present")
    m = _load(rel)
    cfg = m.get_config()
    cfg["h5_path"] = str(SMOKE_V5)
    m.add_provenance(cfg)
    return cfg


@pytest.mark.parametrize("rel,block", SCRIPTS)
def test_training_config_is_readable_by_eval(rel, block):
    cfg = _real_cfg(rel)
    prov = cfg[block]
    for key in ("dataset_version", "quote_filter", "quote_filter_tolerance",
                "quote_filter_bounds"):
        assert prov[key] == cfg[key], f"{rel}: {block}[{key}] != top-level"
    assert eval_to_json.config_dataset_version(cfg) == "v5"
    assert eval_to_json.config_schema_version(cfg) == "v4"  # v5 sample, v4 layout


@pytest.mark.parametrize("rel,block", SCRIPTS)
def test_nested_only_config_still_reads(rel, block):
    """v3/v4-era configs recorded the nested block only."""
    cfg = copy.deepcopy(_real_cfg(rel))
    cfg.pop("dataset_version")
    assert eval_to_json.config_dataset_version(cfg) == "v5"


@pytest.mark.parametrize("rel,block", SCRIPTS)
def test_disagreement_hard_fails(rel, block):
    cfg = copy.deepcopy(_real_cfg(rel))
    cfg[block]["dataset_version"] = "v4"
    with pytest.raises(SystemExit) as e:
        eval_to_json.config_dataset_version(cfg)
    assert "PROVENANCE MISMATCH" in str(e.value)


if __name__ == "__main__":
    ok = True
    for rel, block in SCRIPTS:
        for fn in (test_training_config_is_readable_by_eval,
                   test_nested_only_config_still_reads, test_disagreement_hard_fails):
            try:
                fn(rel, block)
                print(f"PASS {fn.__name__:45s} {rel}")
            except Exception as exc:  # noqa: BLE001 - direct-run reporting
                print(f"FAIL {fn.__name__:45s} {rel}: {exc}")
                ok = False
    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
