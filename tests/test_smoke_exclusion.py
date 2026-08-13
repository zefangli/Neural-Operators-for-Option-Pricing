"""Smoke runs (`results_*_smoke/`) must never reach the aggregated tables.

The GPU plan runs every model twice: a 1-batch smoke run into `*_smoke/`, then
the real run. Smoke weights are meaningless; if their metrics.json is aggregated
it lands in a manuscript table as if it were a result.

Same precision lesson as tests/test_quarantine_filter.py: the predicate matches
a `results_`-prefixed directory part ENDING in `_smoke`, so no legitimate run dir
(`_v4`, `_don_v4`, `_seed43`, the original v3 dirs) can be caught by it.

Run: pytest tests/test_smoke_exclusion.py  or  python tests/test_smoke_exclusion.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from aggregate_results import PROJECT_ROOT, is_quarantined, is_smoke  # noqa: E402

# 1-batch sanity runs -- excluded.
SMOKE = [
    "train_model_v3/call/results_vol_surface_v4_smoke/metrics.json",
    "train_model_v3/call/results_vol_surface_don_v4_smoke/metrics.json",
    "train_model_v3/call/results_spot_history_v4_smoke/metrics.json",
    "train_model_v3/put/results_vix_history_v4_smoke/metrics.json",
    "train_model_v3/put/results_vix_history_don_v4_smoke/metrics.json",
    # the dir itself, not just a file inside it (eval_to_json passes the dir)
    "train_model_v3/call/results_vol_surface_v4_smoke",
]

# Everything a table is allowed to contain -- must NOT be caught by is_smoke.
NOT_SMOKE = [
    # corrected full v4 runs
    "train_model_v3/call/results_vol_surface_v4/metrics.json",
    "train_model_v3/call/results_vol_surface_don_v4/metrics.json",
    "train_model_v3/call/results_spot_history_v4/metrics.json",
    "train_model_v3/put/results_vix_history_v4/metrics.json",
    "train_model_v3/put/results_vix_history_don_v4/metrics.json",
    # seed variants
    "train_model_v3/call/results_vol_surface_v4_seed43/metrics.json",
    "train_model_v3/put/results_vix_history_don_v4_seed43/metrics.json",
    # original v3 runs: historical but valid -- dropping them is a separate
    # editorial call, not this filter's job
    "train_model_v3/call/results_vol_surface/metrics.json",
    "train_model_v3/put/results_spot_history_don/metrics.json",
    # a subsample dump inside a real run is a FILE, not a smoke run dir
    "train_model_v3/call/results_vol_surface_v4/metrics_smoke.json",
]

# Legacy VVIX artifacts -- excluded, but by quarantine (location), not by smoke.
VVIX = [
    "train_model_v3/_vix_is_vvix_LEGACY/call_results_vix_history/metrics.json",
    "train_model_v3/_vix_is_vvix_LEGACY/put_results_vix_history_don/metrics.json",
]


def excluded(rel):
    p = PROJECT_ROOT / rel
    return is_smoke(p) or is_quarantined(p)


@pytest.mark.parametrize("rel", SMOKE)
def test_smoke_runs_are_excluded(rel):
    assert is_smoke(PROJECT_ROOT / rel), f"{rel} must be excluded (1-batch smoke run)"


@pytest.mark.parametrize("rel", NOT_SMOKE)
def test_real_runs_are_not_smoke(rel):
    assert not is_smoke(PROJECT_ROOT / rel), (
        f"{rel} must NOT be excluded -- it is a legitimate run"
    )


@pytest.mark.parametrize("rel", NOT_SMOKE)
def test_real_runs_reach_aggregation(rel):
    assert not excluded(rel), f"{rel} must reach the aggregated tables"


@pytest.mark.parametrize("rel", VVIX)
def test_vvix_excluded_as_quarantine_not_smoke(rel):
    assert excluded(rel), f"{rel} must be excluded (VVIX artifact)"
    assert not is_smoke(PROJECT_ROOT / rel), (
        f"{rel} must be excluded for the QUARANTINE reason, so the warnings stay distinct"
    )


if __name__ == "__main__":
    ok = True
    for rel in SMOKE:
        got = is_smoke(PROJECT_ROOT / rel)
        print(f"{'OK  ' if got else 'FAIL'} smoke={got!s:5s} (want True )  {rel}")
        ok &= got
    for rel in NOT_SMOKE:
        got = excluded(rel)
        print(f"{'OK  ' if not got else 'FAIL'} excluded={got!s:5s} (want False)  {rel}")
        ok &= not got
    for rel in VVIX:
        got = excluded(rel) and not is_smoke(PROJECT_ROOT / rel)
        print(f"{'OK  ' if got else 'FAIL'} quarantined-not-smoke={got!s:5s}  {rel}")
        ok &= got
    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
