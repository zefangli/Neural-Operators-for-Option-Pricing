"""Quarantine must exclude the VVIX legacy runs WITHOUT excluding corrected v4 runs.

Regression test for the 2026-08-12 finding: `is_quarantined()` matched any path
part starting with "results_vix_history", which also matched the corrected
`results_vix_history_v4/` directories -- so the retrained VIX runs would have
been silently dropped from every aggregated table and never reached the paper.
Quarantine is a LOCATION (_vix_is_vvix_LEGACY/), not a name prefix.

Run: pytest tests/test_quarantine_filter.py  or  python tests/test_quarantine_filter.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from aggregate_results import PROJECT_ROOT, is_quarantined  # noqa: E402

QUARANTINED = [
    "train_model_v3/_vix_is_vvix_LEGACY/call_results_vix_history/metrics.json",
    "train_model_v3/_vix_is_vvix_LEGACY/call_results_vix_history_don/metrics.json",
    "train_model_v3/_vix_is_vvix_LEGACY/put_results_vix_history/metrics.json",
    "train_model_v3/_vix_is_vvix_LEGACY/put_results_vix_history_don/metrics.json",
]

# The corrected retrained runs -- these MUST reach aggregation.
NOT_QUARANTINED = [
    "train_model_v3/call/results_vix_history_v4/metrics.json",
    "train_model_v3/call/results_vix_history_don_v4/metrics.json",
    "train_model_v3/put/results_vix_history_v4/metrics.json",
    "train_model_v3/put/results_vix_history_don_v4/metrics.json",
    # and the untouched non-VIX runs
    "train_model_v3/call/results_vol_surface/metrics.json",
    "train_model_v3/call/results_vol_surface_don/metrics.json",
    "train_model_v3/put/results_spot_history/metrics.json",
    "train_model_v3/put/results_spot_history_don/metrics.json",
    # future v4 runs of the other branches
    "train_model_v3/call/results_vol_surface_v4/metrics.json",
    "train_model_v3/put/results_spot_history_v4/metrics.json",
]


@pytest.mark.parametrize("rel", QUARANTINED)
def test_legacy_vvix_is_quarantined(rel):
    assert is_quarantined(PROJECT_ROOT / rel), f"{rel} must be excluded (VVIX artifact)"


@pytest.mark.parametrize("rel", NOT_QUARANTINED)
def test_corrected_runs_are_not_quarantined(rel):
    assert not is_quarantined(PROJECT_ROOT / rel), (
        f"{rel} must NOT be excluded -- corrected/valid runs have to reach the tables"
    )


if __name__ == "__main__":
    ok = True
    for rel in QUARANTINED:
        got = is_quarantined(PROJECT_ROOT / rel)
        print(f"{'OK  ' if got else 'FAIL'} quarantined={got!s:5s} (want True )  {rel}")
        ok &= got
    for rel in NOT_QUARANTINED:
        got = is_quarantined(PROJECT_ROOT / rel)
        print(f"{'OK  ' if not got else 'FAIL'} quarantined={got!s:5s} (want False)  {rel}")
        ok &= not got
    print("\nALL OK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)
