"""Guards on the F2 smile-recovery figure (analysis/fig_smile_recovery.py).

What this pins:
  (a) the SELECTION RULE is mechanical -- dates come from quantiles of the
      ordered test-date axis and the panel expiry is the one nearest the bucket
      midpoint (ties to the shorter maturity). If either drifts, the figure stops
      being "predetermined" and becomes cherry-picked.
  (b) the script runs end to end on a tiny subsample (one date x one bucket) and
      actually writes its PNG.
  (c) the scalar-sigma curve is CONSTANT within every panel -- a real invariant
      of the ablation architecture (sigma_hat sees the market-state latent only),
      not a cosmetic property of the plot.

(b)/(c) need the v5 HDF5 and the four trained runs, so they skip when absent.

Run: pytest tests/test_fig_smile_recovery.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import fig_smile_recovery as f2  # noqa: E402


# ------------------------------------------------------- (a) the selection rule

def test_pick_dates_is_quantiles_of_the_ordered_date_axis():
    dates = np.array([f"2020-01-{d:02d}" for d in range(1, 21)]).repeat(3)
    # 20 unique dates -> indices 5, 10, 14 at the 25/50/75th percentile ("nearest").
    assert f2.pick_dates(dates) == ["2020-01-06", "2020-01-11", "2020-01-15"]


def test_pick_expiry_takes_the_maturity_nearest_the_bucket_midpoint():
    days = np.array([8.0, 20.0, 29.0, 45.0])          # bucket (7, 30], midpoint 18.5
    assert f2.pick_expiry(days, 7.0, 30.0) == 20.0
    assert f2.pick_expiry(days, 30.0, 90.0) == 45.0
    assert f2.pick_expiry(days, 90.0, 365.0) is None   # empty cell, not back-filled


def test_pick_expiry_breaks_ties_toward_the_shorter_maturity():
    days = np.array([17.5, 19.5])                      # both 1.0 day from 18.5
    assert f2.pick_expiry(days, 7.0, 30.0) == 17.5


def test_buckets_are_contiguous_and_fixed():
    assert [b[0] for b in f2.BUCKETS] == ["short (7-30d)", "medium (30-90d)", "long (90-365d)"]
    edges = [(lo, hi) for _, lo, hi in f2.BUCKETS]
    assert edges == [(7.0, 30.0), (30.0, 90.0), (90.0, 365.0)]


# --------------------------------------------- (b)/(c) end-to-end on a tiny cut

@pytest.mark.parametrize("option_type", ["call"])
def test_end_to_end_tiny_and_scalar_curve_is_flat(option_type, tmp_path):
    try:
        f2.load_pair(option_type)          # provenance gate + the four checkpoints
    except (SystemExit, FileNotFoundError) as exc:
        pytest.skip(f"v5 data / trained runs unavailable: {exc}")

    png = tmp_path / "f2.png"
    # One date x one bucket: the cheapest cut that still exercises the whole path.
    _cfg, m_pq, _m_sc, h5 = f2.load_pair(option_type)
    dates = f2.select_panels(h5)[1][:1]
    stats = f2.make_figure(option_type, png, dates=dates, buckets=f2.BUCKETS[:1])

    assert png.exists() and png.stat().st_size > 0
    assert len(stats) == 1 and stats[0]["n"] > 0
    s = stats[0]
    # The invariant: scalar-sigma is constant across moneyness (float32 headroom
    # only), while the per-query curve genuinely varies -- that IS the figure.
    assert s["scalar_ptp"] <= f2.SCALAR_FLAT_RTOL * abs(s["scalar_sigma"])
    assert s["per_query_ptp"] > 1e-3
    assert 0.01 < s["scalar_sigma"] < 2.0
    assert m_pq is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
