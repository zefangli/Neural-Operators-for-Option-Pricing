"""Guards on the F3 error-heatmap figure (analysis/fig_error_heatmap.py).

What this pins:
  (a) the BINNING RULE is fixed and data-independent -- the same maturity and
      log-moneyness edges for calls and puts, F2's three maturity bands intact.
      If the edges drift, the figure stops being predetermined.
  (b) `cell_stats` bins half-open (lo, hi] and computes RMSE, not something else.
  (c) the script runs end to end on a tiny subsample and writes its PNG.
  (d) THE REAL INVARIANT, twice over:
        * the grid is EXHAUSTIVE -- pooling the per-cell RMSEs back up
          (sqrt(sum(n_ij * rmse_ij^2) / sum(n_ij))) reproduces the overall RMSE
          exactly, so no row fell outside the plotted grid and no cell is an
          average of a different population than it claims;
        * the documented near-expiry handling holds -- v5 contains zero rows with
          T <= 1 day, so nothing is silently excluded and no bucket is the
          documented T->0 decoder degeneracy in disguise.

(c)/(d) need the v5 HDF5 and the trained per-query run, so they skip when absent.

Run: pytest tests/test_fig_error_heatmap.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import fig_error_heatmap as f3  # noqa: E402


# ------------------------------------------------------------ (a) the bin edges

def test_maturity_edges_are_fixed_and_contain_f2s_bands():
    assert f3.T_EDGES == [1.0, 7.0, 30.0, 90.0, 365.0, np.inf]
    assert f3.T_LABELS == ["1-7d", "7-30d", "30-90d", "90-365d", ">365d"]
    # F2's (7,30]/(30,90]/(90,365] survive verbatim as the middle three rows.
    assert f3.T_EDGES[1:5] == [7.0, 30.0, 90.0, 365.0]


def test_moneyness_edges_are_fixed_and_symmetric():
    inner = f3.M_EDGES[1:-1]
    assert inner == [-0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20]
    assert inner == [-e for e in reversed(inner)]      # symmetric about ATM
    assert f3.M_EDGES[0] == -np.inf and f3.M_EDGES[-1] == np.inf   # wings unbounded
    assert len(f3.M_LABELS) == len(f3.M_EDGES) - 1


# --------------------------------------------------------------- (b) cell_stats

def test_cell_stats_bins_half_open_and_computes_rmse():
    # Two rows in the (1,7] x ATM cell with log residuals +0.3 / -0.1, one row on
    # the exact bucket edge T=7d (belongs to (1,7], not (7,30]), one deep-wing row.
    T = np.array([3.0, 3.0, 7.0, 200.0])
    lm = np.array([0.0, 0.0, 0.5, -0.5])
    tgt = np.array([-1.0, -1.0, -2.0, -3.0])
    pred = tgt + np.array([0.3, -0.1, 0.0, 0.4])
    rmse_log, _rmse_price, n = f3.cell_stats(pred, tgt, T, lm)

    atm = f3.M_LABELS.index("-.02/.02")
    assert n[0, atm] == 2
    assert rmse_log[0, atm] == pytest.approx(np.sqrt((0.09 + 0.01) / 2))
    assert n[0, -1] == 1          # T = 7.0 lands in (1,7]: half-open upper edge
    assert n[3, 0] == 1           # T = 200d, log_m = -0.5 -> (90,365] x <=-0.20
    assert n.sum() == 4           # every row landed somewhere
    assert np.isnan(rmse_log[n == 0]).all()


def test_cell_stats_grid_is_exhaustive_on_random_input():
    rng = np.random.default_rng(0)
    T = rng.uniform(1.8, 2500.0, 5000)
    lm = rng.normal(0.0, 0.4, 5000)
    tgt = rng.normal(-3.0, 1.0, 5000)
    pred = tgt + rng.normal(0.0, 0.2, 5000)
    rmse_log, _p, n = f3.cell_stats(pred, tgt, T, lm)
    assert n.sum() == 5000
    pooled = np.sqrt(np.nansum(n * rmse_log ** 2) / n.sum())
    assert pooled == pytest.approx(np.sqrt(np.mean((pred - tgt) ** 2)))


# ------------------------------------------- (c)/(d) end-to-end on a tiny cut

@pytest.mark.parametrize("option_type", ["call"])
def test_end_to_end_tiny_grid_is_exhaustive_and_no_near_expiry_rows(option_type, tmp_path):
    try:
        f3.load_per_query(option_type)     # provenance gate + the checkpoint
    except (SystemExit, FileNotFoundError) as exc:
        pytest.skip(f"v5 data / trained run unavailable: {exc}")

    png = tmp_path / "f3.png"
    s = f3.make_figure(option_type, png, max_contracts=3000)

    assert png.exists() and png.stat().st_size > 0
    assert s["n"] == 3000 == s["counts"].sum()
    assert s["counts"].shape == (len(f3.T_LABELS), len(f3.M_LABELS))

    # The documented near-expiry handling: v5's floor is 1.729 days, so the
    # headline `T > 1 day` filter drops nothing and no bucket is the T->0
    # decoder degeneracy. If this ever fires, the bottom row is misleading.
    assert s["n_T_le_1day"] == 0
    assert s["T_days_range"][0] > 1.0

    # Exhaustiveness on real data: the cells pool back to the overall RMSE, so
    # every test row is inside the plotted grid.
    n, rl, rp = s["counts"], s["rmse_log"], s["rmse_price"]
    assert np.sqrt(np.nansum(n * rl ** 2) / n.sum()) == pytest.approx(
        s["rmse_log_overall"], rel=1e-9)
    assert np.sqrt(np.nansum(n * rp ** 2) / n.sum()) == pytest.approx(
        s["rmse_price_overall"], rel=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
