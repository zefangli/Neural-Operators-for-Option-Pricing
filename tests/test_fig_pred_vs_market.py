"""Guards on the F5 predicted-vs-market scatter (analysis/fig_pred_vs_market.py).

What this pins:
  (a) the SAMPLING RULE is the repo's fixed, seeded, uniform draw -- seed 42 via
      `_common.load_test_split`, not an ad-hoc or hand-tuned selection.
  (b) F5 reads the SAME checkpoint under the SAME provenance gate as F3 (it
      imports `load_per_query`), and buckets maturity on F3's edges.
  (c) the script runs end to end on a tiny sample and writes its PNG.
  (d) THE REAL INVARIANT: on that sample the predicted and market normalized
      prices are strictly positive (a BS price at sigma_hat > 0 cannot be
      negative -- if this fails the log-log axes are silently dropping rows) and
      the two genuinely track each other rather than forming a cloud: Pearson r
      on log10 prices above 0.9 and a median absolute relative error below 50%.
      Loose enough to survive retraining, tight enough that a broken decoder,
      a shuffled row order, or a mismatched checkpoint fails it.

(c)/(d) need the v5 HDF5 and the trained per-query run, so they skip when absent.

Run: pytest tests/test_fig_pred_vs_market.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import fig_error_heatmap as f3  # noqa: E402
import fig_pred_vs_market as f5  # noqa: E402


# ------------------------------------------------- (a)/(b) rules, not accidents

def test_sampling_rule_is_the_repo_seeded_uniform_draw():
    assert f5.SAMPLE_SEED == 42
    assert f5.SAMPLE_N == 50_000


def test_shares_f3s_provenance_gate_and_maturity_grid():
    assert f5.load_per_query is f3.load_per_query
    assert f5.T_EDGES is f3.T_EDGES and f5.T_LABELS is f3.T_LABELS
    assert len(f5.COLORS) >= len(f3.T_LABELS)      # one colour per maturity bucket


# ------------------------------------------------ (c)/(d) end-to-end on a tiny cut

@pytest.mark.parametrize("option_type", ["call"])
def test_end_to_end_tiny_prices_positive_and_tracking(option_type, tmp_path):
    try:
        f5.load_per_query(option_type)
    except (SystemExit, FileNotFoundError) as exc:
        pytest.skip(f"v5 data / trained run unavailable: {exc}")

    png = tmp_path / "f5.png"
    s = f5.make_figure(option_type, png, sample_n=2000)

    assert png.exists() and png.stat().st_size > 0
    assert s["n"] == 2000

    # Positivity: both axes are log-scaled, so a non-positive price would be a
    # silently dropped point rather than a visible failure.
    assert s["min_price_nonneg"]
    assert s["market_range"][0] > 0 and s["pred_range"][0] > 0

    # Tracking: the figure's whole claim is that the cloud sits on y = x.
    assert s["pearson_log10"] > 0.9
    assert s["median_abs_rel_err"] < 0.5
    assert s["r2_price"] > 0.9
    assert np.isfinite(s["rmse_log"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
