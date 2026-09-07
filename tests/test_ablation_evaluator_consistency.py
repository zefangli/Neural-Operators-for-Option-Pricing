"""Unit tests for analysis/ablation_evaluator_consistency.py on synthetic arrays.

No HDF5, no checkpoint, no GPU: every test builds its own arrays. What is pinned
here is the comparison arithmetic the diagnostic's conclusion rests on -- the
pairwise |delta| statistics, the divergent-row characterisation (including the
clamp-flip count, which is the mechanism the artifact names), the metric deltas,
and the decode-isolation logic that separates a sigma_hat difference from
decoder cancellation near the 1e-8 price clamp.

Run: pytest tests/test_ablation_evaluator_consistency.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from ablation_evaluator_consistency import (  # noqa: E402
    CLAMP_LOG, compare_arrays, decode, divergent_rows, metric_deltas)
from _common import compute_metrics  # noqa: E402


# ------------------------------------------------------------ compare_arrays

def test_identical_arrays_report_zero_and_full_equality():
    a = np.array([1.0, 2.0, 3.0])
    c = compare_arrays(a, a.copy())
    assert c["n"] == 3 and c["n_exactly_equal"] == 3
    assert c["max"] == 0.0 and c["mean"] == 0.0 and c["p99"] == 0.0
    assert c["n_gt_1e-6"] == 0 and c["n_gt_0.1"] == 0


def test_max_mean_and_percentiles_are_on_absolute_differences():
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, -1.0, 0.0, 0.0])          # signs must not cancel
    c = compare_arrays(a, b)
    assert c["max"] == 1.0
    assert c["mean"] == pytest.approx(0.5)
    assert c["p50"] == pytest.approx(0.5)
    assert c["n_exactly_equal"] == 2


def test_threshold_counts_are_strict_greater_than():
    c = compare_arrays(np.zeros(3), np.array([0.1, 0.1000001, 1.0]))
    assert c["n_gt_0.1"] == 2                     # 0.1 itself does not count
    assert c["n_gt_1"] == 0                       # 1.0 itself does not count


def test_non_finite_rows_are_excluded_from_the_statistics_and_counted():
    c = compare_arrays(np.array([0.0, 0.0]), np.array([1.0, np.inf]))
    assert c["n"] == 2 and c["n_nonfinite"] == 1
    assert c["max"] == 1.0


# ------------------------------------------------------------ divergent_rows

def _rows(delta_logp):
    n = len(delta_logp)
    a = np.full(n, -5.0)
    b = a + np.asarray(delta_logp, dtype=float)
    log_m = np.linspace(-0.5, 0.5, n)
    T = np.full(n, 30.0 / 365.0)
    target = np.full(n, -5.0)
    return a, b, log_m, T, target


def test_no_row_over_threshold_returns_only_the_count():
    a, b, lm, T, tg = _rows([0.0, 0.05, -0.09])
    d = divergent_rows(a, b, lm, T, tg)
    assert d["n_rows"] == 0 and d["pct_rows"] == 0.0
    assert "log_moneyness_min_med_max" not in d


def test_divergent_rows_reports_where_the_moved_rows_sit():
    a, b, lm, T, tg = _rows([0.0, 0.5, 0.0, 2.0])
    d = divergent_rows(a, b, lm, T, tg)
    assert d["n_rows"] == 2
    assert d["pct_rows"] == pytest.approx(50.0)
    assert d["max_abs_delta"] == pytest.approx(2.0)
    assert d["sum_sq_delta_contribution"] == pytest.approx(0.25 + 4.0)
    assert d["T_days_min_med_max"][1] == pytest.approx(30.0)


def test_clamp_flip_is_counted_only_when_the_two_paths_disagree_about_the_floor():
    """The mechanism the artifact names: one path lands on log(1e-8), the other
    does not."""
    a = np.array([CLAMP_LOG, CLAMP_LOG, -3.0])
    b = np.array([CLAMP_LOG, -3.0, -3.0])         # row 1 flips off the clamp
    lm, T, tg = np.zeros(3), np.full(3, 0.1), np.full(3, -3.0)
    d = divergent_rows(a, b, lm, T, tg)
    assert d["n_rows"] == 1
    assert d["n_a_at_clamp"] == 1 and d["n_b_at_clamp"] == 0
    assert d["n_clamp_state_flipped"] == 1


def test_targets_below_the_clamp_are_counted_separately_from_predictions():
    a = np.array([-3.0, -3.0])
    b = np.array([-9.0, -3.0])
    tg = np.array([CLAMP_LOG - 1.0, -3.0])        # a genuinely sub-clamp target
    d = divergent_rows(a, b, np.zeros(2), np.full(2, 0.1), tg)
    assert d["n_rows"] == 1 and d["n_target_below_clamp"] == 1
    assert d["n_a_at_clamp"] == 0                 # the PREDICTIONS are not clamped


# ------------------------------------------------------------- metric_deltas

def test_metric_deltas_subtracts_b_from_a_and_skips_absent_keys():
    a = {"r2_log_full": 0.8, "rmse_log": 1.0, "r2_price": 0.9}
    b = {"r2_log_full": 0.7, "rmse_log": 1.5, "r2_price": 0.9}
    d = metric_deltas(a, b)
    assert d["r2_log_full"] == pytest.approx(0.1)
    assert d["rmse_log"] == pytest.approx(-0.5)
    assert d["r2_price"] == 0.0
    assert "r2_log_filtered" not in d             # absent in both -> not invented


# ----------------------------------------------------- decode isolation logic

def test_naive_decoder_clamps_at_1e_minus_8_and_the_stable_one_does_not():
    """A deep-OTM, short-dated contract at a small sigma prices far below 1e-8."""
    kw = dict(log_m=[-1.5], T=[1.0 / 365.0], r=[0.02], q=[0.01], option_type="call")
    sigma = np.array([0.05], dtype=np.float32)
    naive = decode(sigma, stable=False, **kw)
    stable = decode(sigma, stable=True, **kw)
    assert naive[0] == pytest.approx(CLAMP_LOG, abs=1e-9)
    assert stable[0] < CLAMP_LOG - 1.0            # the floor is what differs


def test_float64_and_stable_decoders_agree_where_the_price_is_not_near_zero():
    kw = dict(log_m=[0.0, 0.05], T=[0.5, 1.0], r=[0.02, 0.02], q=[0.0, 0.0],
              option_type="call")
    sigma = np.array([0.2, 0.3], dtype=np.float32)
    assert decode(sigma, stable=False, **kw) == pytest.approx(
        decode(sigma, stable=True, **kw), rel=1e-9)


def test_a_tiny_sigma_difference_is_amplified_only_near_the_clamp():
    """This is the diagnostic's whole claim, stated as a test: the same 1e-7
    relative perturbation of sigma_hat moves log-price negligibly at the money
    and enormously in the deep wing."""
    s0 = np.float32(0.05)
    s1 = np.nextafter(s0, np.float32(1.0), dtype=np.float32)
    atm = dict(log_m=[0.0], T=[0.5], r=[0.02], q=[0.0], option_type="call")
    wing = dict(log_m=[-1.5], T=[1.0 / 365.0], r=[0.02], q=[0.0], option_type="call")
    d_atm = abs(decode(np.array([s1]), stable=True, **atm)[0]
                - decode(np.array([s0]), stable=True, **atm)[0])
    d_wing = abs(decode(np.array([s1]), stable=True, **wing)[0]
                 - decode(np.array([s0]), stable=True, **wing)[0])
    assert d_atm < 1e-5
    assert d_wing > 100 * max(d_atm, 1e-12)


def test_decode_isolation_separates_sigma_differences_from_decoder_cancellation():
    """Two paths whose sigma_hat is IDENTICAL must show a zero float64 delta even
    when their float32 log-prices differ; that is exactly how the artifact
    concludes 'decoder, not model'."""
    kw = dict(log_m=[0.0, -1.5], T=[0.5, 1.0 / 365.0], r=[0.02, 0.02], q=[0.0, 0.0],
              option_type="call")
    sigma = np.array([0.2, 0.05], dtype=np.float32)
    lp64_a = decode(sigma, stable=False, **kw)
    lp64_b = decode(sigma.copy(), stable=False, **kw)
    assert compare_arrays(lp64_a, lp64_b)["max"] == 0.0

    # ... whereas genuinely different sigma_hat survives the float64 re-decode.
    sigma2 = sigma.copy()
    sigma2[0] = np.float32(0.21)
    assert compare_arrays(lp64_a, decode(sigma2, stable=False, **kw))["max"] > 1e-3


def test_metrics_move_only_through_the_rows_that_moved():
    """Guards the reported metric deltas: an unchanged row contributes nothing."""
    target = np.array([-3.0, -3.0, -12.0])
    T = np.full(3, 0.5)
    base = np.array([-3.0, -3.0, -12.0])
    moved = base.copy()
    moved[2] = CLAMP_LOG
    m0 = compute_metrics(base, target, T)
    m1 = compute_metrics(moved, target, T)
    d = metric_deltas(m0, m1)
    assert d["r2_log_full"] > 0.1                 # the moved row costs real accuracy
    # ... while in price space the same row is worth ~1e-5 and moves nothing:
    assert abs(d["r2_price"]) < 1e-6
    assert abs(d["r2_price"]) < 1e-5 * abs(d["r2_log_full"])
