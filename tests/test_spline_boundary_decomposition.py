"""Unit tests for analysis/spline_boundary_decomposition.py on synthetic arrays.

No HDF5, no surface CSV, no network: every test builds its own tiny arrays.
What is pinned here is the arithmetic the manuscript quotes -- which rows land in
which mask (including exactly-on-the-boundary rows), that the masks overlap the
way the artifact claims, that a partition's SSE shares sum to 100%, that the
within-mask R2 is the R2 of that subset alone, and the tail counts.

Run: pytest tests/test_spline_boundary_decomposition.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from spline_boundary_decomposition import boundary_masks, mask_stats  # noqa: E402

# The real OptionMetrics axes: |delta| 10..90, 10..730 calendar days.
CALL_DELTA_AXIS = np.arange(10.0, 91.0, 5.0)
PUT_DELTA_AXIS = -CALL_DELTA_AXIS[::-1]
DAYS_AXIS = np.array([10.0, 30.0, 60.0, 91.0, 122.0, 152.0, 182.0, 273.0, 365.0, 547.0, 730.0])

MID_T = 100.0 / 365.0          # comfortably inside the tenor grid
MID_DELTA = 50.0               # comfortably inside the delta grid


def _stats(mask, sq_log, sq_price, rel, log_pred, log_target, T):
    return mask_stats(mask, sq_log, sq_price, rel, log_pred, log_target, T)


# ------------------------------------------------------------------ masks

@pytest.mark.parametrize("axis,sign", [(CALL_DELTA_AXIS, 1.0), (PUT_DELTA_AXIS, -1.0)])
def test_delta_boundary_is_exclusive_at_10_and_90(axis, sign):
    """Exactly 10 and exactly 90 delta are IN grid; 9.999/90.001 are out."""
    delta = sign * np.array([9.999, 10.0, 10.001, 50.0, 89.999, 90.0, 90.001])
    T = np.full(delta.shape, MID_T)
    m = boundary_masks(delta, T, axis, DAYS_AXIS)
    assert m["below_10_delta"].tolist() == [True, False, False, False, False, False, False]
    assert m["above_90_delta"].tolist() == [False, False, False, False, False, False, True]
    assert m["in_grid"].tolist() == [False, True, True, True, True, True, False]


def test_tenor_boundary_is_exclusive_at_10_and_730_days():
    days = np.array([9.9, 10.0, 10.1, 365.0, 729.9, 730.0, 730.1])
    delta = np.full(days.shape, MID_DELTA)
    m = boundary_masks(delta, days / 365.0, CALL_DELTA_AXIS, DAYS_AXIS)
    assert m["tenor_out"].tolist() == [True, False, False, False, False, False, True]
    assert m["in_grid"].tolist() == [False, True, True, True, True, True, False]


def test_masks_use_absolute_delta_so_puts_and_calls_agree():
    a = np.array([5.0, 50.0, 95.0])
    T = np.full(3, MID_T)
    call = boundary_masks(a, T, CALL_DELTA_AXIS, DAYS_AXIS)
    put = boundary_masks(-a, T, PUT_DELTA_AXIS, DAYS_AXIS)
    for k in call:
        assert call[k].tolist() == put[k].tolist(), k


def test_in_grid_is_exactly_the_complement_of_the_union():
    rng = np.random.default_rng(0)
    delta = rng.uniform(0.0, 100.0, 500)
    T = rng.uniform(1.0, 900.0, 500) / 365.0
    m = boundary_masks(delta, T, CALL_DELTA_AXIS, DAYS_AXIS)
    union = m["below_10_delta"] | m["above_90_delta"] | m["tenor_out"]
    assert np.array_equal(m["in_grid"], ~union)
    # the two delta masks can never both be true
    assert not (m["below_10_delta"] & m["above_90_delta"]).any()
    # delta_in_grid ignores the tenor clamp, so it is a SUPERSET of in_grid
    assert np.array_equal(m["delta_in_grid"], ~(m["below_10_delta"] | m["above_90_delta"]))
    assert (m["in_grid"] <= m["delta_in_grid"]).all()
    assert m["delta_in_grid"].sum() > m["in_grid"].sum()


def test_delta_and_tenor_masks_overlap_and_are_counted_as_such():
    """A 5-day 3-delta row is clamped on BOTH axes -- the reason the three
    out-of-grid masks do not partition the split."""
    delta = np.array([3.0, 3.0, 50.0, 50.0])
    T = np.array([5.0, 100.0, 5.0, 100.0]) / 365.0
    m = boundary_masks(delta, T, CALL_DELTA_AXIS, DAYS_AXIS)
    assert m["below_10_delta"].tolist() == [True, True, False, False]
    assert m["tenor_out"].tolist() == [True, False, True, False]
    assert int((m["below_10_delta"] & m["tenor_out"]).sum()) == 1
    # counting the masks' rows double-counts that row
    triple = sum(int(m[k].sum()) for k in ("below_10_delta", "above_90_delta", "tenor_out"))
    assert triple == 4 and int((~m["in_grid"]).sum()) == 3


# ------------------------------------------------------------------ stats

def _toy():
    """4 rows with hand-checkable errors: 2 big log errors, 2 small."""
    log_target = np.array([-11.0, -10.0, -1.0, -0.5])
    log_pred = np.array([-7.0, -6.0, -1.1, -0.4])       # resid +4, +4, -0.1, +0.1
    T = np.full(4, MID_T)
    resid = log_pred - log_target
    sq_log = resid ** 2
    sq_price = (np.exp(log_pred) - np.exp(log_target)) ** 2
    rel = np.abs(np.exp(log_pred) - np.exp(log_target)) / np.exp(log_target)
    return log_pred, log_target, T, sq_log, sq_price, rel


def test_sse_shares_over_a_partition_sum_to_100():
    log_pred, log_target, T, sq_log, sq_price, rel = _toy()
    mask = np.array([True, True, False, False])
    a = _stats(mask, sq_log, sq_price, rel, log_pred, log_target, T)
    b = _stats(~mask, sq_log, sq_price, rel, log_pred, log_target, T)
    assert a["share_log_sse_pct"] + b["share_log_sse_pct"] == pytest.approx(100.0)
    assert a["share_price_sse_pct"] + b["share_price_sse_pct"] == pytest.approx(100.0)
    assert a["n_rows"] + b["n_rows"] == 4
    assert a["pct_rows"] + b["pct_rows"] == pytest.approx(100.0)


def test_share_is_the_masked_sse_over_the_total():
    log_pred, log_target, T, sq_log, sq_price, rel = _toy()
    mask = np.array([True, False, True, False])
    s = _stats(mask, sq_log, sq_price, rel, log_pred, log_target, T)
    assert s["share_log_sse_pct"] == pytest.approx(
        100.0 * sq_log[mask].sum() / sq_log.sum())
    assert s["share_price_sse_pct"] == pytest.approx(
        100.0 * sq_price[mask].sum() / sq_price.sum())


def test_within_mask_r2_uses_only_the_masked_rows():
    """R2(log) inside a mask is the R2 of that subset against its OWN mean, not a
    slice of the global one -- that is what makes 'in-grid rows reach R2=0.98'
    a statement about the in-grid subset."""
    rng = np.random.default_rng(3)
    log_target = rng.normal(-2.0, 1.5, 200)
    log_pred = log_target.copy()
    bad = np.zeros(200, bool)
    bad[:50] = True
    log_pred[bad] += 4.0                       # only the masked rows are wrong
    T = np.full(200, MID_T)
    resid = log_pred - log_target
    sq_log = resid ** 2
    sq_price = (np.exp(log_pred) - np.exp(log_target)) ** 2
    rel = np.abs(np.exp(log_pred) - np.exp(log_target)) / np.exp(log_target)

    good = _stats(~bad, sq_log, sq_price, rel, log_pred, log_target, T)
    assert good["r2_log"] == pytest.approx(1.0)          # perfect on its own rows
    assert good["share_log_sse_pct"] == pytest.approx(0.0)

    worst = _stats(bad, sq_log, sq_price, rel, log_pred, log_target, T)
    ss_res = float(np.sum(resid[bad] ** 2))
    ss_tot = float(np.sum((log_target[bad] - log_target[bad].mean()) ** 2))
    assert worst["r2_log"] == pytest.approx(1.0 - ss_res / (ss_tot + 1e-10))
    assert worst["share_log_sse_pct"] == pytest.approx(100.0)


def test_tail_counts():
    log_pred, log_target, T, sq_log, sq_price, rel = _toy()
    all_rows = np.ones(4, bool)
    s = _stats(all_rows, sq_log, sq_price, rel, log_pred, log_target, T)
    assert s["n_sqerr_log_gt10"] == 2           # 16, 16 > 10; 0.01, 0.01 not
    assert s["n_sqerr_log_gt1"] == 2
    # exp(+4) - 1 = 53.6x the true price, so both are > 100% relative error;
    # the two small rows move the price by ~10%, so neither is.
    assert s["n_rel_price_err_gt_100pct"] == 2


def test_tail_thresholds_are_strict():
    """Exactly 10 and exactly 1 squared log error, and exactly 100% relative
    price error, are NOT counted."""
    # built exactly rather than via sqrt/exp round-trips, which land 1 ulp off 10.0
    sq_log = np.array([10.0, 1.0, 0.5])
    rel = np.array([1.0, 1.0, 0.5])
    sq_price = np.array([1.0, 1.0, 1.0])
    log_target = np.zeros(3)
    log_pred = np.sqrt(sq_log)
    T = np.full(3, MID_T)
    s = _stats(np.ones(3, bool), sq_log, sq_price, rel, log_pred, log_target, T)
    assert s["n_sqerr_log_gt10"] == 0
    assert s["n_sqerr_log_gt1"] == 1            # the sq=10 row, not the sq=1 row
    assert s["n_rel_price_err_gt_100pct"] == 0


def test_empty_mask_reports_zeros_and_null_r2():
    log_pred, log_target, T, sq_log, sq_price, rel = _toy()
    s = _stats(np.zeros(4, bool), sq_log, sq_price, rel, log_pred, log_target, T)
    assert s["n_rows"] == 0 and s["pct_rows"] == 0.0
    assert s["share_log_sse_pct"] == 0.0 and s["share_price_sse_pct"] == 0.0
    assert s["r2_log"] is None and s["r2_price"] is None
