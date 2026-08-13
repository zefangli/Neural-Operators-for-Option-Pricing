"""`_common.compute_metrics` against synthetic data with hand-computable answers.

Run either way:
    conda run -n dl_new python -m pytest tests/ -q
    conda run -n dl_new python tests/test_metrics.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from _common import CLAMP_LOG, T_MIN_YEARS, compute_metrics  # noqa: E402

SAFE_T = np.float64(1.0)          # comfortably inside the T > 1/365 filter


def test_perfect_prediction():
    tgt = np.log(np.linspace(0.01, 0.5, 500))
    m = compute_metrics(tgt.copy(), tgt, np.full(500, SAFE_T))
    assert m["n_test"] == 500
    assert abs(m["r2_log_full"] - 1.0) < 1e-9
    assert abs(m["r2_price"] - 1.0) < 1e-9
    assert abs(m["r2_log_filtered"] - 1.0) < 1e-9
    assert m["rmse_log"] == 0.0 and m["rmse_price"] == 0.0 and m["mae_price"] == 0.0
    assert m["n_tail_sqerr_gt10"] == 0 and m["tail_sse_share"] == 0.0
    assert m["n_keep_filtered"] == 500 and m["n_near_expiry_dropped"] == 0


def test_mean_prediction_gives_r2_zero():
    """Predicting the target mean everywhere => R2(log) == 0 by construction."""
    rng = np.random.default_rng(0)
    tgt = rng.normal(-3.0, 1.0, 4000)
    m = compute_metrics(np.full_like(tgt, tgt.mean()), tgt, np.full(4000, SAFE_T))
    assert abs(m["r2_log_full"]) < 1e-9
    assert abs(m["r2_log_filtered"]) < 1e-9


def test_known_r2_rmse_mae():
    """Constant offset d on every row => SSE = n*d^2, R2 = 1 - n*d^2/SST."""
    rng = np.random.default_rng(1)
    tgt = rng.normal(-2.0, 0.5, 1000)
    d = 0.1
    m = compute_metrics(tgt + d, tgt, np.full(1000, SAFE_T))

    sst = float(np.sum((tgt - tgt.mean()) ** 2))
    assert abs(m["r2_log_full"] - (1.0 - 1000 * d ** 2 / sst)) < 1e-9
    assert abs(m["rmse_log"] - d) < 1e-12
    assert abs(m["mse_log"] - d ** 2) < 1e-12

    # price space: p_pred = p_true * exp(d)
    p = np.exp(tgt)
    assert abs(m["mae_price"] - float(np.mean(np.abs(p * np.expm1(d))))) < 1e-12
    assert abs(m["rmse_price"] - float(np.sqrt(np.mean((p * np.expm1(d)) ** 2)))) < 1e-12


def test_T_filter_boundary_is_strict():
    """The filter is `T > 1/365`, strictly. A row sitting exactly on 1/365 is
    DROPPED; a row one ulp above is KEPT. This matters because T is stored as
    float32 in the HDF5 and float32(1/365) upcasts ABOVE the float64 constant,
    so real 1-day contracts land on the KEPT side."""
    T = np.array([0.0, T_MIN_YEARS, np.nextafter(T_MIN_YEARS, 1.0),
                  float(np.float32(1.0 / 365.0)), 1.0])
    tgt = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
    m = compute_metrics(tgt.copy(), tgt, T)
    assert m["n_keep_filtered"] == 3, "expected T=nextafter, float32(1/365) and T=1 kept"
    assert m["n_near_expiry_dropped"] == 2, "expected T=0 and T=1/365 dropped"
    assert float(np.float32(1.0 / 365.0)) > T_MIN_YEARS, (
        "float32 boundary assumption broke; the headline metric's row set changed")


def test_filter_isolates_near_expiry_damage():
    """A handful of catastrophic T==0 rows can wreck R2(log,full) while leaving
    R2(log, T>1/365) at 1.0 -- the whole reason the paper reports both."""
    n = 10000
    rng = np.random.default_rng(2)
    tgt = rng.normal(-3.0, 1.0, n)
    pred = tgt.copy()
    T = np.full(n, SAFE_T)
    bad = slice(0, 100)
    T[bad] = 0.0
    pred[bad] = CLAMP_LOG                       # log(1e-8) = -18.4207, what the clamp produces

    m = compute_metrics(pred, tgt, T)
    assert abs(m["r2_log_filtered"] - 1.0) < 1e-9
    assert m["r2_log_full"] < 0.0, "100 clamped rows should dominate the full-domain R2"
    assert m["n_near_expiry_dropped"] == 100
    assert m["n_tail_sqerr_gt10"] == 100
    assert abs(m["tail_sse_share"] - 1.0) < 1e-9
    # clamp_hit_count uses a hard `< -18.42`, which log(1e-8) = -18.4207 clears
    assert m["clamp_hit_count"] == 100
    assert CLAMP_LOG < -18.42, "clamp_hit_count's hardcoded -18.42 no longer catches the clamp"


def test_empty_filtered_set_is_nan_not_crash():
    T = np.zeros(50)
    tgt = np.linspace(-5, -1, 50)
    m = compute_metrics(tgt.copy(), tgt, T)
    assert m["n_keep_filtered"] == 0
    assert np.isnan(m["r2_log_filtered"]) and np.isnan(m["mse_log_filtered"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS %s" % f.__name__)
    print("%d/%d passed" % (len(fns), len(fns)))
