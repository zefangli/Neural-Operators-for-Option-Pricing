"""
CPU-only, per-branch baselines for the paper.

No training, no checkpoints, no GPU: every number here comes from the raw
CSVs (WRDS + Cboe VIX) + the row-aligned HDF5 tensors, priced through the SAME
analytical Black-Scholes formula the model uses (`_common.bs_normalized_price`)
and scored through the SAME metric schema (`_common.compute_metrics`).

    conda run --no-capture-output -n dl_new python analysis/baselines.py --dataset-version v4 --self-test
    conda run --no-capture-output -n dl_new python analysis/baselines.py --dataset-version v4
    conda run --no-capture-output -n dl_new python analysis/baselines.py --dataset-version v4 --max-contracts 50000

`--dataset-version` is REQUIRED (v3 / v4 / v4smoke): the two samples are not
comparable and must never be silently mixed. Outputs are version-tagged --
v3 keeps `results/baselines_vix.{json,csv}`, v4 writes
`results/baselines_v4.{json,csv}` -- and every output records the resolved HDF5
path and its `schema_version`.  The VVIX-based predecessor results live in
results/baselines_vvix_legacy.{json,csv}.

v4 additions used automatically when the file carries them (guarded by a
presence check, so a v3 file still runs):
  * `impl_volatility` -- OptionMetrics' own per-contract IV. Preferred as the
    "observed IV" for the iv_rmse/iv_bias columns; the numerical inversion is
    still run and the two are cross-reported (coverage + disagreement).
  * `vix_level` -- per-row VIX close. Preferred over re-reading the CSV, and
    asserted to agree with the CSV join to 1e-5 (in close/100 units).

Design notes / conventions (all of these are reported in the JSON `notes`):
  * Every fitted parameter is fitted on split_id==0 (train) rows with T>0 only,
    on a fixed-seed subsample, and applied unchanged to the test split.
  * The IV surface is rebuilt from volatilitySurface_2015_2025.csv rather than
    read out of `branch_u`, because (a) it is keyed by date so it is ~2700x
    smaller, and (b) it gives the *previous* day's surface for free. The
    self-test asserts the rebuilt grid equals `branch_u` row-for-row.
  * Grid axes are derived from the CSV, not assumed: days = 11 tenors,
    delta = 17 points (+10..+90 for calls, -90..-10 for puts, in delta-percent),
    flattened days-major / delta-minor (that is the pipeline's sort order).
  * moneyness -> delta needs sigma and sigma comes from the delta lookup, so the
    surface lookup is a damped fixed point on delta = exp(-qT)*N(d1) (call) /
    -exp(-qT)*N(-d1) (put). Non-convergence and out-of-grid rates are reported.
    Out-of-grid queries are CLAMPED to the boundary (never extrapolated).
  * T == 0 rows are priced as discounted intrinsic for every baseline, because
    BS with T=0 is intrinsic for any sigma (and d1 is 0/0 at the money). So the
    T==0 stratum is identical across all baselines by construction.
  * Every record carries BOTH a `branch` (which neural input branch, if any, this
    baseline is a legitimate comparator for) and an `input_kind` (what the
    baseline's own input actually is). They are not the same question: the
    scalar-VIX baselines are `branch=None` / `input_kind="scalar market
    observable"` -- one VIX close per date, which is NOT the 21x4 close-normalised
    `vix_history` branch tensor. See BRANCH_LABELS below.
  * Every record carries `price_accurate` (R2(price) > 0.99) and a
    `ranking_caveat`, so a reader of the CSV alone cannot rank baselines on
    R2(log) while ignoring a catastrophic R2(price) (this is exactly what
    happens on puts).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402  (path shim above)
    PROJECT_ROOT,
    T_MIN_YEARS,
    add_dataset_arg,
    bs_normalized_price,
    compute_metrics,
    dataset_h5,
    dataset_provenance,
    liquidity_masks,
    output_tag,
)

DATA_DIR = PROJECT_ROOT / "wrds_data_2020-2025"
RESULTS_DIR = PROJECT_ROOT / "results"
SECID = 108105
N_DAYS_AXIS, N_DELTA_AXIS = 11, 17
LOOKBACK = 21
TRADING_DAYS = 252.0
FIT_SUBSAMPLE = 300_000
FIT_SEED = 0
CLAMP_MIN = 1e-8          # matches the training-time log clamp
SIGMA_LO, SIGMA_HI = 1e-4, 5.0

PRICE_ACCURATE_MIN = 0.99   # R2(price) above this = "price-accurate" family

# ---- baseline classification --------------------------------------------
# `branch`      : the neural input branch this baseline is a LEGITIMATE comparator
#                 for -- i.e. it is built from the same information that branch
#                 sees. None means "no branch"; such a baseline is a reference
#                 point, never evidence about a branch.
# `input_kind`  : what the baseline's own input is.
BRANCH_COMPARABLE = "branch-comparable"
SCALAR_OBSERVABLE = "scalar market observable (not a branch input)"
CONTEXT_FREE = "context-free reference (no market input)"
LAGGED_REFERENCE = "lagged market observable (not a branch input)"

# The sentence that must travel with the scalar-VIX numbers, in the data, not
# only in a comment.
VIX_ANTI_INFERENCE = (
    "ANTI-INFERENCE: this is a contemporaneous scalar VIX benchmark (one index "
    "close per date) and NOT the model's `vix_history` branch (a 21x4 OHLC "
    "tensor, close-normalised so the absolute VIX level is destroyed), so it "
    "cannot be used to rank the neural input branches against each other."
)

BRANCH_LABELS_DOC = {
    "branch": "the neural input branch this baseline is a legitimate comparator for "
              "(built from the same information that branch sees); null = none, in "
              "which case the baseline is a context-free reference point and says "
              "nothing about any branch",
    "input_kind": "what the baseline's own input actually is: '%s' | '%s' | '%s' | '%s'"
                  % (BRANCH_COMPARABLE, SCALAR_OBSERVABLE, LAGGED_REFERENCE, CONTEXT_FREE),
    "price_accurate": "R2(price) > %g on the 'all' stratum" % PRICE_ACCURATE_MIN,
    "why": "the scalar-VIX baselines were previously tagged branch='vix_history', which "
           "invites the invalid inference that the VIX-history branch beats the "
           "vol-surface branch. Different input, different information content.",
}

LIQ_STRATA = ("liq_bid_pos", "liq_relspread_le1", "liq_bid_pos_and_relspread_le1")
STRATA = ("all", "T_eq_0", "T_gt_0_le_1d", "T_gt_1d", "T_eq_1d") + LIQ_STRATA
STRATA_DOC = {
    "liq_bid_pos": "REPORTING stratum: best_bid > 0. Shows the results are not carried "
                   "by retained zero-bid quotes. NOT a filter -- no row is dropped.",
    "liq_relspread_le1": "REPORTING stratum: (ask-bid)/mid <= 1 (bounded by 2 when "
                         "bid >= 0, so thresholds above 2 are vacuous). NOT a filter.",
    "liq_bid_pos_and_relspread_le1": "REPORTING stratum: both liquidity conditions. "
                                     "NOT a filter -- dropping rows on liquidity would "
                                     "be a separate dataset decision, not taken.",
    "_liquidity_strata_are_not_a_filter":
        "The liq_* strata subset the REPORT only. Every metric outside them, and the "
        "sample itself, still contains every retained quote.",
    "_liquidity_strata_availability":
        "Empty (n_test=0) on v3, which has no best_bid/best_offer datasets.",
    "all": "the whole test split",
    "T_eq_0": "T == 0 (expiry-day contracts). BS is intrinsic here for ANY sigma, so every "
              "baseline scores identically in this stratum by construction.",
    "T_gt_0_le_1d": "0 < T <= 1/365. EMPTY in this dataset: T = days_to_expiry/365 is "
                    "quantised, and the float32-stored 1-day value (0.00273973) is just ABOVE "
                    "1/365 (0.00273972602...), so 1-day contracts fall in T_gt_1d. This is the "
                    "same comparison _common.compute_metrics makes, so the model's headline "
                    "R2(log, T>1/365) also includes the 1-day contracts.",
    "T_gt_1d": "T > 1/365 -- the headline log-metric domain (see CLAUDE.md).",
    "T_eq_1d": "extra, non-exclusive: round(T*365) == 1, i.e. the 1-day contracts that "
               "T_gt_0_le_1d would have held if T were exact.",
}


# ---------------------------------------------------------------- primitives

def _t(a):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64))


def intrinsic(log_m, T, r, q, option_type):
    """Discounted intrinsic in V/K units (the zero-vol limit of BS)."""
    fwd = np.exp(log_m - q * T)
    disc = np.exp(-r * T)
    return np.maximum(fwd - disc, 0.0) if option_type == "call" else np.maximum(disc - fwd, 0.0)


def bs_price(log_m, T, r, q, sigma, option_type):
    """V/K via _common's pricer, float64, with T<=0 -> intrinsic (BS is 0/0 there)."""
    sig = np.where(np.isfinite(sigma), sigma, np.nan)
    p = bs_normalized_price(_t(log_m), _t(T), _t(r), _t(q), _t(np.nan_to_num(sig, nan=0.2)),
                            option_type).numpy()
    p = np.where(np.isfinite(sig), p, np.nan)
    z = np.asarray(T) <= 0
    if z.any():
        p = np.where(z, intrinsic(log_m, T, r, q, option_type), p)
    return np.maximum(p, 0.0)


def log_price(*args, **kw):
    return np.log(np.clip(bs_price(*args, **kw), CLAMP_MIN, None))


def bs_delta_pct(log_m, T, r, q, sigma, option_type):
    """Black-Scholes spot delta x100 -- the units of the OptionMetrics surface axis."""
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    s = np.maximum(sigma, 1e-10)
    d1 = (log_m + (r - q + 0.5 * s * s) * T) / (s * sqrt_T)
    disc = np.exp(-q * T)
    return 100.0 * (disc * ndtr(d1) if option_type == "call" else -disc * ndtr(-d1))


# ---------------------------------------------------------------- data loading

def date_span(dates):
    """(first, last, n_unique) -- numpy 2 has no min/max ufunc for U dtype."""
    u = np.unique(dates)
    return str(u[0]), str(u[-1]), int(len(u))


# Which sample every load_split() call reads. Set once by main()/self_test();
# a module global rather than a threaded argument because load_split is called
# from fit_on_train's closures and from the self-test, and the whole process
# must read exactly one sample.
# ponytail: global, because "one process = one dataset version" is the invariant.
_DATASET = {"version": None, "h5_dir": None}


def h5_path(option_type):
    if _DATASET["version"] is None:
        raise RuntimeError("dataset version not selected; pass --dataset-version")
    return dataset_h5(_DATASET["version"], option_type, _DATASET["h5_dir"])


def load_split(option_type, split, max_contracts=None, seed=42, with_branch_u=False):
    """Minimal row-wise loader (no branch_u by default -- 187 floats/row is 1 GB).

    v4-only columns (`impl_volatility`, `vix_level`) are loaded when present.
    """
    want = {"train": 0, "val": 1, "test": 2}[split]
    with h5py.File(h5_path(option_type), "r") as f:
        sid = f["split_id"][:]
        idx = np.where(sid == want)[0]
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        assert hi - lo == len(idx), "split rows are not contiguous; slicing assumption broken"
        trunk = f["trunk_y"][lo:hi].astype(np.float64)
        out = {
            "log_m": trunk[:, 0], "T": trunk[:, 1], "r": trunk[:, 2], "q": trunk[:, 3],
            "target_v_log": f["target_v_log"][lo:hi].astype(np.float64).ravel(),
            "date": f["date"][lo:hi].astype("U10"),
            "row0": lo,
        }
        # v4/v5-only columns, presence-checked so v3 still loads. best_bid/best_offer
        # feed the quote-liquidity REPORTING strata -- no row is dropped for them.
        for opt in ("impl_volatility", "vix_level", "best_bid", "best_offer",
                    "spread_norm", "half_spread_norm"):
            if opt in f:
                out[opt] = f[opt][lo:hi].astype(np.float64).ravel()
        if with_branch_u:
            out["branch_u"] = f["branch_u"][lo:hi]
    n = len(out["T"])
    if max_contracts is not None and n > max_contracts:
        keep = np.sort(np.random.default_rng(seed).choice(n, max_contracts, replace=False))
        out = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in out.items()}
        out["sample"] = {"max_contracts": int(max_contracts), "seed": int(seed), "of": int(n)}
    return out


def load_surfaces():
    """-> dict cp -> {dates, days, delta, grids (n_dates, 11, 17)}; axes derived, not assumed."""
    df = (
        pl.scan_csv(DATA_DIR / "volatilitySurface_2015_2025.csv")
        .filter(pl.col("secid") == SECID)
        .select("date", "cp_flag", "days", "delta", "impl_volatility")
        .collect()
        .sort(["date", "cp_flag", "days", "delta"])   # same key order as pre_process_1_data.py
    )
    out = {}
    for cp, key in (("C", "call"), ("P", "put")):
        d = df.filter(pl.col("cp_flag") == cp)
        dates = d["date"].unique().sort().to_numpy().astype("U10")
        n = len(dates)
        assert d.height == n * N_DAYS_AXIS * N_DELTA_AXIS, f"{key}: ragged surface"
        days = d["days"].to_numpy().reshape(n, N_DAYS_AXIS, N_DELTA_AXIS)
        delta = d["delta"].to_numpy().reshape(n, N_DAYS_AXIS, N_DELTA_AXIS)
        days_ax = days[0, :, 0].astype(np.float64)
        delta_ax = delta[0, 0, :].astype(np.float64)
        # days-major / delta-minor: days constant along the last axis, delta along the middle
        assert (days == days_ax[None, :, None]).all(), f"{key}: tenor axis varies by date"
        assert (delta == delta_ax[None, None, :]).all(), f"{key}: delta axis varies by date"
        assert np.all(np.diff(days_ax) > 0) and np.all(np.diff(delta_ax) > 0)
        grids = d["impl_volatility"].to_numpy().reshape(n, N_DAYS_AXIS, N_DELTA_AXIS)
        assert np.isfinite(grids).all(), f"{key}: NaN in surface"
        out[key] = {"dates": dates, "days": days_ax, "delta": delta_ax,
                    "grids": np.ascontiguousarray(grids, dtype=np.float64)}
    return out


def spx_vol_features(lam):
    """-> dict date -> (realized_vol_21d, ewma_vol) annualised with sqrt(252)."""
    d = (pl.scan_csv(DATA_DIR / "spx_price_2015_2025.csv")
         .filter(pl.col("secid") == SECID).select("date", "close").collect().sort("date"))
    dates = d["date"].to_numpy().astype("U10")
    lr = np.diff(np.log(d["close"].to_numpy()))
    w = lam ** np.arange(LOOKBACK - 2, -1, -1)          # oldest .. newest
    w = w / w.sum()
    out = {}
    for i in range(LOOKBACK - 1, len(dates)):
        r = lr[i - LOOKBACK + 1:i]                       # 20 returns in the 21-close window
        out[dates[i]] = (float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)),
                         float(np.sqrt(np.sum(w * r * r) * TRADING_DAYS)))
    return out


def vix_levels():
    """CBOE SPX VIX daily close -> dict date -> close.

    Reads the wide-format Cboe CSV (vixo/vixh/vixl/vix/vxno/.../vxd) and keeps
    only `vix`. The VXN (Nasdaq-100) and VXD (DJIA) columns are intentionally
    unused. Zero/negative closes are filtered out (none in practice)."""
    d = (pl.scan_csv(DATA_DIR / "vix_2015_2025.csv")
         .select("date", pl.col("vix").alias("close"))
         .filter(pl.col("close") > 0)
         .collect().sort("date"))
    return dict(zip(d["date"].to_numpy().astype("U10"), d["close"].to_numpy()))


# ---------------------------------------------------------------- surface lookup

def _interp(surf, method):
    # ponytail: sigma is interpolated linearly in (calendar days, delta-pct); no
    # total-variance-in-T reparametrisation. Upgrade there if the tenor edge matters.
    return lambda grid: RegularGridInterpolator(
        (surf["days"], surf["delta"]), grid, method=method,
        bounds_error=False, fill_value=None)


def surface_sigma_group(interp_fn, grid, surf, log_m, T, r, q, option_type, method,
                        tol=1e-6, max_iter=80, damping=0.7):
    """Damped fixed point sigma = surface(days, delta(sigma)). Returns (sigma, converged, oob)."""
    f = interp_fn(grid)
    days_q = np.clip(T * 365.0, surf["days"][0], surf["days"][-1])
    d_lo, d_hi = surf["delta"][0], surf["delta"][-1]
    atm = np.full_like(days_q, 50.0 if option_type == "call" else -50.0)
    sigma = np.clip(f(np.stack([days_q, atm], -1)), SIGMA_LO, SIGMA_HI)
    step = np.full_like(sigma, np.inf)
    for _ in range(max_iter):
        dpct = bs_delta_pct(log_m, T, r, q, sigma, option_type)
        new = np.clip(f(np.stack([days_q, np.clip(dpct, d_lo, d_hi)], -1)), SIGMA_LO, SIGMA_HI)
        step = damping * (new - sigma)
        sigma = sigma + step
        if np.abs(step).max() < tol:
            break
    dpct = bs_delta_pct(log_m, T, r, q, sigma, option_type)
    oob_d = (dpct < d_lo) | (dpct > d_hi)
    oob_t = (T * 365.0 < surf["days"][0]) | (T * 365.0 > surf["days"][-1])
    return sigma, np.abs(step) < tol, oob_d, oob_t


def surface_sigma(rows, surf, option_type, method="linear", lag=0):
    """Per-date grouped surface lookup. `lag=1` uses the previous available trade date."""
    interp_fn = _interp(surf, method)
    sigma = np.full(len(rows["T"]), np.nan)
    conv = np.zeros(len(sigma), bool)
    oob_d = np.zeros(len(sigma), bool)
    oob_t = np.zeros(len(sigma), bool)
    missing = 0
    uniq, inv = np.unique(rows["date"], return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bnd = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    for i, d in enumerate(uniq):
        sel = order[bnd[i]:bnd[i + 1]]
        j = int(np.searchsorted(surf["dates"], d))
        assert j < len(surf["dates"]) and surf["dates"][j] == d, f"no surface for {d}"
        j -= lag
        if j < 0:
            missing += len(sel)
            continue
        s, c, od, ot = surface_sigma_group(
            interp_fn, surf["grids"][j], surf, rows["log_m"][sel], rows["T"][sel],
            rows["r"][sel], rows["q"][sel], option_type, method)
        sigma[sel], conv[sel], oob_d[sel], oob_t[sel] = s, c, od, ot
    live = rows["T"] > 0                      # T==0 never uses sigma
    nl = max(int(live.sum()), 1)
    diag = {"nonconvergence_rate": float((~conv & live).sum() / nl),
            "n_nonconverged": int((~conv & live).sum()),
            "out_of_grid_rate": float(((oob_d | oob_t) & live).sum() / nl),
            "n_out_of_grid": int(((oob_d | oob_t) & live).sum()),
            "n_out_of_grid_delta": int((oob_d & live).sum()),
            "n_out_of_grid_tenor": int((oob_t & live).sum()),
            "n_live_rows": int(live.sum()),
            "n_missing_surface": int(missing),
            "clamped": "out-of-grid queries clamped to the nearest grid boundary"}
    return sigma, diag


# ---------------------------------------------------------------- fitting (train only)

def _fit_scalar(objective, lo, hi):
    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x), float(res.fun)


def fit_on_train(option_type, sigma_of, lo, hi, space):
    """Fit one scalar on a fixed-seed TRAIN subsample (T>0 rows only).

    `space="log"` minimises MSE on log(V/K) -- the model's own training loss.
    `space="price"` minimises MSE on V/K -- the space R2(price) is reported in.
    Both are given because for puts the two disagree violently (the log objective
    is owned by the deep-OTM wing, the price objective by the deep-ITM wing)."""
    tr = load_split(option_type, "train", max_contracts=FIT_SUBSAMPLE, seed=FIT_SEED)
    keep = tr["T"] > 0
    tr = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
    base = sigma_of(tr)
    if base is not None:
        ok = np.isfinite(base)
        tr = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
        base = base[ok]
    target_px = np.exp(tr["target_v_log"])

    def obj(x):
        s = x * base if base is not None else np.full(len(tr["T"]), x)
        if space == "log":
            lp = log_price(tr["log_m"], tr["T"], tr["r"], tr["q"], s, option_type)
            return float(np.mean((lp - tr["target_v_log"]) ** 2))
        px = bs_price(tr["log_m"], tr["T"], tr["r"], tr["q"], s, option_type)
        return float(np.mean((px - target_px) ** 2))

    x, fun = _fit_scalar(obj, lo, hi)
    return x, {"fitted_on": "split_id==0 (train)", "n_fit_rows": int(len(tr["T"])),
               "subsample": FIT_SUBSAMPLE, "subsample_seed": FIT_SEED,
               "objective": f"MSE on {'log(V/K)' if space == 'log' else 'V/K'}, T>0 rows",
               "objective_value": fun}


# ---------------------------------------------------------------- observed IV

def bs_vega(log_m, T, r, q, sigma, option_type):
    """d(V/K)/dsigma (same for calls and puts)."""
    sqrt_T = np.sqrt(np.maximum(T, 1e-10))
    s = np.maximum(sigma, 1e-10)
    d1 = (log_m + (r - q + 0.5 * s * s) * T) / (s * sqrt_T)
    return np.exp(log_m - q * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * sqrt_T


def implied_vol(log_m, T, r, q, price, option_type, iters=64, vega_min=1e-4):
    """Bisection inversion of the BS pricer.

    NaN where T==0 or the price is outside the no-arbitrage band
    [sigma->0 price, sigma=SIGMA_HI price]. Also NaN where vega at the solution is
    below `vega_min`: deep OTM/ITM prices carry no sigma information (the inverse
    problem is the same ill-conditioning the near-expiry tail suffers from), so an
    "implied vol" there would be numerical noise rather than an observation."""
    lo = np.full_like(T, SIGMA_LO)
    hi = np.full_like(T, SIGMA_HI)
    p_lo = bs_price(log_m, T, r, q, lo, option_type)
    p_hi = bs_price(log_m, T, r, q, hi, option_type)
    in_band = (price > p_lo) & (price < p_hi)
    ok = (T > 0) & in_band
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_low = bs_price(log_m, T, r, q, mid, option_type) < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    iv = 0.5 * (lo + hi)
    low_vega = bs_vega(log_m, T, r, q, iv, option_type) <= vega_min
    ok &= ~low_vega
    n = len(T)
    info = {"n": n, "n_ok": int(ok.sum()), "n_failed": int((~ok).sum()),
            "failure_rate": float((~ok).mean()),
            "n_failed_T_eq_0": int((T == 0).sum()),
            "n_failed_out_of_band": int(((T > 0) & ~in_band).sum()),
            "n_failed_low_vega": int(((T > 0) & in_band & low_vega).sum()),
            "vega_min": vega_min, "bisection_iters": iters,
            "bracket": [SIGMA_LO, SIGMA_HI]}
    return np.where(ok, iv, np.nan), info


# ------------------------------------------------- v4: stored IV / VIX columns

def choose_observed_iv(rows, iv_inv):
    """Pick the observed IV: OptionMetrics' stored value (v4) over the inversion.

    Returns (iv_obs, info). Where the stored IV is missing (OptionMetrics leaves
    it null for illiquid / bound-violating quotes, which arrive here as NaN or
    <=0) the inverted value is used, so coverage is never *reduced* by preferring
    it. The disagreement between the two is reported on the overlap.
    """
    stored = rows.get("impl_volatility")
    if stored is None:
        return iv_inv, {"preferred": "inverted",
                        "reason": "no `impl_volatility` dataset (v3 schema)",
                        "coverage_pct": 100.0 * float(np.isfinite(iv_inv).mean())}
    ok_s = np.isfinite(stored) & (stored > 0)
    ok_i = np.isfinite(iv_inv)
    iv_obs = np.where(ok_s, stored, iv_inv)
    both = ok_s & ok_i
    e = stored[both] - iv_inv[both]
    return iv_obs, {
        "preferred": "stored OptionMetrics impl_volatility (falling back to the "
                     "inversion where it is missing)",
        "coverage_stored_pct": 100.0 * float(ok_s.mean()),
        "coverage_inverted_pct": 100.0 * float(ok_i.mean()),
        "coverage_combined_pct": 100.0 * float((ok_s | ok_i).mean()),
        "n_compared": int(both.sum()),
        "disagreement_vs_inverted": {
            "rmse": float(np.sqrt(np.mean(e ** 2))) if both.any() else None,
            "bias_stored_minus_inverted": float(np.mean(e)) if both.any() else None,
            "median_abs": float(np.median(np.abs(e))) if both.any() else None,
            "p95_abs": float(np.percentile(np.abs(e), 95)) if both.any() else None,
            "max_abs": float(np.max(np.abs(e))) if both.any() else None,
        },
        "note": "the inversion is NaN wherever vega is uninformative (deep wings) or "
                "the quote violates the no-arbitrage band, so a large p95 there is "
                "expected and is a property of those rows, not of either source",
    }


VIX_AGREEMENT_TOL = 1e-5     # in close/100 units, i.e. 1e-3 VIX points


def vix_sigma(rows, vix_by_date):
    """Per-row VIX/100. Prefers the stored `vix_level` column (v4) over the CSV
    join, and asserts the two agree -- a silent mismatch would mean the HDF5 and
    the CSV describe different series (exactly the VVIX mislabel of 2026-08-11)."""
    csv = np.array([vix_by_date[d] / 100.0 for d in rows["date"]])
    stored = rows.get("vix_level")
    if stored is None:
        return csv, {"source": "vix_2015_2025.csv joined on date (no `vix_level` dataset)"}
    s = stored / 100.0 if np.nanmedian(stored) > 1.0 else stored.copy()
    dmax = float(np.nanmax(np.abs(s - csv)))
    assert dmax < VIX_AGREEMENT_TOL, (
        "stored vix_level disagrees with vix_2015_2025.csv by %.3g (> %.0e): the HDF5 "
        "and the CSV are not the same series" % (dmax, VIX_AGREEMENT_TOL))
    return s, {"source": "HDF5 `vix_level` dataset (per row)",
               "cross_check": "agrees with vix_2015_2025.csv joined on date",
               "max_abs_diff_vs_csv": dmax,
               "stored_units": "index points" if np.nanmedian(stored) > 1.0 else "already /100"}


# ---------------------------------------------------------------- scoring

def score(rows, sigma, option_type, iv_obs):
    """-> {stratum: metrics}. sigma=None means the zero-vol / intrinsic baseline."""
    T = rows["T"]
    if sigma is None:
        price = intrinsic(rows["log_m"], T, rows["r"], rows["q"], option_type)
        lp = np.log(np.clip(price, CLAMP_MIN, None))
    else:
        lp = log_price(rows["log_m"], T, rows["r"], rows["q"], sigma, option_type)
    assert np.isfinite(lp).all(), "non-finite predicted log price"
    masks = {"all": np.ones(len(T), bool), "T_eq_0": T == 0,
             "T_gt_0_le_1d": (T > 0) & (T <= T_MIN_YEARS), "T_gt_1d": T > T_MIN_YEARS,
             "T_eq_1d": np.round(T * 365.0) == 1}
    # Quote-liquidity REPORTING strata (v4/v5 only; empty on v3). These select rows
    # for the metric report and nothing else -- `rows` is never subset by them.
    for name, m in liquidity_masks(rows.get("best_bid"), rows.get("best_offer")).items():
        masks["liq_" + name] = m
    for name in LIQ_STRATA:
        masks.setdefault(name, np.zeros(len(T), bool))
    out = {}
    for name, m in masks.items():
        if not m.any():
            out[name] = {"n_test": 0}
            continue
        met = compute_metrics(lp[m], rows["target_v_log"][m], T[m])
        met["date_min"], met["date_max"], met["n_dates"] = date_span(rows["date"][m])
        if sigma is not None:
            good = m & np.isfinite(iv_obs) & np.isfinite(sigma)
            met["n_iv_compared"] = int(good.sum())
            if good.any():
                e = sigma[good] - iv_obs[good]
                met["iv_rmse"] = float(np.sqrt(np.mean(e ** 2)))
                met["iv_bias"] = float(np.mean(e))
                met["iv_mae"] = float(np.mean(np.abs(e)))
                met["iv_median_abs"] = float(np.median(np.abs(e)))
                met["iv_p95_abs"] = float(np.percentile(np.abs(e), 95))
            else:
                met.update({k: None for k in
                            ("iv_rmse", "iv_bias", "iv_mae", "iv_median_abs", "iv_p95_abs")})
        out[name] = met
    return out


# ---------------------------------------------------------------- self test

def self_test():
    print("self-test against %s / %s" % (_DATASET["version"], h5_path("call")))
    rng = np.random.default_rng(7)
    n = 2000
    log_m = rng.uniform(-0.5, 0.5, n); T = rng.uniform(0.01, 2.0, n)
    r = rng.uniform(0.0, 0.06, n); q = rng.uniform(0.0, 0.04, n); s = rng.uniform(0.05, 0.9, n)
    c = bs_price(log_m, T, r, q, s, "call"); p = bs_price(log_m, T, r, q, s, "put")
    parity = np.exp(log_m - q * T) - np.exp(-r * T)
    assert np.abs(c - p - parity).max() < 1e-10, "put-call parity"

    iv, _ = implied_vol(log_m, T, r, q, c, "call")
    assert np.nanmax(np.abs(iv - s)) < 1e-6, "IV inversion round-trip"
    assert np.isfinite(iv).mean() > 0.9, "IV inversion rejected too much of a clean sample"

    surfaces = load_surfaces()
    for ot, cp in (("call", "C"), ("put", "P")):
        surf = surfaces[ot]
        ax = surf["delta"]
        assert len(surf["days"]) * len(ax) == 187
        assert (ax > 0).all() if ot == "call" else (ax < 0).all()

        # (a) rebuilt grid == branch_u, i.e. the days-major/delta-minor reshape is right
        rows = load_split(ot, "test", max_contracts=64, seed=3, with_branch_u=True)
        for k in range(len(rows["T"])):
            j = int(np.searchsorted(surf["dates"], rows["date"][k]))
            assert np.abs(surf["grids"][j].ravel()
                          - rows["branch_u"][k].astype(np.float64)).max() < 2e-6, "reshape order"

        # (b) flat synthetic surface -> fixed point returns it exactly
        flat = np.full((N_DAYS_AXIS, N_DELTA_AXIS), 0.25)
        sig, conv, _, _ = surface_sigma_group(_interp(surf, "linear"), flat, surf,
                                           log_m[:200], T[:200], r[:200], q[:200], ot, "linear")
        assert conv.all() and np.abs(sig - 0.25).max() < 1e-8, "flat-surface fixed point"

        # (c) linear-in-delta synthetic surface vs an independent brentq solve
        a, b = 0.30, -0.0015 if ot == "call" else 0.0015
        lin = a + b * np.tile(ax, (N_DAYS_AXIS, 1))
        sig, conv, _, _ = surface_sigma_group(_interp(surf, "linear"), lin, surf,
                                           log_m[:200], T[:200], r[:200], q[:200], ot, "linear")
        assert conv.all(), "linear-surface fixed point did not converge"
        for k in range(0, 200, 17):
            g = lambda x: a + b * np.clip(bs_delta_pct(log_m[k], T[k], r[k], q[k], x, ot),
                                          ax[0], ax[-1]) - x
            assert abs(sig[k] - brentq(g, SIGMA_LO, SIGMA_HI, xtol=1e-12)) < 1e-5, "fixed point"

        # (d) real data: OptionMetrics' own impl_strike must reproduce the grid delta.
        #     Pinned to one date, so it is skipped when that date is not in the
        #     sample under test (the smoke files cover 2019-11..2020-02 only).
        d0 = load_split(ot, "test", max_contracts=20000, seed=11)   # r,q from the aligned HDF5
        near = (d0["date"] == "2024-08-08") & (np.abs(d0["T"] * 365 - 30) < 4)
        if not near.any():
            print("  (d) skipped for %s: 2024-08-08 not in this sample's test split" % ot)
        else:
            _delta_convention_check(ot, cp, d0, near)

        # (e) spot_history is close-normalised, but log-returns are scale-invariant:
        #     the 21-close window must reproduce the raw SPX log-returns exactly.
        first = load_split(ot, "test", max_contracts=None)
        px = (pl.scan_csv(DATA_DIR / "spx_price_2015_2025.csv")
              .filter(pl.col("secid") == SECID).select("date", "close").collect().sort("date"))
        px_dates = px["date"].to_numpy().astype("U10")
        px_close = px["close"].to_numpy()
        with h5py.File(h5_path(ot), "r") as f:
            hist = f["spot_history"][first["row0"]].reshape(LOOKBACK, 5)[:, 3].astype(np.float64)
        i = int(np.searchsorted(px_dates, first["date"][0]))
        assert px_dates[i] == first["date"][0]
        raw_win = px_close[i - LOOKBACK + 1:i + 1]
        assert abs(hist[-1] - 1.0) < 1e-6, "spot_history not normalised by last close"
        assert np.abs(np.diff(np.log(hist)) - np.diff(np.log(raw_win))).max() < 1e-6, \
            "spot_history log-returns differ from the raw SPX window"

        # (f) v4 only: the stored vix_level must equal the CSV series row for row
        if "vix_level" in first:
            _, info = vix_sigma(first, vix_levels())
            print("  (f) %s: stored vix_level agrees with the CSV (max diff %.2e)"
                  % (ot, info["max_abs_diff_vs_csv"]))
    print("self-test OK")


def _delta_convention_check(ot, cp, d0, near):
    """OptionMetrics' own impl_strike must reproduce the grid delta on 2024-08-08."""
    raw = (pl.scan_csv(DATA_DIR / "volatilitySurface_2015_2025.csv")
               .filter((pl.col("secid") == SECID) & (pl.col("cp_flag") == cp)
                       & (pl.col("date") == "2024-08-08") & (pl.col("days") == 30))
           .select("delta", "impl_volatility", "impl_strike").collect())
    px = (pl.scan_csv(DATA_DIR / "spx_price_2015_2025.csv")
          .filter(pl.col("secid") == SECID).select("date", "close").collect().sort("date"))
    px_dates = px["date"].to_numpy().astype("U10")
    spot = float(px["close"].to_numpy()[px_dates == "2024-08-08"][0])
    rr, qq = float(np.median(d0["r"][near])), float(np.median(d0["q"][near]))
    got = bs_delta_pct(np.log(spot / raw["impl_strike"].to_numpy()), 30 / 365.0, rr, qq,
                       raw["impl_volatility"].to_numpy(), ot)
    err = np.abs(got - raw["delta"].to_numpy()).max()
    assert err < 2.5, f"{ot}: delta convention off by {err:.2f} delta points"


# ---------------------------------------------------------------- driver

def build_records(option_type, surfaces, args, prov):
    surf = surfaces[option_type]
    rows = load_split(option_type, "test", max_contracts=args.max_contracts, seed=args.seed)
    n = len(rows["T"])
    d_lo, d_hi, n_dates = date_span(rows["date"])
    print(f"[{option_type}] test rows={n} dates={n_dates} {d_lo}..{d_hi}")
    print(f"[{option_type}] sample={prov['dataset_version']} "
          f"schema={prov['schema_version']} {prov['h5_path']}")

    t0 = time.time()
    iv_inv, iv_info = implied_vol(rows["log_m"], rows["T"], rows["r"], rows["q"],
                                  np.exp(rows["target_v_log"]), option_type)
    iv_info["method"] = ("vectorised bisection on _common.bs_normalized_price over every "
                         "evaluated row (no subsampling)")
    iv_info["source"] = ("inverted from mid/K stored in the HDF5; OptionMetrics' per-contract "
                         "impl_volatility is not retained there and the 8.6 GB "
                         "optionPrice_2015_2025.csv was NOT read")
    print(f"[{option_type}] IV inversion: {iv_info['failure_rate']:.4%} failed "
          f"({time.time() - t0:.0f}s)")

    # v4 stores OptionMetrics' own per-contract IV: prefer it as the observation,
    # but keep the inversion and report how far apart the two are.
    iv_obs, iv_info["observed_iv"] = choose_observed_iv(rows, iv_inv)

    spot_vol = spx_vol_features(args.ewma_lambda)
    vix = vix_levels()
    rv = np.array([spot_vol[d][0] for d in rows["date"]])
    ew = np.array([spot_vol[d][1] for d in rows["date"]])
    vx, vix_src = vix_sigma(rows, vix)

    specs = []   # (name, branch, input_kind, sigma, params, diag, note)

    def add_fitted(name, branch, kind, sigma_of, lo, hi, pname, note):
        """Emit the log-objective and the price-objective fit of one scalar parameter."""
        base = sigma_of(rows)
        for space, suffix in (("log", ""), ("price", "_pxfit")):
            x, p = fit_on_train(option_type, sigma_of, lo, hi, space)
            sig = np.full(n, x) if base is None else x * base
            specs.append((name + suffix, branch, kind, sig,
                          {pname: x, "fit_space": space, **p}, {},
                          f"{note} [{pname} fitted on train, {space}-space objective]"))

    specs.append(("intrinsic_zero_vol", None, CONTEXT_FREE, None, {}, {},
                  "max(M*exp(-qT)-exp(-rT),0); zero parameters, no market input; a floor, "
                  "not a comparator for any branch"))

    add_fitted("const_sigma", None, CONTEXT_FREE, lambda d: None, 0.01, 2.0, "sigma",
               "one global sigma, no market input; not a comparator for any branch")

    for method, label in (("nearest", "nearest"), ("linear", "bilinear"), ("cubic", "spline")):
        sig, diag = surface_sigma(rows, surf, option_type, method=method)
        specs.append((f"surface_{label}", "vol_surface", BRANCH_COMPARABLE, sig, {}, diag,
                      f"RegularGridInterpolator(method={method}) on the 11x17 tenor-delta grid, "
                      "delta fixed point, out-of-grid clamped to boundary. Same-day IV surface = "
                      "exactly the `branch_u` input, so this IS a like-for-like comparator for "
                      "the vol_surface branch"))

    sig_prev, diag_prev = surface_sigma(rows, surf, option_type, method="linear", lag=1)
    prev_note = ("previous available trade date's surface (lag=1 on the full 2015-2025 surface "
                 "date axis), bilinear + delta fixed point. The lagged IV surface is NOT any "
                 "branch's input (the vol_surface branch sees the SAME-day surface; the "
                 "spot_history / vix_history branches see price and VIX histories, not a lagged "
                 "surface), so this is a stale-information reference point and must not be read "
                 "as a comparator for any branch")
    specs.append(("prev_day_surface_bilinear", None, LAGGED_REFERENCE, sig_prev, {},
                  diag_prev, prev_note))

    specs.append(("realized_vol_21d", "spot_history", BRANCH_COMPARABLE, rv,
                  {"window": LOOKBACK, "annualisation": TRADING_DAYS, "ddof": 1,
                   "fitted": "none (parameter-free)"}, {},
                  "close-to-close stdev of the 21-day SPX window (20 log-returns) -- the same "
                  "21-close window the spot_history branch sees, so a like-for-like comparator"))
    add_fitted("realized_vol_21d_scaled", "spot_history", BRANCH_COMPARABLE,
               lambda d: np.array([spot_vol[x][0] for x in d["date"]]), 0.2, 8.0, "scale",
               "realized vol x scale (variance-risk-premium proxy); same 21-close window as the "
               "spot_history branch")

    specs.append(("ewma_vol", "spot_history", BRANCH_COMPARABLE, ew,
                  {"lambda": args.ewma_lambda, "lambda_source": args.ewma_lambda_source,
                   "annualisation": TRADING_DAYS, "fitted": "none (lambda not fitted)"}, {},
                  "RiskMetrics EWMA over the same 21-day window as the spot_history branch"))
    add_fitted("ewma_vol_scaled", "spot_history", BRANCH_COMPARABLE,
               lambda d: np.array([spot_vol[x][1] for x in d["date"]]), 0.2, 8.0, "scale",
               "EWMA vol x scale; same 21-close window as the spot_history branch")

    vnote = ("CBOE SPX VIX daily close, source: %s. VIX is the expected 30-day SPX "
             "volatility, so this is a simple 30-day market-volatility benchmark, NOT a "
             "maturity-matched option IV. %s Its numbers are unaffected by the v3 VVIX "
             "mislabel, because it never touches the `vix_history` branch."
             % (vix_src["source"], VIX_ANTI_INFERENCE))
    specs.append(("vix_level_over_100", None, SCALAR_OBSERVABLE, vx,
                  {**vix_src, "normalisation": "close/100",
                   "fitted": "none (parameter-free)"}, {},
                  "VIX close / 100 used directly as sigma; " + vnote))
    add_fitted("vix_level_scaled", None, SCALAR_OBSERVABLE,
               lambda d: vix_sigma(d, vix)[0], 0.02, 5.0, "scale",
               "VIX close/100 x scale (fitted on train split); " + vnote)

    records = []
    for name, branch, kind, sigma, params, diag, note in specs:
        rec = {"option_type": option_type, **prov,
               "baseline": name, "branch": branch, "input_kind": kind,
               "params": params, "diagnostics": diag, "note": note,
               "n_test_rows": n, "n_test_dates": n_dates,
               "date_min": d_lo, "date_max": d_hi,
               "sigma_mean": None if sigma is None else float(np.nanmean(sigma)),
               "metrics": score(rows, sigma, option_type, iv_obs)}
        if "sample" in rows:
            rec["sampling"] = rows["sample"]
        records.append(rec)
        print(f"  {name:28s} R2(price)={rec['metrics']['all']['r2_price']:.6f} "
              f"R2(log,T>1d)={rec['metrics']['all']['r2_log_filtered']:.6f}")
    annotate(records, rows, prov)
    return records, iv_info


def sample_note(rows, prov):
    """One sentence naming the sample these numbers were computed on, plus the
    caveat that actually differs between v3 and v4 (the T==0 stratum)."""
    n0 = int((rows["T"] == 0).sum())
    base = ("computed on the %s test split (%s, %d rows). "
            % (prov["dataset_version"], Path(prov["h5_path"]).name, len(rows["T"])))
    if n0:
        return base + (
            "%d rows have T == 0 and are priced as discounted intrinsic for EVERY baseline "
            "(BS is intrinsic there for any sigma), so R2(log, full) and R2(log, T>1/365) are "
            "different numbers and the full-domain one is partly a constant across baselines."
            % n0)
    return base + (
        "there are NO T == 0 rows in this sample, so the T > 1/365 filter is a no-op here and "
        "R2(log, T>1day) == R2(log, full) exactly -- do not read them as two separate results.")


def annotate(records, rows, prov):
    """Attach the sample note, the price-accuracy flag and the ranking caveat.

    Done in one pass over the finished records rather than per-spec, so every
    baseline is labelled by the same rule and none can be forgotten.
    """
    note = sample_note(rows, prov)
    ok = [r for r in records if r["metrics"]["all"]["r2_price"] > PRICE_ACCURATE_MIN]
    best_log = max(records, key=lambda r: r["metrics"]["all"]["r2_log_filtered"])
    best_ok = max(ok, key=lambda r: r["metrics"]["all"]["r2_log_filtered"]) if ok else None
    tradeoff = (
        "RANKING: on this option type the best R2(log) baseline is %s (log %.4f, price %.6f); "
        "the best price-accurate one (R2(price)>%g) is %s (log %.4f, price %.6f). Quote BOTH "
        "metrics -- neither alone identifies a 'best baseline'."
        % (best_log["baseline"], best_log["metrics"]["all"]["r2_log_filtered"],
           best_log["metrics"]["all"]["r2_price"], PRICE_ACCURATE_MIN,
           best_ok["baseline"] if best_ok else "none",
           best_ok["metrics"]["all"]["r2_log_filtered"] if best_ok else float("nan"),
           best_ok["metrics"]["all"]["r2_price"] if best_ok else float("nan")))
    for r in records:
        px = r["metrics"]["all"]["r2_price"]
        r["price_accurate"] = bool(px > PRICE_ACCURATE_MIN)
        r["note"] = r["note"] + " | " + note
        r["ranking_caveat"] = (
            tradeoff if r["price_accurate"] else
            "NOT price-accurate: R2(price)=%.6f, so a good/less-bad R2(log) here does not make "
            "this a good baseline. %s" % (px, tradeoff))


CSV_METRICS = ["n_test", "n_dates", "date_min", "date_max", "r2_price", "rmse_price",
               "mae_price", "r2_log_full", "rmse_log", "r2_log_filtered", "mse_log_filtered",
               "iv_rmse", "iv_mae", "iv_bias", "iv_median_abs", "iv_p95_abs", "n_iv_compared"]
CSV_DIAG = ["nonconvergence_rate", "n_nonconverged", "out_of_grid_rate", "n_out_of_grid",
            "n_out_of_grid_delta", "n_out_of_grid_tenor"]


def write_outputs(records, meta, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each sample gets its own files: v3 keeps baselines_vix.*, v4 writes
    # baselines_v4.*, and the VVIX-era files stay archived as
    # baselines_vvix_legacy.{json,csv}. Nothing is ever overwritten across samples.
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"meta": meta, "records": records}, indent=2, default=str))

    cols = (["option_type", "dataset_version", "schema_version", "h5_path",
             "baseline", "branch", "input_kind", "price_accurate", "ranking_caveat",
             "note", "params", "n_test_rows",
             "n_test_dates", "date_min", "date_max", "sigma_mean"] + CSV_DIAG
            + [f"{s}__{m}" for s in STRATA for m in CSV_METRICS])
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: r.get(k) for k in cols}
            row["params"] = json.dumps(r["params"], default=str)
            for k in CSV_DIAG:
                row[k] = r["diagnostics"].get(k)
            for s in STRATA:
                for m in CSV_METRICS:
                    row[f"{s}__{m}"] = r["metrics"].get(s, {}).get(m)
            w.writerow(row)
    print(f"\nwrote {out_dir / (stem + '.json')} and {out_dir / (stem + '.csv')}")


def print_table(records):
    w = max(len(r["baseline"]) for r in records) + 1
    hdr = (f"{'type':5s} {'baseline':{w}s} {'branch':14s} {'pxOK':5s} {'R2(px)':>11s} "
           f"{'RMSE(px)':>9s} "
           f"{'MAE(px)':>9s} {'R2(log,T>1d)':>13s} {'RMSE(log,T>1d)':>15s} "
           f"{'IVrmse':>8s} {'IVmed':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in records:
        a, k = r["metrics"]["all"], r["metrics"]["T_gt_1d"]
        f2 = lambda v, s: "n/a" if v is None else format(v, s)  # noqa: E731
        print(f"{r['option_type']:5s} {r['baseline']:{w}s} {str(r['branch'] or '-'):14s} "
              f"{('yes' if r['price_accurate'] else 'NO'):5s} "
              f"{a['r2_price']:11.6f} {a['rmse_price']:9.2e} {a['mae_price']:9.2e} "
              f"{k['r2_log_filtered']:13.6f} {np.sqrt(k['mse_log_filtered']):15.4f} "
              f"{f2(k.get('iv_rmse'), '8.4f'):>8s} {f2(k.get('iv_median_abs'), '7.4f'):>7s}")
    print("\nranking guidance:")
    for ot in dict.fromkeys(r["option_type"] for r in records):
        r0 = next((r for r in records if r["option_type"] == ot), None)
        print(f"  {ot}: {r0['ranking_caveat'].split('RANKING: ')[-1]}")
    print("\nper-stratum  n / R2(price) / R2(log,full):")
    for r in records:
        cells = " | ".join(
            f"{s}: n={r['metrics'][s]['n_test']} "
            f"R2px={r['metrics'][s].get('r2_price', float('nan')):.6f} "
            f"R2log={r['metrics'][s].get('r2_log_full', float('nan')):.4f}"
            for s in STRATA[1:])
        print(f"  {r['option_type']:5s} {r['baseline']:{w}s} {cells}")
    print("\nsurface fixed-point diagnostics (T>0 rows):")
    for r in records:
        d = r["diagnostics"]
        if d:
            print(f"  {r['option_type']:5s} {r['baseline']:{w}s} "
                  f"nonconv={d['nonconvergence_rate']:.4%} ({d['n_nonconverged']}) "
                  f"out-of-grid={d['out_of_grid_rate']:.4%} "
                  f"(delta {d['n_out_of_grid_delta']}, tenor {d['n_out_of_grid_tenor']}) "
                  f"of {d['n_live_rows']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--option-type", choices=["call", "put", "both"], default="both")
    ap.add_argument("--max-contracts", type=int, default=None,
                    help="subsample the test split (iteration only; final numbers use all rows)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ewma-lambda", type=float, default=0.94)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--h5-dir", default=None,
                    help="override the directory holding the HDF5 files (default: "
                         "whatever --dataset-version resolves to)")
    ap.add_argument("--self-test", action="store_true")
    add_dataset_arg(ap)
    args = ap.parse_args()
    args.ewma_lambda_source = ("RiskMetrics standard 0.94, not fitted"
                               if args.ewma_lambda == 0.94 else "user-supplied via --ewma-lambda")
    _DATASET.update(version=args.dataset_version, h5_dir=args.h5_dir)
    if args.self_test:
        self_test()
        return

    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    surfaces = load_surfaces()
    types = ["call", "put"] if args.option_type == "both" else [args.option_type]
    # v3 keeps its historical stem so the existing results/baselines_vix.* stay put
    stem = "baselines_vix" if args.dataset_version == "v3" else "baselines" + output_tag(
        args.dataset_version)
    records, meta = [], {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "device": "cpu", "iv_inversion": {},
                         "dataset": {}, "dataset_version": args.dataset_version,
                         "surface_axes": {k: {"days": list(v["days"]), "delta": list(v["delta"])}
                                          for k, v in surfaces.items()},
                         "strata": STRATA_DOC,
                         "labels": BRANCH_LABELS_DOC,
                         "vix_anti_inference": VIX_ANTI_INFERENCE,
                         "metric_source": "analysis/_common.compute_metrics (unmodified)",
                         "pricer": "analysis/_common.bs_normalized_price, float64, "
                                   "log(clip(price, 1e-8)) exactly as in training",
                         "max_contracts": args.max_contracts, "seed": args.seed}
    for ot in types:
        prov = dataset_provenance(args.dataset_version, ot, args.h5_dir)
        meta["dataset"][ot] = prov
        recs, iv_info = build_records(ot, surfaces, args, prov)
        meta["iv_inversion"][ot] = iv_info
        records += recs
    print_table(records)
    write_outputs(records, meta, args.out_dir, stem)


if __name__ == "__main__":
    main()
