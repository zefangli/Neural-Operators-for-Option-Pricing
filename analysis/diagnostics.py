"""
Model-free diagnostics for the near-expiry log-space ceiling.

Answers one question with no network involved: *how much of the log-space SSE is
structurally unattainable, and why?*

The mechanism (verified in `wrds_data_2020-2025/pre_process_1_data.py:~283`):

    T_years = (exdate - date).total_days() / 365          # CALENDAR days

so a contract quoted on its own expiry date gets **exactly T = 0.0**. The BS
decoder (`_common.bs_normalized_price`) then evaluates

    sqrt_T = sqrt(clamp(T, min=1e-10))   ->  1e-5
    d1     = log_m / (sigma * 1e-5)      ->  +-inf for any sigma != 0

so the price collapses to the intrinsic value `max(M-1,0)` (call) /
`max(1-M,0)` (put) *for every sigma*. Vega underflows to 0, the `clamp(price,
1e-8)` in the log path zeroes the gradient a second time on the OTM side, and
the row becomes **unfittable**: the decoder output does not depend on the
quantity being learned. That is decoder degeneracy, not model error.

Three strata are kept strictly separate because their causes differ:

    days == 0   T == 0            decoder degeneracy    -> impossible
    days == 1   T == 1/365        ill-conditioned       -> hard but possible
    days >= 2   T >  1/365        well-conditioned      -> ordinary regime

`days = round(T * 365)`, which is exact here (T is a float32 of k/365).

--------------------------------------------------------------------------
A float32 boundary fact you need before comparing any two "T > 1/365" numbers
--------------------------------------------------------------------------
T is stored float32.  float32(1/365) upcasts to 0.0027397261001..., which is
*greater* than the float64 constant 1/365 = 0.0027397260274...  So

    T >  1.0/365.0   keeps the 1-day contracts   (== days >= 1)
    T <= 1.0/365.0   keeps only T == 0           (== days == 0)

i.e. `_common.compute_metrics`'s headline "R2(log, T > 1/365)" filter drops
*only* the T == 0 rows in practice. This module therefore never uses a float
threshold; it buckets on `days` so every number is unambiguous.

--------------------------------------------------------------------------
v3 vs v4
--------------------------------------------------------------------------
Everything above describes the **v3** sample. The **v4** sample applies
`T > 1 day` at export, so it contains **zero T==0 rows**: part A then reports
n=0 with "ceiling not binding", which is the correct and intended outcome of
the v4 preprocessing, not a failure. The three canonical strata (T==0,
0<T<=1day, T>1day) are always reported, empty or not, so the v3/v4 contrast is
explicit. `--dataset-version` is REQUIRED -- results from the two samples must
never be silently mixed.

Usage
-----
    conda run -n dl_new python analysis/diagnostics.py --dataset-version v4
    conda run -n dl_new python analysis/diagnostics.py --dataset-version v3 --option-type call
    conda run -n dl_new python analysis/diagnostics.py --dataset-version v4smoke --max-contracts 50000

Writes results/diagnostics_tail{tag}.json, results/diagnostics_strata{tag}.csv,
results/diagnostics_conditioning{tag}.csv, where tag is "" for v3 and "_v4" /
"_v4smoke" otherwise.  CPU only; touches no checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from _common import (PROJECT_ROOT, T_MIN_YEARS, add_dataset_arg, bs_normalized_price,
                     dataset_h5, dataset_provenance, liquidity_masks, output_tag)

RESULTS = PROJECT_ROOT / "results"
CLAMP = 1e-8

# The three canonical strata, on T itself (not the days rounding), always
# reported even when empty -- in v4 the first two are empty by construction.
CANONICAL = [
    ("T==0", "decoder-degenerate (unfittable)", lambda T: T == 0.0),
    ("0<T<=1day", "ill-conditioned (hard)", lambda T: (T > 0.0) & (T <= T_MIN_YEARS)),
    ("T>1day", "well-conditioned", lambda T: T > T_MIN_YEARS),
]

# (label, lo_days, hi_days) -- hi inclusive, None = open ended
T_BUCKETS = [
    ("d=0", 0, 0),
    ("d=1", 1, 1),
    ("d=2-3", 2, 3),
    ("d=4-7", 4, 7),
    ("d=8-30", 8, 30),
    ("d=31-90", 31, 90),
    ("d=91-365", 91, 365),
    ("d>365", 366, None),
]
PCTS = [1, 5, 25, 50, 75, 95, 99]


# ---------------------------------------------------------------- data load


def load_split(h5_path, split=2, max_contracts=None, seed=42):
    """Small columns only -- never touches branch_u/spot_history/vix_history."""
    with h5py.File(h5_path, "r") as f:
        idx = np.where(f["split_id"][:] == split)[0]
        if max_contracts is not None and len(idx) > max_contracts:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_contracts, replace=False))
        trunk = f["trunk_y"][:][idx].astype(np.float64)
        out = {
            "idx": idx,
            "log_m": trunk[:, 0],
            "T": trunk[:, 1],
            "r": trunk[:, 2],
            "q": trunk[:, 3],
            "target": f["target_v_log"][:][idx].ravel().astype(np.float64),
            "M": f["moneyness"][:][idx].ravel().astype(np.float64),
            "price": f["normalized_price"][:][idx].ravel().astype(np.float64),
            "date": f["date"][:][idx],
        }
        # v4/v5 only; feeds the quote-liquidity REPORTING strata. Presence-checked
        # so v3 still runs, and never used to drop a row.
        for opt in ("best_bid", "best_offer"):
            if opt in f:
                out[opt] = f[opt][:][idx].ravel().astype(np.float64)
    out["days"] = np.rint(out["T"] * 365.0).astype(np.int64)
    return out


def bucket_mask(days, lo, hi):
    m = days >= lo
    return m if hi is None else m & (days <= hi)


def intrinsic(M, option_type):
    return np.maximum(M - 1.0, 0.0) if option_type == "call" else np.maximum(1.0 - M, 0.0)


# ------------------------------------------------------- part A: the ceiling


def t0_ceiling(d, option_type, solved=None):
    """Irreducible SSE from the T==0 rows and the R2(log, full) ceiling it implies.

    At T==0 the decoder returns intrinsic for *any* sigma, so the best any model
    can do on those rows is log(clip(intrinsic, 1e-8)). Everything else in the
    split is assumed perfectly fitted, so this is an upper bound on R2.

    `solved` (from `implied_vol`) corroborates the claim independently: it is the
    mask of rows whose observed price is reachable by the decoder for *some*
    sigma in [1e-4, 5]. ~0% of the T==0 rows are.
    """
    tgt = d["target"]
    z = d["days"] == 0
    best = np.log(np.clip(intrinsic(d["M"], option_type), CLAMP, None))
    resid = tgt[z] - best[z]
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((tgt - tgt.mean()) ** 2))

    otm = z & (intrinsic(d["M"], option_type) <= 0.0)
    r_otm = tgt[otm] - best[otm]
    itm = z & ~otm
    r_itm = tgt[itm] - best[itm]

    # v4 filters T <= 1 day at export, so z is empty there: sse == 0, the ceiling
    # is 1.0 and non-binding. Every share is then 0/0 -- report 0.0, never divide.
    share = lambda x: 100.0 * float(x) / sse if sse > 0 else 0.0     # noqa: E731
    med = lambda a: float(np.median(np.abs(a))) if a.size else float("nan")  # noqa: E731

    return {
        "n_rows": int(tgt.size),
        "n_T0": int(z.sum()),
        "pct_T0": 100.0 * float(z.mean()),
        "ceiling_binding": bool(z.any()),
        "note": ("no T==0 rows in this sample (v4 filters T <= 1 day at export): the "
                 "decoder-degeneracy ceiling does not bind" if not z.any() else
                 "T==0 rows present: the decoder-degeneracy ceiling binds"),
        "pct_T0_price_unreachable_any_sigma": (
            100.0 * float((~solved[z]).mean()) if solved is not None and z.any()
            else float("nan")),
        "sse_T0": sse,
        "sst_total": sst,
        "sse_share_of_sst_pct": 100.0 * sse / sst,
        "r2_log_full_ceiling": 1.0 - sse / sst,
        "otm": {
            "n": int(otm.sum()),
            "sse": float(np.sum(r_otm ** 2)),
            "share_of_T0_sse_pct": share(np.sum(r_otm ** 2)),
            "median_abs_err": med(r_otm),
            "median_target": float(np.median(tgt[otm])) if otm.any() else float("nan"),
        },
        "itm": {
            "n": int(itm.sum()),
            "sse": float(np.sum(r_itm ** 2)),
            "share_of_T0_sse_pct": share(np.sum(r_itm ** 2)),
            "median_abs_err": med(r_itm),
        },
    }


def strata_table(d, option_type):
    """Per-stratum rows / SST share / (for T==0) the irreducible SSE.

    Two bases in one table, distinguished by the `basis` column:
      * basis="T"    -- the three canonical strata, ALWAYS emitted even when
                        empty (v4 has no T<=1day rows at all);
      * basis="days" -- the finer days=round(T*365) buckets, empty ones dropped.
    """
    tgt = d["target"]
    sst = float(np.sum((tgt - tgt.mean()) ** 2))
    best = np.log(np.clip(intrinsic(d["M"], option_type), CLAMP, None))

    def row_for(label, cause, m, basis, forced):
        n = int(m.sum())
        sse = float(np.sum((tgt[m] - best[m]) ** 2)) if (forced and n) else 0.0
        return {
            "basis": basis, "stratum": label, "cause": cause, "n": n,
            "pct_of_split": 100.0 * n / tgt.size,
            "sst_share_pct": 100.0 * float(np.sum((tgt[m] - tgt.mean()) ** 2)) / sst,
            "target_median": float(np.median(tgt[m])) if n else float("nan"),
            "pct_clamped_target": 100.0 * float(np.mean(d["price"][m] <= CLAMP)) if n else 0.0,
            # the intrinsic-only floor is only *forced* at T==0; report it there alone.
            "irreducible_sse": sse,
            "irreducible_sse_share_pct": 100.0 * sse / sst,
        }

    rows = [row_for(lab, cause, sel(d["T"]), "T", lab == "T==0")
            for lab, cause, sel in CANONICAL]
    for label, lo, hi in T_BUCKETS:
        m = bucket_mask(d["days"], lo, hi)
        if not m.any():
            continue
        rows.append(row_for(
            label,
            {"d=0": "decoder-degenerate (unfittable)",
             "d=1": "ill-conditioned (hard)"}.get(label, "well-conditioned"),
            # d=0 is "T rounds to 0 days", which only IMPLIES exact T==0 under
            # the v3 k/365 quantisation; under v4's settlement T it does not.
            m, "days", label == "d=0" and bool((d["T"] == 0.0).any())))
    # basis="liquidity": quote-liquidity REPORTING strata (v4/v5 only, absent on
    # v3). These describe how the error sits across quote quality; they do NOT
    # filter the sample -- no caller drops rows on liquidity, and doing so would
    # be a separate dataset decision that has not been taken.
    for label, m in liquidity_masks(d.get("best_bid"), d.get("best_offer")).items():
        rows.append(row_for(label, "reporting stratum (NOT a filter)", m,
                            "liquidity", False))
    return rows


# ------------------------------------------- part B: vega / conditioning


def _vega_norm(log_m, T, r, q, sigma):
    """d(V/K)/d(sigma) in normalized units. Closed form, matching the decoder's
    clamping so the T==0 degeneracy shows up rather than being papered over."""
    M = np.exp(log_m)
    sqrt_T = np.sqrt(np.clip(T, 1e-10, None))
    s = np.clip(sigma, 1e-10, None)
    d1 = (log_m + (r - q + 0.5 * s ** 2) * T) / (s * sqrt_T)
    phi = np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi)
    return M * np.exp(-q * T) * phi * sqrt_T


def implied_vol(price, log_m, T, r, q, option_type, iters=80, hi=5.0):
    """Vectorised bisection on the *decoder's own* pricer (torch, float64).

    Returns (sigma, solved_mask). Rows whose price is outside the pricer's
    reachable range at [1e-4, hi] get solved=False -- notably every T==0 row,
    where the pricer is constant, which is the point.
    """
    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    lm, TT, rr, qq, px = map(t, (log_m, T, r, q, price))
    lo_s = torch.full_like(px, 1e-4)
    hi_s = torch.full_like(px, hi)
    p_lo = bs_normalized_price(lm, TT, rr, qq, lo_s, option_type)
    p_hi = bs_normalized_price(lm, TT, rr, qq, hi_s, option_type)
    solved = (px >= p_lo) & (px <= p_hi)
    for _ in range(iters):
        mid = 0.5 * (lo_s + hi_s)
        p = bs_normalized_price(lm, TT, rr, qq, mid, option_type)
        too_low = p < px
        lo_s = torch.where(too_low, mid, lo_s)
        hi_s = torch.where(too_low, hi_s, mid)
    sigma = (0.5 * (lo_s + hi_s)).numpy()
    return sigma, solved.numpy()


def conditioning_profile(d, option_type, sigma, solved, sigma_ref=0.20):
    """Vega and d log(price)/d sigma percentiles per T bucket.

    Two evaluation points, because neither alone tells the whole story:
      * at the **market-implied sigma** (the actual solution of the inverse
        problem) -- the honest conditioning number, but undefined wherever the
        price is unreachable, which is ~every T==0 row;
      * at a **fixed sigma_ref=0.20** -- defined everywhere, so the d=0 row of
        the table shows the vega underflow instead of a blank.
    """
    vega = _vega_norm(d["log_m"], d["T"], d["r"], d["q"], sigma)
    vega_ref = _vega_norm(d["log_m"], d["T"], d["r"], d["q"],
                          np.full_like(d["T"], sigma_ref))
    price = np.clip(d["price"], CLAMP, None)
    dlogp = vega / price          # d log(price) / d sigma -- the conditioning number
    pct = (lambda a, p: float(np.percentile(a, p)) if a.size else float("nan"))

    def row_for(label, m, basis):
        ms = m & solved
        n = int(m.sum())
        row = {"basis": basis, "stratum": label, "n": n,
               "pct_iv_solvable": 100.0 * float(solved[m].mean()) if n else float("nan"),
               "sigma_median": pct(sigma[ms], 50),
               "vega_at_sigma_ref_median": pct(vega_ref[m], 50),
               "vega_at_sigma_ref_p99": pct(vega_ref[m], 99)}
        for p in PCTS:
            row["vega_p%d" % p] = pct(vega[ms], p)
        for p in PCTS:
            row["dlogp_dsigma_p%d" % p] = pct(dlogp[ms], p)
        return row

    # canonical strata always present (empty in v4), then the finer days buckets
    rows = [row_for(lab, sel(d["T"]), "T") for lab, _, sel in CANONICAL]
    for label, lo, hi in T_BUCKETS:
        m = bucket_mask(d["days"], lo, hi)
        if m.any():
            rows.append(row_for(label, m, "days"))
    return rows


def degeneracy_check(option_type, n_sigma=25):
    """Direct numeric proof that T==0 kills the sigma dependence and T=1/365 does not.

    Sweeps sigma over [0.01, 3.0] at five moneyness points and reports how much
    the decoder's output actually moves, plus its distance from intrinsic.
    """
    lm = torch.tensor([-0.20, -0.05, 0.0, 0.05, 0.20], dtype=torch.float64)
    intr = (torch.clamp(torch.exp(lm) - 1.0, min=0.0) if option_type == "call"
            else torch.clamp(1.0 - torch.exp(lm), min=0.0)).unsqueeze(1)
    sig = torch.linspace(0.01, 3.0, n_sigma, dtype=torch.float64)
    g_lm = lm.repeat_interleave(n_sigma)
    g_s = sig.repeat(lm.numel())
    zero = torch.zeros_like(g_lm)
    out = {}
    for label, T in (("T=0", 0.0), ("T=1/365", 1.0 / 365.0), ("T=30/365", 30.0 / 365.0)):
        p = bs_normalized_price(g_lm, torch.full_like(g_lm, T), zero, zero, g_s,
                                option_type).reshape(lm.numel(), n_sigma)
        out[label] = {
            "max_price_spread_over_sigma": float((p.max(1).values - p.min(1).values).max()),
            "max_abs_dev_from_intrinsic": float((p - intr).abs().max()),
        }
    return out


# ------------------------------------------------------------------- report


def _fmt_pcts(row, prefix):
    return "  ".join("p%d=%.3g" % (p, row["%s_p%d" % (prefix, p)]) for p in (5, 50, 95))


def run(option_type, h5_path, max_contracts=None, prov=None):
    d = load_split(h5_path, max_contracts=max_contracts)
    sigma, solved = implied_vol(d["price"], d["log_m"], d["T"], d["r"], d["q"], option_type)
    ceil = t0_ceiling(d, option_type, solved)
    strata = strata_table(d, option_type)
    cond = conditioning_profile(d, option_type, sigma, solved)
    degen = degeneracy_check(option_type)

    print("=" * 78)
    print("%s  test split: %d rows, %d dates%s" % (
        option_type.upper(), ceil["n_rows"], len(np.unique(d["date"])),
        "  [SUBSAMPLE]" if max_contracts else ""))
    if prov:
        print("  sample: %s  schema_version=%s\n  %s" % (
            prov["dataset_version"], prov["schema_version"], prov["h5_path"]))
    print("=" * 78)
    print("\n-- A. T==0 decoder degeneracy -> hard R2(log,full) ceiling")
    print("   T==0 rows            : %d  (%.4f%% of split)" % (ceil["n_T0"], ceil["pct_T0"]))
    if not ceil["ceiling_binding"]:
        print("   CEILING NOT BINDING  : %s" % ceil["note"])
    else:
        print("   ...of which the observed price is UNREACHABLE by the decoder at any"
              "\n      sigma in [1e-4,5]: %.4f%%" % ceil["pct_T0_price_unreachable_any_sigma"])
    print("   irreducible SSE      : %.6g  (%.4f%% of SST)" % (ceil["sse_T0"], ceil["sse_share_of_sst_pct"]))
    print("   R2(log, full) CEILING: %.6f" % ceil["r2_log_full_ceiling"])
    print("   OTM subset           : n=%d  %.3f%% of the T==0 SSE  median|err|=%.4f" % (
        ceil["otm"]["n"], ceil["otm"]["share_of_T0_sse_pct"], ceil["otm"]["median_abs_err"]))
    print("   ITM subset           : n=%d  %.3f%% of the T==0 SSE" % (
        ceil["itm"]["n"], ceil["itm"]["share_of_T0_sse_pct"]))

    print("\n-- B. strata (never conflate T==0 with 0<T<=1day)")
    print("   %-5s %-11s %-32s %9s %8s %9s %12s" % (
        "basis", "stratum", "cause", "n", "%split", "%SST", "irred.SSE%"))
    for r in strata:
        print("   %-5s %-11s %-32s %9d %7.3f%% %8.3f%% %11.4f%%" % (
            r["basis"], r["stratum"], r["cause"], r["n"], r["pct_of_split"],
            r["sst_share_pct"], r["irreducible_sse_share_pct"]))

    print("\n-- C. conditioning (vega, dlogP/dsigma) -- percentiles at the market-implied sigma")
    print("   %-5s %-11s %9s %8s %8s %11s  %-32s %s" % (
        "basis", "stratum", "n", "%solved", "sigma~", "vega@.20~", "vega", "dlogP/dsigma"))
    for r in cond:
        print("   %-5s %-11s %9d %7.2f%% %8.4f %11.3e  %-32s %s" % (
            r["basis"], r["stratum"], r["n"], r["pct_iv_solvable"], r["sigma_median"],
            r["vega_at_sigma_ref_median"],
            _fmt_pcts(r, "vega"), _fmt_pcts(r, "dlogp_dsigma")))

    print("\n-- D. decoder degeneracy, direct check (price spread over sigma in [0.01,3])")
    for k, v in degen.items():
        print("   %-9s max spread over sigma = %.3e   max |p - intrinsic| = %.3e" % (
            k, v["max_price_spread_over_sigma"], v["max_abs_dev_from_intrinsic"]))
    print()
    return {"dataset": prov,
            "ceiling": ceil, "strata": strata, "conditioning": cond, "degeneracy": degen,
            "n_dates": int(len(np.unique(d["date"]))),
            "T_min": float(d["T"].min()), "T_max": float(d["T"].max()),
            "subsampled": max_contracts is not None}


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--option-type", default="both", choices=["call", "put", "both"])
    ap.add_argument("--max-contracts", type=int, default=None,
                    help="subsample the test split (iteration only; omit for final numbers)")
    ap.add_argument("--h5-dir", default=None,
                    help="override the directory holding the HDF5 files (default: "
                         "whatever --dataset-version resolves to)")
    ap.add_argument("--out", default=str(RESULTS))
    add_dataset_arg(ap)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = output_tag(args.dataset_version)
    types = ["call", "put"] if args.option_type == "both" else [args.option_type]

    report, strata_rows, cond_rows = {}, [], []
    for ot in types:
        prov = dataset_provenance(args.dataset_version, ot, args.h5_dir)
        h5 = dataset_h5(args.dataset_version, ot, args.h5_dir)
        report[ot] = run(ot, h5, args.max_contracts, prov)
        stamp = {"option_type": ot, **prov}
        for r in report[ot]["strata"]:
            strata_rows.append({**stamp, **r})
        for r in report[ot]["conditioning"]:
            cond_rows.append({**stamp, **r})

    report["_note_float32_boundary"] = (
        "T is float32; float32(1/365)=0.0027397261001 > 1/365=0.0027397260274, so "
        "`T > 1/365` keeps the 1-day contracts and `T <= 1/365` selects exactly T==0. "
        "The basis='days' strata are bucketed on days=round(T*365) instead; the "
        "basis='T' strata use the exact comparisons and are always emitted."
    )
    report["_note_dataset"] = (
        "v4 applies T > 1 day at export, so the T==0 and 0<T<=1day strata are EMPTY "
        "by construction and the decoder-degeneracy ceiling does not bind. That is the "
        "intended outcome of the v4 preprocessing, not a failed diagnostic.")
    report["_note_liquidity_strata"] = (
        "basis='liquidity' rows (bid_pos: best_bid > 0; relspread_le1: (ask-bid)/mid "
        "<= 1, which is bounded by 2 when bid >= 0; and their intersection) are "
        "REPORTING strata only. NOTHING is dropped for liquidity anywhere in this "
        "pipeline -- every other row in this file still covers the whole sample. "
        "Empty on v3, which has no best_bid/best_offer datasets.")
    (out / ("diagnostics_tail%s.json" % tag)).write_text(json.dumps(report, indent=2))
    _write_csv(out / ("diagnostics_strata%s.csv" % tag), strata_rows)
    _write_csv(out / ("diagnostics_conditioning%s.csv" % tag), cond_rows)
    print("wrote %s/diagnostics_{tail%s.json,strata%s.csv,conditioning%s.csv}"
          % (out, tag, tag, tag))


if __name__ == "__main__":
    main()
