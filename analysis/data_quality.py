"""
Data-quality report for the HDF5 test split (model-free, CPU-only, read-only).

Three checks, each a known publication risk:

(a) **Settlement-duplicate pairs.** The raw OptionMetrics table carries both the
    AM-settled monthly `SPX` contracts and the PM-settled weeklies `SPXW`, and
    `pre_process_1_data.py` selects neither `symbol`, `optionid` nor
    `am_settlement` (see its `_build_options_lazy` select list). Two genuinely
    different contracts that share (date, strike, expiry) therefore collapse to
    **identical model inputs** with different targets. Grouping on the full
    input key (date, log_moneyness, T_years, r, q) recovers the collision rate
    and the irreducible SSE it forces.

(b) **Black-Scholes static-bound violations.** In normalized V/K units with
    `fwd = M*exp(-qT)` and `disc = exp(-rT)`:
        call in [max(fwd-disc, 0), fwd]        put in [max(disc-fwd, 0), disc]
    Rows outside these bounds are unpriceable by *any* sigma, so they cap the
    attainable fit exactly like the T==0 rows do.

(c) **The extreme outlier.** One call violation is ~4 orders of magnitude deeper
    than the median. `--trace-raw` chases it back into the 8.6 GB
    `optionPrice_2015_2025.csv` (via a pushed-down `scan_csv` filter, never
    materialised) and prints the underlying quote.

v3 vs v4
--------
Check (a) is the reason `am_settlement` is retained in **v4**, where the
collision rate collapses to ~0%. When the file carries `am_settlement`, (a) is
reported TWICE -- on the model-input key alone, and on that key PLUS
`am_settlement` -- so the before/after contrast is explicit and verifiable in
one run. `--dataset-version` is REQUIRED: results from the v3 and v4 samples
must never be silently mixed.

Usage
-----
    conda run -n dl_new python analysis/data_quality.py --dataset-version v4
    conda run -n dl_new python analysis/data_quality.py --dataset-version v3 --max-contracts 200000
    conda run -n dl_new python analysis/data_quality.py --dataset-version v4 --no-trace-raw

Writes results/data_quality{tag}.json plus
results/data_quality_{duplicates,bounds}{tag}.csv, where tag is "" for v3 and
"_v4" / "_v4smoke" otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from _common import (PROJECT_ROOT, add_dataset_arg, dataset_h5, dataset_provenance,
                     output_tag)

RESULTS = PROJECT_ROOT / "results"
SECID = 108105

T_BUCKETS = [("d=0", 0, 0), ("d=1", 1, 1), ("d=2-7", 2, 7), ("d=8-30", 8, 30),
             ("d=31-90", 31, 90), ("d=91-365", 91, 365), ("d>365", 366, None)]
M_BUCKETS = [("deep OTM/ITM  M<0.8", None, 0.8), ("0.8-0.95", 0.8, 0.95),
             ("near ATM 0.95-1.05", 0.95, 1.05), ("1.05-1.2", 1.05, 1.2),
             ("M>1.2", 1.2, None)]


def _mask(v, lo, hi):
    m = np.ones(v.shape, bool)
    if lo is not None:
        m &= v >= lo
    if hi is not None:
        m &= v < hi
    return m


def load_test(h5_path, max_contracts=None, seed=42):
    with h5py.File(h5_path, "r") as f:
        idx = np.where(f["split_id"][:] == 2)[0]
        if max_contracts is not None and len(idx) > max_contracts:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_contracts, replace=False))
        trunk = f["trunk_y"][:][idx]
        d = {
            "idx": idx,
            "trunk_f32": trunk,                       # kept raw for exact-bit grouping
            "log_m": trunk[:, 0].astype(np.float64),
            "T": trunk[:, 1].astype(np.float64),
            "r": trunk[:, 2].astype(np.float64),
            "q": trunk[:, 3].astype(np.float64),
            "target": f["target_v_log"][:][idx].ravel().astype(np.float64),
            "M": f["moneyness"][:][idx].ravel().astype(np.float64),
            "price": f["normalized_price"][:][idx].ravel().astype(np.float64),
            "date": f["date"][:][idx],
        }
        # v4-only columns; absent in v3, so every use is guarded on presence.
        for opt in ("am_settlement", "impl_volatility", "optionid", "symbol"):
            if opt in f:
                d[opt] = f[opt][:][idx].ravel()
    d["days"] = np.rint(d["T"] * 365.0).astype(np.int64)
    return d


# ------------------------------------------------ (a) settlement duplicates


def duplicate_report(d, extra_keys=()):
    """Group on the exact float32 bits of the full model input key.

    `extra_keys` names additional columns of `d` to append to the key. Passing
    ("am_settlement",) answers "would keeping the AM/PM flag have separated
    these rows?" -- the v3 -> v4 fix, measured rather than asserted.
    """
    n = d["target"].size
    tr = d["trunk_f32"]
    cols = [d["date"], tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]]
    names = ["date", "log_m", "T", "r", "q"]
    for k in extra_keys:
        cols.append(d[k])
        names.append(k)
    key = np.rec.fromarrays(cols, names=",".join(names))
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.ones(n, bool)
    starts[1:] = ks[1:] != ks[:-1]
    start_idx = np.flatnonzero(starts)
    gid = np.cumsum(starts) - 1
    sizes = np.bincount(gid)

    tg = d["target"][order]
    gmin = np.minimum.reduceat(tg, start_idx)
    gmax = np.maximum.reduceat(tg, start_idx)
    gmean = np.add.reduceat(tg, start_idx) / sizes
    spread = gmax - gmin

    dup_g = sizes > 1
    dup_row = dup_g[gid]
    # Best possible constant prediction per group is the group mean; whatever is
    # left is unfittable by construction (identical inputs, different targets).
    resid = tg - gmean[gid]
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((tg - tg.mean()) ** 2))

    sp = spread[dup_g]
    pct = lambda a, p: float(np.percentile(a, p)) if a.size else 0.0   # noqa: E731
    rep = {
        "key": names,
        "n_rows": n,
        "n_groups": int(sizes.size),
        "max_group_size": int(sizes.max()),
        "group_size_histogram": {str(k): int(v) for k, v in
                                 enumerate(np.bincount(sizes)) if v},
        "n_dup_groups": int(dup_g.sum()),
        "pct_rows_in_dup_groups": 100.0 * float(dup_row.mean()),
        # sp is empty when nothing collides (the v4 expectation) -- report 0, not nan.
        "pct_dup_groups_disagreeing": 100.0 * float((sp > 0).mean()) if sp.size else 0.0,
        "target_spread": {("p%d" % p): pct(sp, p) for p in (50, 75, 90, 99, 100)},
        "irreducible_sse": sse,
        "irreducible_sse_share_of_sst_pct": 100.0 * sse / sst,
        "by_T_bucket": [],
    }
    days_o = d["days"][order]
    for label, lo, hi in T_BUCKETS:
        m = (days_o >= lo) & (True if hi is None else days_o <= hi)
        if not m.any():
            continue
        rep["by_T_bucket"].append({
            "stratum": label, "n": int(m.sum()),
            "pct_rows_in_dup_groups": 100.0 * float(dup_row[m].mean()),
            "irreducible_sse": float(np.sum(resid[m] ** 2)),
            "irreducible_sse_share_of_sst_pct": 100.0 * float(np.sum(resid[m] ** 2)) / sst,
        })
    return rep


# ------------------------------------------------------- (b) static bounds


def bounds(d, option_type):
    fwd = d["M"] * np.exp(-d["q"] * d["T"])
    disc = np.exp(-d["r"] * d["T"])
    if option_type == "call":
        lo, hi = np.maximum(fwd - disc, 0.0), fwd
    else:
        lo, hi = np.maximum(disc - fwd, 0.0), disc
    depth = np.maximum(lo - d["price"], d["price"] - hi)
    return lo, hi, depth


def bounds_report(d, option_type):
    lo, hi, depth = bounds(d, option_type)
    v = depth > 0
    dv = depth[v]
    n = depth.size
    # every stat below is conditional on there being at least one violation
    frac = lambda m: 100.0 * float(m[v].mean()) if dv.size else 0.0     # noqa: E731
    pct = lambda p: float(np.percentile(dv, p)) if dv.size else 0.0     # noqa: E731
    rep = {
        "n_rows": n,
        "n_violations": int(v.sum()),
        "pct_violations": 100.0 * float(v.mean()),
        "pct_lower_bound": frac((lo - d["price"]) > 0),
        "pct_upper_bound": frac((d["price"] - hi) > 0),
        "depth": {("p%d" % p): pct(p) for p in (25, 50, 75, 90, 99, 100)},
        # "is the deepest violation a lone outlier or the tip of a cluster?"
        "depth_tail_counts": {("gt_%g" % thr): int((dv > thr).sum())
                              for thr in (1e-3, 1e-2, 1e-1, 1.0)},
        # Both spellings, because they differ: T is float32 and float32(1/365)
        # upcasts ABOVE the float64 constant 1/365 (see analysis/diagnostics.py).
        "pct_violations_with_T0": frac(d["days"] == 0),
        "pct_violations_with_days_le_1": frac(d["days"] <= 1),
        "by_T_bucket": [], "by_moneyness": [],
    }
    for label, blo, bhi in T_BUCKETS:
        m = (d["days"] >= blo) & (True if bhi is None else d["days"] <= bhi)
        if not m.any():
            continue
        rep["by_T_bucket"].append({
            "stratum": label, "n": int(m.sum()),
            "n_violations": int((v & m).sum()),
            "pct_violations": 100.0 * float(v[m].mean()),
            "pct_of_all_violations": 100.0 * float((v & m).sum()) / max(int(v.sum()), 1),
            "median_depth": float(np.median(depth[v & m])) if (v & m).any() else 0.0,
        })
    for label, blo, bhi in M_BUCKETS:
        m = _mask(d["M"], blo, bhi)
        if not m.any():
            continue
        rep["by_moneyness"].append({
            "bucket": label, "n": int(m.sum()),
            "n_violations": int((v & m).sum()),
            "pct_violations": 100.0 * float(v[m].mean()),
            "pct_of_all_violations": 100.0 * float((v & m).sum()) / max(int(v.sum()), 1),
            "median_depth": float(np.median(depth[v & m])) if (v & m).any() else 0.0,
        })
    top = np.argsort(-depth)[:20]
    top = top[depth[top] > 0]        # never list non-violations as "top violations"
    rep["top20_violations"] = [{
        "h5_row": int(d["idx"][i]), "date": d["date"][i].decode(),
        "moneyness": float(d["M"][i]), "implied_strike_from_M": None,
        "T_years": float(d["T"][i]), "days": int(d["days"][i]),
        "r": float(d["r"][i]), "q": float(d["q"][i]),
        "normalized_price": float(d["price"][i]),
        "lower_bound": float(lo[i]), "upper_bound": float(hi[i]),
        "depth": float(depth[i]),
        "bound_broken": "lower" if (lo[i] - d["price"][i]) > 0 else "upper",
    } for i in top]
    return rep


# ------------------------------------------------------ (c) trace the outlier


def trace_raw(rec, data_dir):
    """Pull the source quote for one violation out of the 8.6 GB raw CSV.

    Filters are pushed into `scan_csv` (secid + date + cp_flag) so polars streams
    the file and only the handful of matching rows is ever materialised.
    """
    import polars as pl

    date = rec["date"]
    spot = (pl.scan_csv(Path(data_dir) / "spx_price_2015_2025.csv")
            .filter((pl.col("secid") == SECID) & (pl.col("date") == date))
            .select("close").collect())
    if spot.height == 0:
        return {"error": "no spot for %s" % date}
    S = float(spot[0, "close"])
    strike = S / rec["moneyness"]
    cp = "C" if rec["_option_type"] == "call" else "P"
    # strike_price is in 1/1000 dollars; allow a wide window then pick the closest.
    lo, hi = int(strike * 1000 * 0.98), int(strike * 1000 * 1.02)
    raw = (pl.scan_csv(Path(data_dir) / "optionPrice_2015_2025.csv", infer_schema_length=10000)
           .filter((pl.col("secid") == SECID) & (pl.col("date") == date)
                   & (pl.col("cp_flag") == cp)
                   & (pl.col("strike_price").is_between(lo, hi))
                   & (pl.col("volume") > 0))
           .select("date", "symbol", "exdate", "cp_flag", "strike_price", "best_bid",
                   "best_offer", "volume", "open_interest", "impl_volatility",
                   "optionid", "am_settlement", "exercise_style", "index_flag",
                   "contract_size", "cfadj")
           .collect(engine="streaming"))
    if raw.height == 0:
        return {"spot": S, "implied_strike": strike, "error": "no raw match"}
    raw = raw.with_columns(
        ((pl.col("best_bid") + pl.col("best_offer")) / 2.0).alias("mid"),
        (pl.col("strike_price") / 1000.0).alias("strike"),
    ).with_columns(
        (pl.col("mid") / pl.col("strike")).alias("normalized_price"),
        (pl.col("exdate").str.to_date() - pl.col("date").str.to_date())
        .dt.total_days().alias("days_to_expiry"),
    )
    # pick the row whose normalized_price reproduces the HDF5 value
    raw = raw.with_columns(
        (pl.col("normalized_price") - rec["normalized_price"]).abs().alias("_err")
    ).sort("_err")
    return {"spot": S, "implied_strike": strike, "n_raw_candidates": raw.height,
            "match": raw.head(1).to_dicts()[0],
            "all_candidates": raw.drop("_err").to_dicts()[:10]}


# ------------------------------------------------------------------- driver


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")


def run(option_type, h5, max_contracts=None, do_trace=True, prov=None, data_dir=None):
    d = load_test(h5, max_contracts=max_contracts)
    dup = duplicate_report(d)
    # v4 keeps am_settlement, so the same grouping WITH it shows the fix directly.
    dup_am = duplicate_report(d, ("am_settlement",)) if "am_settlement" in d else None
    bnd = bounds_report(d, option_type)

    print("=" * 84)
    print("%s  test split: %d rows, %d dates%s" % (
        option_type.upper(), d["target"].size, len(np.unique(d["date"])),
        "  [SUBSAMPLE]" if max_contracts else ""))
    if prov:
        print("  sample: %s  schema_version=%s\n  %s" % (
            prov["dataset_version"], prov["schema_version"], prov["h5_path"]))
    print("=" * 84)

    def show_dup(dup, title):
        print("\n%s" % title)
        print("    groups                     : %d   max group size: %d" % (
            dup["n_groups"], dup["max_group_size"]))
        print("    group size histogram       : %s" % dup["group_size_histogram"])
        print("    rows in duplicate groups   : %.4f%%  (%d groups)" % (
            dup["pct_rows_in_dup_groups"], dup["n_dup_groups"]))
        print("    dup groups disagreeing     : %.4f%%" % dup["pct_dup_groups_disagreeing"])
        print("    within-group target spread : med=%.5f p90=%.5f p99=%.5f max=%.5f" % (
            dup["target_spread"]["p50"], dup["target_spread"]["p90"],
            dup["target_spread"]["p99"], dup["target_spread"]["p100"]))
        print("    IRREDUCIBLE SSE share      : %.6f%% of total variance" %
              dup["irreducible_sse_share_of_sst_pct"])
        print("    %-10s %10s %14s %14s" % ("stratum", "n", "%dup rows", "irred.SSE%"))
        for r in dup["by_T_bucket"]:
            print("    %-10s %10d %13.3f%% %13.5f%%" % (
                r["stratum"], r["n"], r["pct_rows_in_dup_groups"],
                r["irreducible_sse_share_of_sst_pct"]))

    show_dup(dup, "(a) SETTLEMENT-DUPLICATE PAIRS  (key = date, log_m, T, r, q)")
    if dup_am is not None:
        show_dup(dup_am, "(a') SAME, KEY + am_settlement  (what retaining AM/PM separates)")
        print("\n    contrast: %.4f%% of rows collide on the model-input key alone, "
              "%.4f%% still collide once am_settlement is part of the key."
              % (dup["pct_rows_in_dup_groups"], dup_am["pct_rows_in_dup_groups"]))
    else:
        print("\n    (a') skipped: no `am_settlement` dataset in this file (v3 schema).")

    print("\n(b) BS STATIC-BOUND VIOLATIONS")
    print("    violations                 : %d  (%.4f%%)   lower=%.1f%% upper=%.1f%%" % (
        bnd["n_violations"], bnd["pct_violations"], bnd["pct_lower_bound"], bnd["pct_upper_bound"]))
    print("    depth (V/K units)          : " + "  ".join(
        "%s=%.3e" % (k, v) for k, v in bnd["depth"].items()))
    print("    n violations deeper than           : " + "  ".join(
        "%s=%d" % (k, v) for k, v in bnd["depth_tail_counts"].items()))
    print("    share of violations at T==0        : %.2f%%" % bnd["pct_violations_with_T0"])
    print("    share of violations at days<=1     : %.2f%%" % bnd["pct_violations_with_days_le_1"])
    print("    %-22s %10s %10s %12s %12s" % ("bucket", "n", "n_viol", "%viol", "med depth"))
    for r in bnd["by_T_bucket"] + bnd["by_moneyness"]:
        print("    %-22s %10d %10d %11.3f%% %12.3e" % (
            r.get("stratum", r.get("bucket")), r["n"], r["n_violations"],
            r["pct_violations"], r["median_depth"]))

    print("\n    top 20 violations by depth:")
    for r in bnd["top20_violations"]:
        print("      row=%-8d %s M=%9.5f d=%4d r=%.5f q=%.5f  V/K=%.6f  "
              "bound[%.6f,%.6f] %s depth=%.5f" % (
                  r["h5_row"], r["date"], r["moneyness"], r["days"], r["r"], r["q"],
                  r["normalized_price"], r["lower_bound"], r["upper_bound"],
                  r["bound_broken"], r["depth"]))

    out = {"dataset": prov, "duplicates": dup, "duplicates_with_am_settlement": dup_am,
           "bounds": bnd,
           "n_dates": int(len(np.unique(d["date"]))),
           "subsampled": max_contracts is not None}

    if do_trace and not bnd["top20_violations"]:
        print("\n(c) skipped: no static-bound violations to trace.")
    elif do_trace:
        rec = dict(bnd["top20_violations"][0], _option_type=option_type)
        print("\n(c) EXTREME OUTLIER -- tracing h5 row %d back to optionPrice_2015_2025.csv"
              % rec["h5_row"])
        tr = trace_raw(rec, data_dir)
        out["outlier_trace"] = {"h5_record": bnd["top20_violations"][0], "raw": tr}
        if "match" in tr:
            m = tr["match"]
            print("    spot close on %s      : %.2f" % (rec["date"], tr["spot"]))
            print("    strike implied by M       : %.3f" % tr["implied_strike"])
            print("    RAW RECORD:")
            for k in ("symbol", "optionid", "exdate", "cp_flag", "strike_price", "strike",
                      "best_bid", "best_offer", "mid", "volume", "open_interest",
                      "impl_volatility", "am_settlement", "exercise_style",
                      "contract_size", "cfadj", "days_to_expiry", "normalized_price"):
                print("      %-18s %s" % (k, m.get(k)))
            print("    h5 normalized_price=%.6f  vs raw mid/K=%.6f  (delta %.2e)" % (
                rec["normalized_price"], m["normalized_price"],
                abs(rec["normalized_price"] - m["normalized_price"])))
        else:
            print("    %s" % tr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--option-type", default="both", choices=["call", "put", "both"])
    ap.add_argument("--max-contracts", type=int, default=None)
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "wrds_data_2020-2025"),
                    help="directory holding the raw CSVs (used only by --trace-raw)")
    ap.add_argument("--h5-dir", default=None,
                    help="override the directory holding the HDF5 files (default: "
                         "whatever --dataset-version resolves to)")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--no-trace-raw", action="store_true",
                    help="skip the raw-CSV trace for the deepest violation (8.6 GB scan)")
    add_dataset_arg(ap)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = output_tag(args.dataset_version)
    types = ["call", "put"] if args.option_type == "both" else [args.option_type]

    report, dup_rows, bnd_rows = {}, [], []
    for ot in types:
        prov = dataset_provenance(args.dataset_version, ot, args.h5_dir)
        h5 = dataset_h5(args.dataset_version, ot, args.h5_dir)
        report[ot] = run(ot, h5, args.max_contracts, not args.no_trace_raw,
                         prov=prov, data_dir=args.data_dir)
        stamp = {"option_type": ot, **prov}
        for label, key in (("input_key", "duplicates"),
                           ("input_key+am_settlement", "duplicates_with_am_settlement")):
            if report[ot][key] is None:
                continue
            for r in report[ot][key]["by_T_bucket"]:
                dup_rows.append({**stamp, "grouping": label, **r})
        for r in report[ot]["bounds"]["by_T_bucket"]:
            bnd_rows.append({**stamp, "axis": "T", **r})
        for r in report[ot]["bounds"]["by_moneyness"]:
            bnd_rows.append({**stamp, "axis": "moneyness",
                             "stratum": r["bucket"], **{k: v for k, v in r.items()
                                                        if k != "bucket"}})
        for r in report[ot]["bounds"]["top20_violations"]:
            bnd_rows.append({**stamp, "axis": "top20", "stratum": "row%d" % r["h5_row"],
                             "n": 1, "n_violations": 1, "pct_violations": 100.0,
                             "pct_of_all_violations": 0.0, "median_depth": r["depth"]})

    (out / ("data_quality%s.json" % tag)).write_text(json.dumps(report, indent=2, default=str))
    _write_csv(out / ("data_quality_duplicates%s.csv" % tag), dup_rows)
    _write_csv(out / ("data_quality_bounds%s.csv" % tag), bnd_rows)
    print("\nwrote %s/{data_quality%s.json,data_quality_duplicates%s.csv,"
          "data_quality_bounds%s.csv}" % (out, tag, tag, tag))


if __name__ == "__main__":
    main()
