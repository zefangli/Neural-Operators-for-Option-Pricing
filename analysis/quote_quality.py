"""
Quote-quality preflight (read-only, CPU-only, model-free).

Quantifies candidate deterministic filters for economically meaningless raw WRDS
quotes.  **Applies nothing** -- it only counts what each rule *would* remove, so a
human can pick one before any rebuild.

Motivation.  v4 fixed maturity, contract identity and the T=0 degeneracy but does
not filter quotes.  `results/data_quality_v4.json` still shows call static-bound
violation depths of ~6.5 in normalized V/K units against a median of ~3e-4.  Both
extremes are stale one-sided quotes on deep-ITM LEAPS (bid collapsed to ~0 while
the ask sits near intrinsic), where `mid = (bid+ask)/2` is not a price.  Since the
target is `log(mid/K)`, those rows are unfittable by any sigma.

Rules evaluated (each reported alone, and incrementally beyond the others):
  bid_nonpos        best_bid <= 0
  crossed           best_bid > best_offer
  iv_missing        impl_volatility NaN (vendor declined)
  relspread_gt_X    (ask-bid)/mid > X for X in 0.5, 1, 2, 5
  bound_violation   mid/K outside the static no-arbitrage band (+ tolerance bands)
  combos            unions of the above

Note (ask-bid)/mid -> 2 as bid -> 0, so any threshold <= 2 subsumes bid_nonpos-ish
collapse; thresholds above 2 can only fire on crossed/degenerate quotes.

Usage
-----
    conda run -n dl_new python analysis/quote_quality.py --dataset-version v4
    conda run -n dl_new python analysis/quote_quality.py --dataset-version v4 \
        --max-contracts 200000 --no-trace-raw          # fast iteration
    conda run -n dl_new python analysis/quote_quality.py --self-check

Writes results/quote_quality{tag}.json and results/quote_quality{tag}.csv.
Never writes to any .h5/.parquet/.csv under wrds_data_2020-2025/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from _common import (PROJECT_ROOT, add_dataset_arg, dataset_h5, dataset_provenance,
                     output_tag)
from data_quality import M_BUCKETS, T_BUCKETS, _mask

RESULTS = PROJECT_ROOT / "results"
SECID = 108105
SPLITS = {0: "train", 1: "val", 2: "test"}

# Columns pulled from the HDF5.  The three branch grids (branch_u, spot_history,
# vix_history) are deliberately NOT read -- they are 99% of the file size.
COLS_2D = ("trunk_y", "moneyness", "normalized_price", "best_bid", "best_offer",
           "spread_norm", "impl_volatility", "strike", "T_calendar")
COLS_1D = ("split_id", "date", "optionid", "symbol", "root", "am_settlement")


def load(h5_path, max_contracts=None, seed=42):
    """Read the small columns of the whole file (all splits)."""
    with h5py.File(h5_path, "r") as f:
        n = f["split_id"].shape[0]
        idx = None
        if max_contracts is not None and n > max_contracts:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=max_contracts, replace=False))
        get = lambda k: (f[k][:] if idx is None else f[k][:][idx])
        d = {k: get(k).ravel() for k in COLS_1D}
        d["row"] = np.arange(n) if idx is None else idx
        trunk = get("trunk_y")
        d["log_m"] = trunk[:, 0].astype(np.float64)
        d["T"] = trunk[:, 1].astype(np.float64)
        d["r"] = trunk[:, 2].astype(np.float64)
        d["q"] = trunk[:, 3].astype(np.float64)
        for k in COLS_2D:
            if k == "trunk_y":
                continue
            d[k] = get(k).ravel().astype(np.float64)
    d["mid"] = 0.5 * (d["best_bid"] + d["best_offer"])
    d["days"] = np.rint(d["T_calendar"] * 365.0).astype(np.int64)
    d["subsampled"] = max_contracts is not None and idx is not None
    d["n_file_rows"] = n
    return d


def bounds(d, option_type):
    """Static no-arbitrage band in normalized V/K units, plus violation depth."""
    fwd = d["moneyness"] * np.exp(-d["q"] * d["T"])
    disc = np.exp(-d["r"] * d["T"])
    if option_type == "call":
        lo, hi = np.maximum(fwd - disc, 0.0), fwd
    else:
        lo, hi = np.maximum(disc - fwd, 0.0), disc
    depth = np.maximum(np.maximum(lo - d["price"], d["price"] - hi), 0.0)
    return lo, hi, depth


def build_rules(d, option_type):
    """-> (dict name->bool mask, dict of extras used for reporting)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        relspread = np.where(d["mid"] > 0, (d["best_offer"] - d["best_bid"]) / d["mid"],
                             np.inf)
    lo, hi, depth = bounds(d, option_type)
    r = {
        "bid_nonpos": d["best_bid"] <= 0.0,
        "crossed": d["best_bid"] > d["best_offer"],
        "iv_missing": ~np.isfinite(d["impl_volatility"]),
    }
    for x in (0.5, 1.0, 2.0, 5.0):
        r["relspread_gt_%g" % x] = relspread > x
    r["bound_violation"] = depth > 0.0
    for tol in (1e-4, 1e-3, 1e-2):
        r["bound_violation_tol_%g" % tol] = depth > tol
    # combinations
    r["combo_bid_or_crossed"] = r["bid_nonpos"] | r["crossed"]
    r["combo_bid_or_crossed_or_iv"] = r["combo_bid_or_crossed"] | r["iv_missing"]
    r["combo_bid_or_crossed_or_rs1"] = r["combo_bid_or_crossed"] | r["relspread_gt_1"]
    r["combo_bid_or_crossed_or_rs0.5"] = r["combo_bid_or_crossed"] | r["relspread_gt_0.5"]
    r["combo_bid_or_crossed_or_rs1_or_iv"] = (r["combo_bid_or_crossed_or_rs1"]
                                              | r["iv_missing"])
    r["combo_bid_or_crossed_or_rs1_or_bound"] = (r["combo_bid_or_crossed_or_rs1"]
                                                 | r["bound_violation"])
    return r, {"relspread": relspread, "lo": lo, "hi": hi, "depth": depth}


# ------------------------------------------------------------------ reporting


def _bucket_rows(mask, d, rule, option_type):
    """Per-split x {overall, maturity, moneyness} removal counts for one rule."""
    out = []
    for sid in (None, 0, 1, 2):
        sm = np.ones(mask.shape, bool) if sid is None else (d["split_id"] == sid)
        if not sm.any():
            continue
        split = "all" if sid is None else SPLITS[sid]
        groups = [("overall", "all", sm)]
        groups += [("maturity", lab, sm & _mask(d["days"], a, b))
                   for lab, a, b in T_BUCKETS]
        groups += [("moneyness", lab, sm & _mask(d["moneyness"], a, b))
                   for lab, a, b in M_BUCKETS]
        for dim, lab, g in groups:
            ng = int(g.sum())
            if ng == 0:
                continue
            nr = int((mask & g).sum())
            out.append({"option_type": option_type, "rule": rule, "split": split,
                        "dimension": dim, "bucket": lab, "n": ng, "n_removed": nr,
                        "pct_removed": 100.0 * nr / ng})
    return out


def incremental(rules, order):
    """How much each rule adds beyond the union of ALL the others (joint view)."""
    full = np.zeros_like(rules[order[0]])
    for k in order:
        full |= rules[k]
    out = {}
    for k in order:
        others = np.zeros_like(full)
        for j in order:
            if j != k:
                others |= rules[j]
        out[k] = {"alone": int(rules[k].sum()),
                  "unique_to_this_rule": int((rules[k] & ~others).sum())}
    out["_union"] = {"alone": int(full.sum()), "unique_to_this_rule": int(full.sum())}
    return out


def top_violations(d, extras, option_type, rules, k=20):
    depth = extras["depth"]
    order = np.argsort(-depth)[:k]
    order = order[depth[order] > 0]
    recs = []
    for i in order:
        recs.append({
            "h5_row": int(d["row"][i]), "date": d["date"][i].decode(),
            "optionid": int(d["optionid"][i]), "symbol": d["symbol"][i].decode().strip(),
            "root": d["root"][i].decode().strip(), "option_type": option_type,
            "strike": float(d["strike"][i]), "moneyness": float(d["moneyness"][i]),
            "log_moneyness": float(d["log_m"][i]),
            "T_years": float(d["T"][i]), "days_calendar": int(d["days"][i]),
            "r": float(d["r"][i]), "q": float(d["q"][i]),
            "best_bid": float(d["best_bid"][i]), "best_offer": float(d["best_offer"][i]),
            "mid": float(d["mid"][i]),
            "normalized_price": float(d["price"][i]),
            "impl_volatility": (None if not np.isfinite(d["impl_volatility"][i])
                                else float(d["impl_volatility"][i])),
            "am_settlement": int(d["am_settlement"][i]),
            "split": SPLITS.get(int(d["split_id"][i]), "?"),
            "lower_bound": float(extras["lo"][i]), "upper_bound": float(extras["hi"][i]),
            "bound_broken": "lower" if extras["lo"][i] > d["price"][i] else "upper",
            "depth": float(depth[i]),
            "relative_spread": float(extras["relspread"][i]),
            "caught_by": sorted(n for n, m in rules.items()
                                if m[i] and not n.startswith("combo_")),
        })
    return recs


def trace_raw(rec, data_dir):
    """Pull the source quote out of the 8.6 GB raw CSV by (secid, date, optionid).

    Filters are pushed into `scan_csv`, so polars streams and only the matching
    rows are materialised.  Read-only.
    """
    import polars as pl
    cols = ["date", "symbol", "exdate", "cp_flag", "strike_price", "best_bid",
            "best_offer", "volume", "open_interest", "impl_volatility", "delta",
            "optionid", "am_settlement", "exercise_style", "index_flag",
            "contract_size", "cfadj", "ss_flag"]
    lf = (pl.scan_csv(Path(data_dir) / "optionPrice_2015_2025.csv",
                      infer_schema_length=10000)
          .filter((pl.col("secid") == SECID) & (pl.col("date") == rec["date"])
                  & (pl.col("optionid") == rec["optionid"])))
    have = lf.collect_schema().names()
    raw = lf.select([c for c in cols if c in have]).collect(engine="streaming")
    if raw.height == 0:
        return {"error": "no raw row for optionid=%d on %s" % (rec["optionid"], rec["date"])}
    return raw.to_dicts()[0]


# --------------------------------------------------------------- self-check


def self_check():
    """Minimal assert-based check of the bound arithmetic (no framework)."""
    # Synthetic: S=100, K=100 (M=1), T=1, r=5%, q=2%.
    d = {"moneyness": np.array([1.0, 2.0, 0.5]), "T": np.full(3, 1.0),
         "r": np.full(3, 0.05), "q": np.full(3, 0.02)}
    fwd, disc = np.exp(-0.02), np.exp(-0.05)
    # a mid exactly on the call lower bound must not be a violation, one below must
    d["price"] = np.maximum(d["moneyness"] * fwd - disc, 0.0)
    _, _, dep = bounds(d, "call")
    assert np.allclose(dep, 0.0), dep
    d["price"] = np.maximum(d["moneyness"] * fwd - disc, 0.0) - 0.01
    lo, hi, dep = bounds(d, "call")
    assert np.allclose(dep[lo > 0], 0.01), dep          # only where lower bound > 0
    # put-call parity: C - P = fwd - disc in normalized units => a parity-consistent
    # pair sits inside both bands simultaneously.
    c = 0.5 * (np.maximum(d["moneyness"] * fwd - disc, 0.0)
               + d["moneyness"] * fwd)                  # mid-band (interior) call
    p = c - (d["moneyness"] * fwd - disc)               # parity partner
    d["price"] = c
    lo_c, hi_c, dc = bounds(d, "call")
    d["price"] = p
    lo_p, hi_p, dp = bounds(d, "put")
    assert np.allclose(dc, 0.0) and np.allclose(dp, 0.0), (dc, dp)
    assert np.allclose((c - p), d["moneyness"] * fwd - disc)
    # bands are non-empty and ordered
    assert np.all(lo_c <= hi_c) and np.all(lo_p <= hi_p)
    # relative spread -> 2 as bid -> 0 (the property the recommendation leans on)
    dd = {"mid": np.array([0.5 * (1e-9 + 100.0)]), "best_bid": np.array([1e-9]),
          "best_offer": np.array([100.0]), "impl_volatility": np.array([np.nan]),
          "moneyness": np.array([1.0]), "T": np.array([1.0]), "r": np.array([0.0]),
          "q": np.array([0.0]), "price": np.array([50.0])}
    rr, ex = build_rules(dd, "call")
    assert abs(ex["relspread"][0] - 2.0) < 1e-6, ex["relspread"]
    assert rr["relspread_gt_1"][0] and not rr["relspread_gt_2"][0]
    print("self-check OK")


# ------------------------------------------------------------------- driver


def analyse(option_type, args):
    h5 = dataset_h5(args.dataset_version, option_type)
    print("\n%s  <- %s" % (option_type.upper(), h5))
    d = load(h5, args.max_contracts)
    d["price"] = d["normalized_price"]
    n = d["price"].size
    print("  rows loaded: %d / %d in file%s"
          % (n, d["n_file_rows"], "  (SUBSAMPLED)" if d["subsampled"] else ""))

    rules, extras = build_rules(d, option_type)
    rep = {"option_type": option_type, "n_rows": int(n),
           "n_file_rows": int(d["n_file_rows"]), "subsampled": bool(d["subsampled"]),
           "provenance": dataset_provenance(args.dataset_version, option_type),
           "split_sizes": {SPLITS[s]: int((d["split_id"] == s).sum())
                           for s in (0, 1, 2)},
           "rules": {}, "top20_violations": []}

    csv_rows = []
    for name, m in rules.items():
        per_split = {}
        for s in (0, 1, 2):
            sm = d["split_id"] == s
            per_split[SPLITS[s]] = {"n": int(sm.sum()),
                                    "n_removed": int((m & sm).sum()),
                                    "pct_removed": 100.0 * float((m & sm).sum())
                                    / max(int(sm.sum()), 1)}
        rep["rules"][name] = {
            "n_removed": int(m.sum()), "pct_removed": 100.0 * float(m.mean()),
            "by_split": per_split,
            # the differential that actually matters: test-minus-train removal rate
            "test_minus_train_pct": per_split["test"]["pct_removed"]
                                    - per_split["train"]["pct_removed"],
        }
        csv_rows += _bucket_rows(m, d, name, option_type)
        print("  %-34s removes %9d (%6.3f%%)   train %.3f%% val %.3f%% test %.3f%%"
              % (name, m.sum(), 100.0 * m.mean(), per_split["train"]["pct_removed"],
                 per_split["val"]["pct_removed"], per_split["test"]["pct_removed"]))

    base = ["bid_nonpos", "crossed", "iv_missing", "relspread_gt_1", "bound_violation"]
    rep["incremental_base"] = incremental(rules, base)
    rep["incremental_recommended"] = incremental(
        rules, ["bid_nonpos", "crossed", "relspread_gt_1"])
    # pairwise overlap of the primitive rules
    rep["pairwise_overlap"] = {
        "%s & %s" % (a, b): int((rules[a] & rules[b]).sum())
        for i, a in enumerate(base) for b in base[i + 1:]}

    rep["top20_violations"] = top_violations(d, extras, option_type, rules)
    if rep["top20_violations"]:
        deepest = rep["top20_violations"][0]
        print("  deepest violation: %s optionid=%d K=%.0f depth=%.4f bid=%.2f ask=%.2f"
              % (deepest["date"], deepest["optionid"], deepest["strike"],
                 deepest["depth"], deepest["best_bid"], deepest["best_offer"]))
        if args.trace_raw:
            try:
                deepest["raw_csv_record"] = trace_raw(
                    deepest, PROJECT_ROOT / "wrds_data_2020-2025")
                print("  raw CSV: %s" % deepest["raw_csv_record"])
            except Exception as e:                       # noqa: BLE001
                deepest["raw_csv_record"] = {"error": repr(e)}
                print("  raw CSV trace failed: %r" % e)
    return rep, csv_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(ap)
    ap.add_argument("--max-contracts", type=int, default=None,
                    help="subsample for fast iteration; final numbers use the full file")
    ap.add_argument("--no-trace-raw", dest="trace_raw", action="store_false")
    ap.add_argument("--self-check", action="store_true", help="run asserts and exit")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    self_check()

    report, csv_rows = {}, []
    for ot in ("call", "put"):
        rep, rows = analyse(ot, args)
        report[ot] = rep
        csv_rows += rows

    tag = output_tag(args.dataset_version)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / ("quote_quality%s.json" % tag)).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    import csv as _csv
    with open(RESULTS / ("quote_quality%s.csv" % tag), "w", newline="",
              encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("\nwrote %s/quote_quality%s.{json,csv}   (NOTHING was filtered or rebuilt)"
          % (RESULTS, tag))


if __name__ == "__main__":
    main()
