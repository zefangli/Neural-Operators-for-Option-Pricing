#!/usr/bin/env python
"""
Why the cubic-spline IV-surface baseline scores R2(price)=0.9966 and
R2(log)=-2.5186 on puts: decompose its test-set error by which side of the
OptionMetrics grid boundary each row falls on.

CPU only. No training, no checkpoint, no GPU. Every number is recomputed from
the canonical v5 test split through analysis/baselines.py's own functions --
the same cubic RegularGridInterpolator, the same damped fixed point on delta
(damping 0.7, initialised at the |delta|=50 column, tol 1e-6, 80 iterations
max), the same boundary clamping, the same log(clip(V/K, 1e-8)) clamp and the
same `_common.compute_metrics` schema. Nothing here re-implements the lookup;
if this file and baselines.py ever disagree the run aborts (see --tol).

    python analysis/spline_boundary_decomposition.py --dataset-version v5
    python analysis/spline_boundary_decomposition.py --dataset-version v5 --option-type put

Writes results/spline_boundary_decomposition_{version}.json and .md.

MASK DEFINITIONS (spelled out because the delta a row is classified by is not
an input, it is a solved quantity):

  * The delta a row is classified by is the delta of the CONVERGED sigma -- i.e.
    `bs_delta_pct(log_m, T, r, q, sigma_final, option_type)`, recomputed after
    the fixed point stops, exactly the array baselines.py decides
    `n_out_of_grid_delta` on. It is NOT the |delta|=50 initial guess, and it is
    NOT a delta implied by the market quote.
  * below_10_delta : |delta_pct| < 10, the low-|delta| (deep-OTM) side. The grid
    stops at 10, so these rows are priced with the 10-delta column's IV.
  * above_90_delta : |delta_pct| > 90, the high-|delta| (deep-ITM) side, priced
    with the 90-delta column's IV.
  * tenor_out      : T*365 outside [10, 730] calendar days, the grid's tenor
    span, so the row is priced with the 10-day or 730-day row's IV.
  * in_grid        : none of the above -- clamped on NEITHER axis.
  * delta_in_grid  : not below_10_delta and not above_90_delta, i.e. in-grid on
    the DELTA axis alone, tenor clamping ignored. Reported alongside in_grid
    because "the in-grid rows" is ambiguous between the two and they give
    different R2(log) (put: 0.9939 strict, 0.9843 delta-only); a manuscript
    sentence must say which one it means.
  * Boundaries are EXCLUSIVE on both sides, matching baselines.py: a row at
    exactly |delta| = 10 or 90, or at exactly 10 or 730 days, is in-grid.
  * The delta masks are mutually exclusive with each other, but EITHER can
    co-occur with tenor_out -- a 5-day 3-delta put is clamped on both axes.
    below_10_delta / above_90_delta / tenor_out therefore do NOT partition the
    split and their SSE shares do not sum to 100%. The partition that does is
    {in_grid, not in_grid}; the overlap counts are reported explicitly under
    "overlaps" so a reader can reconstruct the disjoint cells.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402  (path shim above)
from _common import PROJECT_ROOT, compute_metrics, dataset_provenance  # noqa: E402
from eval_to_json import sha256_of  # noqa: E402

BASELINE = "surface_spline"
INTERP_METHOD = "cubic"          # the method baselines.py labels "spline"
SQ_LOG_TAIL = (10.0, 1.0)        # squared-log-error thresholds reported
REL_PRICE_TAIL = 1.0             # |relative price error| > 100%


# ---------------------------------------------------------------- pure masks

def boundary_masks(delta_pct, T_years, delta_axis, days_axis):
    """-> {mask name: bool array}, on the SAME boundaries baselines.py clamps to.

    `delta_pct` is the converged-sigma delta in delta-percent (signed: positive
    for calls, negative for puts -- only |.| is used). Boundaries exclusive:
    exactly 10 / 90 delta and exactly 10 / 730 days are in-grid.
    """
    a = np.abs(np.asarray(delta_pct, dtype=np.float64))
    d = np.abs(np.asarray(delta_axis, dtype=np.float64))
    days = np.asarray(T_years, dtype=np.float64) * 365.0
    t = np.asarray(days_axis, dtype=np.float64)
    below = a < d.min()
    above = a > d.max()
    tenor = (days < t.min()) | (days > t.max())
    return {"below_10_delta": below, "above_90_delta": above, "tenor_out": tenor,
            "in_grid": ~(below | above | tenor), "delta_in_grid": ~(below | above)}


def mask_stats(mask, sq_log, sq_price, rel_price_err, log_pred, log_target, T):
    """Row counts, SSE shares, within-mask R2 and tail counts for one mask."""
    mask = np.asarray(mask, dtype=bool)
    n_all = mask.size
    n = int(mask.sum())
    tot_log = float(sq_log.sum())
    tot_price = float(sq_price.sum())
    out = {
        "n_rows": n,
        "pct_rows": 100.0 * n / n_all if n_all else 0.0,
        "share_log_sse_pct": 100.0 * float(sq_log[mask].sum()) / tot_log if tot_log else 0.0,
        "share_price_sse_pct": (100.0 * float(sq_price[mask].sum()) / tot_price
                                if tot_price else 0.0),
        "n_sqerr_log_gt10": int((sq_log[mask] > SQ_LOG_TAIL[0]).sum()),
        "n_sqerr_log_gt1": int((sq_log[mask] > SQ_LOG_TAIL[1]).sum()),
        "n_rel_price_err_gt_100pct": int((rel_price_err[mask] > REL_PRICE_TAIL).sum()),
    }
    if n:
        m = compute_metrics(log_pred[mask], log_target[mask], T[mask])
        out["r2_log"] = m["r2_log_full"]
        out["r2_price"] = m["r2_price"]
        out["rmse_log"] = m["rmse_log"]
        out["rmse_price"] = m["rmse_price"]
        out["mean_price_true"] = float(np.mean(np.exp(log_target[mask])))
    else:
        out.update({k: None for k in ("r2_log", "r2_price", "rmse_log", "rmse_price",
                                      "mean_price_true")})
    return out


# ---------------------------------------------------------------- the run

def expected_metrics(option_type, baselines_json):
    """The surface_spline 'all' metrics recorded in results/baselines_v5.json."""
    recs = json.loads(Path(baselines_json).read_text())["records"]
    hit = [r for r in recs if r["baseline"] == BASELINE and r["option_type"] == option_type]
    if len(hit) != 1:
        raise SystemExit("%s: expected exactly one %s/%s record, found %d"
                         % (baselines_json, option_type, BASELINE, len(hit)))
    return hit[0]


def decompose(option_type, surfaces, args, prov, expected):
    surf = surfaces[option_type]
    rows = B.load_split(option_type, "test")
    n = len(rows["T"])
    t0 = time.time()
    rowwise = {}
    sigma, diag = B.surface_sigma(rows, surf, option_type, method=INTERP_METHOD,
                                  rowwise=rowwise)
    print(f"[{option_type}] {n} test rows, surface lookup {time.time() - t0:.0f}s")

    log_pred = B.log_price(rows["log_m"], rows["T"], rows["r"], rows["q"], sigma, option_type)
    assert np.isfinite(log_pred).all(), "non-finite predicted log price"
    log_target = rows["target_v_log"]
    T = rows["T"]

    got = compute_metrics(log_pred, log_target, T)
    for key, want_key in (("r2_price", "r2_price"), ("r2_log_full", "r2_log_full")):
        want = expected["metrics"]["all"][want_key]
        if not abs(got[key] - want) <= args.tol:
            raise SystemExit(
                "ABORT: recomputed %s/%s %s=%.12f disagrees with %s (%.12f) by %.3g > %g.\n"
                "  This script must reproduce the published baseline bit-for-bit before its\n"
                "  decomposition may be cited. Do not publish these numbers."
                % (option_type, BASELINE, key, got[key], args.baselines_json, want,
                   abs(got[key] - want), args.tol))
    print(f"[{option_type}] sanity OK: R2(price)={got['r2_price']:.6f} "
          f"R2(log)={got['r2_log_full']:.6f} (matches {Path(args.baselines_json).name})")

    # Cross-check the mask boundaries against baselines.py's own clamp decision:
    # the two delta masks must together be exactly its `oob_delta`, and tenor_out
    # exactly its `oob_tenor`. If this fails the masks are not the clamped rows.
    masks = boundary_masks(rowwise["delta_pct"], T, surf["delta"], surf["days"])
    assert np.array_equal(masks["below_10_delta"] | masks["above_90_delta"],
                          rowwise["oob_delta"]), "delta masks != baselines.py oob_delta"
    assert np.array_equal(masks["tenor_out"], rowwise["oob_tenor"]), \
        "tenor mask != baselines.py oob_tenor"

    resid = log_pred - log_target
    sq_log = resid ** 2
    p_pred, p_true = np.exp(log_pred), np.exp(log_target)
    sq_price = (p_pred - p_true) ** 2
    rel_price_err = np.abs(p_pred - p_true) / p_true

    per_mask = {}
    for name, m in masks.items():
        per_mask[name] = mask_stats(m, sq_log, sq_price, rel_price_err,
                                    log_pred, log_target, T)
        per_mask["not_" + name] = mask_stats(~m, sq_log, sq_price, rel_price_err,
                                             log_pred, log_target, T)

    overlaps = {
        "below_10_delta_and_tenor_out": int((masks["below_10_delta"] & masks["tenor_out"]).sum()),
        "above_90_delta_and_tenor_out": int((masks["above_90_delta"] & masks["tenor_out"]).sum()),
        "below_10_delta_and_above_90_delta":
            int((masks["below_10_delta"] & masks["above_90_delta"]).sum()),
        "any_out_of_grid": int((~masks["in_grid"]).sum()),
        "delta_only": int(((masks["below_10_delta"] | masks["above_90_delta"])
                           & ~masks["tenor_out"]).sum()),
        "tenor_only": int((masks["tenor_out"]
                           & ~(masks["below_10_delta"] | masks["above_90_delta"])).sum()),
        "note": ("the delta masks are mutually exclusive; either may co-occur with "
                 "tenor_out, so the three do not partition the split. {in_grid, "
                 "not_in_grid} does, and its two SSE shares sum to 100%."),
    }

    return {
        "option_type": option_type,
        "baseline": BASELINE,
        "provenance": prov,
        "n_test_rows": n,
        "grid": {"delta_pct_axis": [float(x) for x in surf["delta"]],
                 "days_axis": [float(x) for x in surf["days"]],
                 "abs_delta_bounds": [float(np.abs(surf["delta"]).min()),
                                      float(np.abs(surf["delta"]).max())],
                 "day_bounds": [float(surf["days"].min()), float(surf["days"].max())]},
        "aggregate": {"r2_price": got["r2_price"], "r2_log_full": got["r2_log_full"],
                      "rmse_log": got["rmse_log"], "rmse_price": got["rmse_price"],
                      "n_tail_sqerr_gt10": got["n_tail_sqerr_gt10"],
                      "total_log_sse": float(sq_log.sum()),
                      "total_price_sse": float(sq_price.sum())},
        "aggregate_expected": {"r2_price": expected["metrics"]["all"]["r2_price"],
                               "r2_log_full": expected["metrics"]["all"]["r2_log_full"],
                               "source": str(args.baselines_json),
                               "tolerance": args.tol},
        "baselines_diagnostics": diag,
        "masks": per_mask,
        "overlaps": overlaps,
    }


MECHANISM = (
    "Mechanism. The OptionMetrics surface is standardized on |delta| in [10, 90] and "
    "10 to 730 calendar days, and baselines.py clamps any query outside that box to the "
    "nearest boundary rather than extrapolating. For puts, {below_pct:.2f}% of the test rows "
    "converge to |delta| below 10 and are therefore all priced off the single 10-delta "
    "column, an implied volatility that is far too low for the wing they actually sit in. "
    "Those rows carry {below_log_sse:.2f}% of the total log-space SSE and "
    "{below_tail}/{tail_all} of the squared-log-error observations above 10, which is what "
    "drives the aggregate R2(log) to {r2_log:.4f}; on the rows clamped on neither axis the "
    "same spline reaches R2(log) = {in_grid_r2_log:.4f} ({delta_in_grid_r2_log:.4f} if only "
    "the delta clamp is excluded and tenor-clamped rows are kept). The error there is "
    "multiplicative, and the price scale it acts on is small: the mean true normalized price "
    "on those rows is {below_mean:.3g} against {in_grid_mean:.3g} on the in-grid rows and "
    "{above_mean:.3g} on the deep-ITM rows, so a mispricing worth a squared log error above "
    "10 -- a factor of {factor:.0f} or more on the price -- still only costs "
    "{below_rmse:.3g} in RMSE(V/K). Those rows do carry {below_price_sse:.2f}% of the "
    "price-space SSE, a large share and not a negligible one, but R2(price) is scored against "
    "the total variance of V/K, which the in-the-money rows set; that denominator is large "
    "enough that {below_price_sse:.2f}% of the residual sum still leaves R2(price) = "
    "{r2_price:.6f}. The two metrics are not in conflict -- they weight the same errors on "
    "different scales -- and a single clamped boundary column is what separates them."
)


def markdown(results, meta):
    cols = ["mask", "n_rows", "%_rows", "%_log_SSE", "%_price_SSE", "R2(log)", "R2(price)",
            "n_sq_log>10", "n_sq_log>1", "n_rel_px_err>100%"]
    out = ["# Cubic-spline IV-surface baseline: grid-boundary error decomposition",
           "",
           f"Generated {meta['generated']} - CPU only, no checkpoint loaded.",
           f"Dataset {meta['dataset_version']}, script `{meta['script']}`, "
           f"git HEAD `{meta['git_head'][:12]}`.",
           "",
           "Rows are classified by the delta of the **converged** sigma from the damped "
           "fixed point (the same array `baselines.py` decides `n_out_of_grid_delta` on), "
           "not by the |delta|=50 initial guess. Boundaries are exclusive: exactly 10 or 90 "
           "delta, and exactly 10 or 730 days, count as in-grid. The two delta masks are "
           "mutually exclusive but either may co-occur with `tenor_out`, so the three "
           "out-of-grid masks do not partition the split -- only "
           "`{in_grid, not_in_grid}` does, and only that pair's SSE shares sum to 100%. "
           "`delta_in_grid` is the looser reading of \"in-grid\" -- in-grid on the delta "
           "axis alone, tenor clamping ignored; it is listed because the two readings give "
           "different R2(log) and a citation must say which it means.",
           ""]
    for res in results:
        ot = res["option_type"]
        agg = res["aggregate"]
        out += [f"## {ot}", "",
                f"{res['n_test_rows']:,} test rows. Aggregate R2(price) = "
                f"{agg['r2_price']:.6f}, R2(log) = {agg['r2_log_full']:.6f} "
                f"(reproduces `{Path(res['aggregate_expected']['source']).name}` to "
                f"{res['aggregate_expected']['tolerance']:g}).",
                "",
                "| " + " | ".join(cols) + " |",
                "|" + "|".join(["---"] * len(cols)) + "|"]
        order = ["below_10_delta", "above_90_delta", "tenor_out", "in_grid", "not_in_grid",
                 "delta_in_grid", "not_delta_in_grid"]
        for name in order:
            s = res["masks"][name]
            out.append("| `%s` | %s | %.2f | %.2f | %.2f | %s | %s | %s | %s | %s |" % (
                name, f"{s['n_rows']:,}", s["pct_rows"], s["share_log_sse_pct"],
                s["share_price_sse_pct"],
                "n/a" if s["r2_log"] is None else f"{s['r2_log']:.4f}",
                "n/a" if s["r2_price"] is None else f"{s['r2_price']:.6f}",
                f"{s['n_sqerr_log_gt10']:,}", f"{s['n_sqerr_log_gt1']:,}",
                f"{s['n_rel_price_err_gt_100pct']:,}"))
        ov = res["overlaps"]
        out += ["",
                f"Overlaps: {ov['below_10_delta_and_tenor_out']:,} rows are both "
                f"below-10-delta and tenor-clamped, {ov['above_90_delta_and_tenor_out']:,} "
                f"both above-90-delta and tenor-clamped, "
                f"{ov['below_10_delta_and_above_90_delta']:,} in both delta masks (0 by "
                f"construction). {ov['any_out_of_grid']:,} rows are clamped on at least one "
                f"axis: {ov['delta_only']:,} delta-only, {ov['tenor_only']:,} tenor-only, "
                f"{ov['below_10_delta_and_tenor_out'] + ov['above_90_delta_and_tenor_out']:,} "
                f"both.",
                "",
                "Baseline diagnostics recomputed here (must match "
                "`baselines_%s.csv`): n_out_of_grid_delta = %d, n_out_of_grid_tenor = %d."
                % (meta["dataset_version"], res["baselines_diagnostics"]["n_out_of_grid_delta"],
                   res["baselines_diagnostics"]["n_out_of_grid_tenor"]),
                ""]
    put = next((r for r in results if r["option_type"] == "put"), None)
    if put is not None:
        b = put["masks"]["below_10_delta"]
        out += ["## Mechanism", "",
                MECHANISM.format(
                    below_pct=b["pct_rows"], below_log_sse=b["share_log_sse_pct"],
                    below_price_sse=b["share_price_sse_pct"],
                    below_tail=f"{b['n_sqerr_log_gt10']:,}",
                    tail_all=f"{put['aggregate']['n_tail_sqerr_gt10']:,}",
                    r2_log=put["aggregate"]["r2_log_full"],
                    r2_price=put["aggregate"]["r2_price"],
                    in_grid_r2_log=put["masks"]["in_grid"]["r2_log"],
                    delta_in_grid_r2_log=put["masks"]["delta_in_grid"]["r2_log"],
                    below_mean=b["mean_price_true"], below_rmse=b["rmse_price"],
                    in_grid_mean=put["masks"]["in_grid"]["mean_price_true"],
                    above_mean=put["masks"]["above_90_delta"]["mean_price_true"],
                    factor=np.exp(np.sqrt(SQ_LOG_TAIL[0]))),
                ""]
    return "\n".join(out)


def git_head():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True)
        return r.stdout.strip(), bool(dirty.stdout.strip())
    except Exception as exc:                      # not a checkout / no git on PATH
        return "unavailable (%s)" % exc, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--option-type", choices=["call", "put", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--h5-dir", default=None)
    ap.add_argument("--baselines-json", type=Path, default=None,
                    help="published baselines JSON to reproduce (default: "
                         "results/baselines_<version>.json)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed |recomputed - published| aggregate R2 difference")
    ap.add_argument("--skip-hash", action="store_true",
                    help="do not re-hash the HDF5 (the artifact then records the expected "
                         "hash and 'not verified'; never use for a publishable run)")
    B.add_dataset_arg(ap)
    args = ap.parse_args()
    if args.baselines_json is None:
        args.baselines_json = PROJECT_ROOT / "results" / (
            "baselines%s.json" % B.output_tag(args.dataset_version))
    B._DATASET.update(version=args.dataset_version, h5_dir=args.h5_dir)

    surfaces = B.load_surfaces()
    types = ["call", "put"] if args.option_type == "both" else [args.option_type]
    head, dirty = git_head()
    results = []
    for ot in types:
        prov = dict(dataset_provenance(args.dataset_version, ot, args.h5_dir))
        if args.skip_hash:
            prov["observed_sha256"] = None
            prov["sha256_verified"] = "skipped (--skip-hash)"
        else:
            t0 = time.time()
            prov["observed_sha256"] = sha256_of(prov["h5_path"])
            print(f"[{ot}] sha256 {prov['observed_sha256'][:12]}... "
                  f"({time.time() - t0:.0f}s)")
            if prov["observed_sha256"] != prov["expected_sha256"]:
                raise SystemExit(
                    "ABORT: %s hashes to %s but the manifest declares %s -- this is not the "
                    "canonical %s sample." % (prov["h5_path"], prov["observed_sha256"],
                                              prov["expected_sha256"], args.dataset_version))
            prov["sha256_verified"] = True
        results.append(decompose(ot, surfaces, args, prov,
                                 expected_metrics(ot, args.baselines_json)))

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "analysis/spline_boundary_decomposition.py",
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "baselines_py_sha256": sha256_of(Path(__file__).resolve().parent / "baselines.py"),
        "baselines_py_marker": ("analysis/baselines.py has no __version__; identity is the "
                                "sha256 above plus the git HEAD"),
        "git_head": head,
        "git_worktree_dirty": dirty,
        "dataset_version": args.dataset_version,
        "device": "cpu",
        "reproduces": str(args.baselines_json),
        "tolerance": args.tol,
        "metric_source": "analysis/_common.compute_metrics (unmodified)",
        "lookup_source": ("analysis/baselines.py surface_sigma(method='cubic'), behaviour "
                          "unchanged: the per-row masks come from its `rowwise` "
                          "out-parameter, and the aggregate R2 is asserted against the "
                          "published baselines JSON before anything here is written"),
        "mask_definitions": __doc__.split("MASK DEFINITIONS")[1].strip(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "spline_boundary_decomposition" + B.output_tag(args.dataset_version)
    (args.out_dir / (stem + ".json")).write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    (args.out_dir / (stem + ".md")).write_text(markdown(results, meta), encoding="utf-8")
    print("wrote %s.{json,md}" % (args.out_dir / stem))


if __name__ == "__main__":
    main()
