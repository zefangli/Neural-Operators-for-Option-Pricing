"""Validation report for the v4 / v5 datasets.

v4 = unfiltered robustness sample. v5 = canonical: same tensor schema
(schema_version stays "v4"), rows restricted to quotes that satisfy the static
no-arbitrage bounds. Pass --dataset-version to pick; the report file is named
after it, so a v5 run can never overwrite VALIDATION_v4.json.

Read-only. Verifies the full v4 build against its own provenance and against the
raw sources, and writes a machine-readable report next to the HDF5 files.

Checks (each prints PASS/FAIL and lands in the JSON):
  1. Row counts: parquet parts vs HDF5, per split.
  2. Split integrity: date boundaries, time-ordering, no date in two splits.
  3. Hashes: raw VIX CSV, and each HDF5.
  4. Provenance attrs: schema_version, t_basis, maturity threshold, VIX source.
  5. Finite/null checks on every required dataset.
  6. Every exported T is above the configured minimum maturity.
  7. Raw-VIX -> HDF5 spot checks on early / middle / late dates.
  8. Contract identity + duplicate-input rate (the v3 defect this build fixes).
  9. VXN / VXD / VVIX cannot have entered any tensor.

  10. (v5 only) dataset_version + quote-filter attrs, and ZERO surviving
      static-bound violations.

Usage:
    python validate_v4.py                            # full v4 files
    python validate_v4.py --dataset-version v5       # full v5 files
    python validate_v4.py --smoke                    # smoke files
    python validate_v4.py --out report_v4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
VIX_CSV = "vix_2015_2025.csv"
EXPECTED_VIX_SHA = "e2f18c516c9d48fb1730c2c0203fce36900c70aa0635b99e64afdde157346301"
# VVIX ran 60-200 over this period and VIX 9-83, so a max above this is the
# tell that the wrong series came back.
VIX_LEVEL_MAX = 90.0

REQUIRED = ["branch_u", "spot_history", "vix_history", "trunk_y", "target_v_log",
            "moneyness", "normalized_price", "date", "split_id"]

results: list[dict] = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _s(attrs, key):
    v = attrs.get(key, "")
    return v.decode() if isinstance(v, bytes) else str(v)


def bound_violations(f, opt_type: str) -> dict:
    """Static no-arbitrage bounds on the stored (float32) midpoints, float64 math.

    V/K units: fwd = moneyness*exp(-q*T), disc = exp(-r*T);
    call in [max(fwd-disc,0), fwd], put in [max(disc-fwd,0), disc].
    """
    tr = f["trunk_y"][:].astype(np.float64)
    T, r, q = tr[:, 1], tr[:, 2], tr[:, 3]
    M = f["moneyness"][:].astype(np.float64).ravel()
    v = f["normalized_price"][:].astype(np.float64).ravel()
    fwd, disc = M * np.exp(-q * T), np.exp(-r * T)
    lo, hi = (np.maximum(fwd - disc, 0.0), fwd) if opt_type == "call" else \
             (np.maximum(disc - fwd, 0.0), disc)
    below, above = v < lo, v > hi
    gap = np.maximum(lo - v, 0.0) + np.maximum(v - hi, 0.0)
    n = len(v)
    return {"rows": n, "below_lower": int(below.sum()), "above_upper": int(above.sum()),
            "violations": int((below | above).sum()),
            "violation_pct": 100.0 * float((below | above).sum()) / max(n, 1),
            "max_gap": float(gap.max()) if n else 0.0}


def validate_one(h5_path: Path, opt_type: str, parts_dir: Path | None, dataset_version: str):
    print(f"\n=== {h5_path.name} ({opt_type}) ===")
    out: dict = {"file": h5_path.name, "option_type": opt_type}

    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        n = f["date"].shape[0]
        out["rows"] = int(n)

        # --- 4. provenance ---
        schema = _s(attrs, "schema_version")
        check(f"{opt_type}: schema_version == v4", schema == "v4", schema)
        t_basis = _s(attrs, "export_t_basis") or _s(attrs, "phase1_t_basis")
        check(f"{opt_type}: settlement basis recorded", t_basis == "settlement", t_basis)
        min_mat = float(attrs.get("export_min_maturity_days", float("nan")))
        out["min_maturity_days"] = min_mat
        check(f"{opt_type}: maturity threshold recorded", np.isfinite(min_mat), min_mat)
        vix_series = _s(attrs, "phase1_vix_source_series")
        check(f"{opt_type}: VIX source is CBOE VIX", "CBOE VIX" in vix_series, vix_series)
        vix_sha = _s(attrs, "phase1_vix_sha256").lower()
        check(f"{opt_type}: VIX CSV sha matches reviewed file",
              vix_sha == EXPECTED_VIX_SHA, vix_sha[:16] + "...")
        out["provenance"] = {
            "schema_version": schema, "t_basis": t_basis,
            "vix_source_series": vix_series, "vix_sha256": vix_sha,
            "vix_raw_row_count": _s(attrs, "phase1_vix_raw_row_count"),
            "vix_usable_row_count": _s(attrs, "phase1_vix_usable_row_count"),
        }

        # --- 10. dataset_version + quote filter (v5) ---
        dsv = _s(attrs, "dataset_version")
        out["dataset_version"] = dsv
        qf = _s(attrs, "quote_filter")
        out["quote_filter"] = qf
        if dataset_version == "v5":
            check(f"{opt_type}: dataset_version == v5", dsv == "v5", dsv)
            check(f"{opt_type}: quote_filter recorded",
                  qf == "static_bounds_midpoint", qf)
            tol = float(attrs.get("quote_filter_tolerance", float("nan")))
            check(f"{opt_type}: quote_filter_tolerance == 0", tol == 0.0, tol)
            doc = _s(attrs, "quote_filter_bounds")
            check(f"{opt_type}: bound convention documented",
                  "fwd" in doc and "disc" in doc, doc[:60] + "...")
            pre = json.loads(_s(attrs, "rows_prefilter_by_split") or "{}")
            post = json.loads(_s(attrs, "rows_by_split") or "{}")
            out["rows_prefilter_by_split"] = pre
            out["rows_by_split"] = post
            ok_counts = bool(pre) and bool(post) and all(
                pre.get(k, -1) >= post.get(k, 0) > 0 for k in ("train", "val", "test"))
            removed = sum(pre.values()) - sum(post.values())
            check(f"{opt_type}: pre/post filter counts recorded and consistent", ok_counts,
                  f"{sum(pre.values())} -> {sum(post.values())} "
                  f"({100.0 * removed / max(sum(pre.values()), 1):.4f}% removed)")
            lin = _s(attrs, "lineage_v4_h5_sha256")
            check(f"{opt_type}: lineage recorded", len(lin) == 64,
                  f"{_s(attrs, 'lineage_v4_h5')} {lin[:16]}...")
        else:
            check(f"{opt_type}: not mislabelled as v5", dsv != "v5", dsv or "(absent)")

        viol = bound_violations(f, opt_type)
        out["static_bounds"] = viol
        print(f"      static bounds: {viol['violations']} violations "
              f"({viol['violation_pct']:.4f}%), max gap {viol['max_gap']:.3e}")
        if dataset_version == "v5":
            check(f"{opt_type}: zero static-bound violations survive the filter",
                  viol["violations"] == 0,
                  f"{viol['violations']} rows (below {viol['below_lower']}, "
                  f"above {viol['above_upper']}), max gap {viol['max_gap']:.3e}")

        # --- 5. finite / null ---
        missing = [d for d in REQUIRED if d not in f]
        check(f"{opt_type}: all required datasets present", not missing, missing or "all present")
        bad = {}
        for d in REQUIRED:
            if d in ("date", "split_id") or d not in f:
                continue
            arr = f[d][:]
            nfin = int((~np.isfinite(arr)).sum())
            if nfin:
                bad[d] = nfin
        check(f"{opt_type}: no NaN/Inf in float datasets", not bad, bad or "clean")
        out["nonfinite"] = bad

        # --- 6. maturity floor ---
        T = f["trunk_y"][:, 1].astype(np.float64)
        tdays = T * 365.0
        out["min_T_days"] = float(tdays.min())
        out["max_T_days"] = float(tdays.max())
        if np.isfinite(min_mat):
            check(f"{opt_type}: every T > {min_mat} days",
                  tdays.min() > min_mat, f"min = {tdays.min():.4f} days")
        check(f"{opt_type}: no T <= 0 rows", (tdays > 0).all(), f"min = {tdays.min():.6f}")

        # --- 2. split integrity ---
        dates = f["date"][:].astype("U10")
        split = f["split_id"][:]
        check(f"{opt_type}: split ids are exactly {{0,1,2}}",
              set(np.unique(split).tolist()) <= {0, 1, 2}, np.unique(split).tolist())
        bounds = {}
        for sid, nm in ((0, "train"), (1, "val"), (2, "test")):
            m = split == sid
            if m.any():
                # np.unique sorts; ndarray.min() has no ufunc loop for '<U10'
                u = np.unique(dates[m])
                bounds[nm] = {"rows": int(m.sum()), "n_dates": int(len(u)),
                              "first": str(u[0]), "last": str(u[-1])}
        out["splits"] = bounds
        for nm, b in bounds.items():
            print(f"      {nm:5s} {b['rows']:>10,d} rows  {b['n_dates']:>4d} dates  "
                  f"{b['first']} .. {b['last']}")
        sets = {nm: set(np.unique(dates[split == sid]).tolist())
                for sid, nm in ((0, "train"), (1, "val"), (2, "test"))}
        overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
        check(f"{opt_type}: no date appears in two splits", not overlap, f"{len(overlap)} overlapping")
        ordered = bounds["train"]["last"] < bounds["val"]["first"] <= bounds["val"]["last"] < bounds["test"]["first"]
        check(f"{opt_type}: splits are time-ordered", ordered,
              f"{bounds['train']['last']} < {bounds['val']['first']} .. {bounds['test']['last']}")

        # --- 8. contract identity / duplicate inputs ---
        tr = f["trunk_y"][:].astype(np.float64)
        te = split == 2
        cols = [dates[te], tr[te, 0], tr[te, 1], tr[te, 2], tr[te, 3]]
        # lexsort takes the last key as primary; group-starts are rows differing
        # from their predecessor in any key.
        order = np.lexsort(cols[::-1])
        same = np.ones(max(len(order) - 1, 0), dtype=bool)
        for c in cols:
            s = c[order]
            same &= s[1:] == s[:-1]
        gid = np.concatenate([[0], np.cumsum(~same)])
        _, counts = np.unique(gid, return_counts=True)
        dup_rows = int(counts[counts > 1].sum())
        pct = 100.0 * dup_rows / max(te.sum(), 1)
        out["duplicate_input_pct_test"] = pct
        check(f"{opt_type}: duplicate-input rate ~0% (v3 was 17.8/19.8%)",
              pct < 0.5, f"{pct:.4f}% ({dup_rows} rows)")
        if "am_settlement" in f:
            am = f["am_settlement"][:][te]
            out["am_settlement_counts"] = {int(k): int(v) for k, v in
                                           zip(*np.unique(am, return_counts=True))}

        # --- 9. VVIX / VXN / VXD cannot be present ---
        if "vix_level" in f:
            lv = f["vix_level"][:].astype(np.float64).ravel()
            lv = lv[np.isfinite(lv)]
            out["vix_level"] = {"min": float(lv.min()), "median": float(np.median(lv)),
                               "max": float(lv.max())}
            check(f"{opt_type}: vix_level in VIX range (not VVIX)",
                  lv.max() < VIX_LEVEL_MAX,
                  f"min {lv.min():.2f} / med {np.median(lv):.2f} / max {lv.max():.2f}")

        # --- 7. raw VIX -> HDF5 spot checks ---
        vix = (pl.scan_csv(SCRIPT_DIR / VIX_CSV)
               .select(pl.col("date").str.to_date("%Y-%m-%d"), pl.col("vix"))
               .filter(pl.col("vix") > 0).sort("date").collect())
        vmap = {d.isoformat(): v for d, v in zip(vix["date"].to_list(), vix["vix"].to_list())}
        if "vix_level" in f:
            uniq = np.unique(dates)
            probes = [uniq[0], uniq[len(uniq) // 2], uniq[-1]]
            diffs = []
            for p in probes:
                idx = int(np.flatnonzero(dates == p)[0])
                got = float(np.asarray(f["vix_level"][idx]).item())
                want = vmap.get(str(p))
                if want is not None:
                    diffs.append(abs(got - want))
                    print(f"      {p}: hdf5 {got:.4f} vs csv {want:.4f}")
            check(f"{opt_type}: vix_level matches raw CSV (early/mid/late)",
                  diffs and max(diffs) < 1e-3, f"max |diff| = {max(diffs):.2e}" if diffs else "no probes")

    # --- 1. parquet vs HDF5 ---
    if parts_dir and parts_dir.is_dir():
        pq = sorted(parts_dir.glob("*.parquet"))
        if pq:
            total = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in pq)
            out["parquet_rows_all_types"] = int(total)
            out["parquet_parts"] = len(pq)
            print(f"      parquet: {len(pq)} parts, {total:,d} rows (both types, pre-maturity-cut)")

    out["sha256"] = sha256(h5_path)
    print(f"      sha256: {out['sha256']}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Validate the v4 / v5 dataset")
    ap.add_argument("--smoke", action="store_true", help="Validate smoke files instead")
    ap.add_argument("--dataset-version", choices=["v4", "v5"], default="v4",
                    help="v4 = unfiltered sample, v5 = canonical filtered sample")
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args()
    ver = args.dataset_version

    if args.smoke:
        files = [(SCRIPT_DIR / f"smoke_{ver}/deeponet_tensors_{t}_{ver}_smoke.h5", t) for t in ("call", "put")]
        parts = SCRIPT_DIR / "smoke_v4/parts_diag"  # v5 smoke is re-exported from the same parts
        default_out = SCRIPT_DIR / f"smoke_{ver}/VALIDATION_{ver}_smoke.json"
    else:
        files = [(SCRIPT_DIR / f"deeponet_tensors_{t}_{ver}.h5", t) for t in ("call", "put")]
        parts = SCRIPT_DIR / "deeponet_training_data_parts_v4"
        default_out = SCRIPT_DIR / f"VALIDATION_{ver}.json"

    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        raise SystemExit(f"HDF5 not found (build not finished?): {missing}")

    print("=== raw VIX CSV ===")
    got = sha256(SCRIPT_DIR / VIX_CSV)
    check("VIX CSV sha256 matches reviewed file", got == EXPECTED_VIX_SHA, got)

    report = {"dataset_version": ver, "vix_csv_sha256": got, "files": []}
    for p, t in files:
        report["files"].append(validate_one(p, t, parts, ver))

    report["checks"] = results
    n_fail = sum(1 for r in results if not r["pass"])
    report["n_checks"] = len(results)
    report["n_failed"] = n_fail

    out_path = Path(args.out) if args.out else default_out
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed -> {out_path}")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
