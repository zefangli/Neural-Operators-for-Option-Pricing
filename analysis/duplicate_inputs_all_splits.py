#!/usr/bin/env python
"""
Duplicate-input rates over EVERY split, not just the test split.

wrds_data_2020-2025/VALIDATION_v5.json reports `duplicate_input_pct_test`, and
analysis/data_quality.py reports the same statistic -- both computed on
`split_id == 2` alone (validate_v4.py, section "8. contract identity /
duplicate inputs"). A reviewer is entitled to ask whether the train and
validation splits are equally clean, because a duplicated input in TRAIN is
what would let a model memorise a target it cannot otherwise fit. This script
answers that: train, validation, test and the full combined file, for both
option types, plus cross-split collisions.

    python analysis/duplicate_inputs_all_splits.py --dataset-version v5

Writes results/duplicate_inputs_{version}.{json,md}. CPU only, read-only, no
model, no checkpoint. The canonical HDF5 is re-hashed and checked against
DATASET_MANIFEST.json before anything is computed (--skip-hash exists for
development and stamps the artifact as unverified).

THE KEYS
--------
`input_key` = (date, log_moneyness, T_years, r, q)
    The full model input for one row: the date identifies the market state
    (branch_u / spot_history / vix_history are functions of the date alone) and
    the four trunk_y columns are the per-contract query. Two rows sharing this
    key are, to the model, THE SAME QUESTION -- if their targets differ the
    difference is unfittable by construction. This is the key validate_v4.py
    and analysis/data_quality.py group on, and the one the reported
    duplicate-input rate refers to.

`query_key` = (log_moneyness, T_years, r, q)
    The same key with the date dropped. It is reported as a DIAGNOSTIC ONLY.
    Two rows with the same query on different dates are NOT duplicate inputs
    for a market-state-conditioned model: the branch tensor differs, so the
    model sees different inputs and can legitimately predict different prices.
    A high `query_key` collision rate says the (log_m, T, r, q) grid repeats
    across dates -- which it must, since strikes and maturities are listed on a
    schedule -- and says nothing about leakage or unfittable targets.

PRECISION
---------
Grouping is on the EXACT float32 bits stored in the file. No rounding, no
tolerance: `trunk_y` is read as float32 and compared with `==`. validate_v4.py
upcasts the same columns to float64 first, which is exact and value-preserving
for float32 inputs, so the two agree bit for bit -- the test-split number here
must reproduce `duplicate_input_pct_test` in VALIDATION_v5.json exactly, and
the run aborts if it does not (--tol-pct).

CROSS-SPLIT
-----------
`input_key` collisions across splits are ZERO BY CONSTRUCTION: the splits are
cut by date (validate_v4.py checks no date appears in two splits), and the date
is in the key, so no key can span two splits. Reporting it anyway makes the
construction falsifiable rather than assumed.

`query_key` cross-split collisions would not be leakage either -- see above --
but they are worth measuring rather than guessing at, and the measurement is
not what a reader might assume. A listed strike does recur day after day, yet
the query does not: log-moneyness is log(S/K) and therefore carries the day's
spot level, and the settlement-aware maturity is a continuous quantity in
years, so the same contract on two dates lands on a different (log_m, T). The
numbers in the artifact are what settles this, not this paragraph.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_ROOT, add_dataset_arg, dataset_provenance, output_tag  # noqa: E402
from eval_to_json import sha256_of  # noqa: E402

SPLITS = (("train", 0), ("val", 1), ("test", 2))
KEY_NAMES = {"input_key": ["date", "log_moneyness", "T_years", "r", "q"],
             "query_key": ["log_moneyness", "T_years", "r", "q"]}


# ------------------------------------------------------------------ grouping

def group_ids(cols):
    """Group rows by exact equality on every column of `cols`.

    Returns (order, gid_sorted, sizes): `order` sorts the rows into key order,
    `gid_sorted[i]` is the group of the i-th row OF THE SORTED ORDER, and
    `sizes[g]` is that group's row count. Same lexsort/adjacent-difference
    construction validate_v4.py uses, so the group boundaries are identical.
    """
    n = len(cols[0])
    if n == 0:
        return np.empty(0, np.intp), np.empty(0, np.intp), np.empty(0, np.intp)
    order = np.lexsort([np.asarray(c) for c in cols][::-1])
    same = np.ones(n - 1, dtype=bool)
    for c in cols:
        s = np.asarray(c)[order]
        same &= s[1:] == s[:-1]
    starts = np.concatenate([[True], ~same])
    gid_sorted = np.cumsum(starts) - 1
    return order, gid_sorted, np.bincount(gid_sorted)


def duplicate_stats(cols):
    """Duplicate-group counts and the duplicate-row PERCENTAGE for one key."""
    n = len(cols[0])
    _order, _gid, sizes = group_ids(cols)
    dup = sizes > 1
    n_dup_rows = int(sizes[dup].sum())
    return {
        "n_rows": int(n),
        "n_groups": int(sizes.size),
        "n_dup_groups": int(dup.sum()),
        "n_rows_in_dup_groups": n_dup_rows,
        "pct_rows_in_dup_groups": 100.0 * n_dup_rows / n if n else 0.0,
        "max_group_size": int(sizes.max()) if sizes.size else 0,
        "group_size_histogram": {str(k): int(v) for k, v in
                                 enumerate(np.bincount(sizes)) if v},
    }


def cross_split_stats(cols, split):
    """Groups (on the FULL file) whose rows do not all live in one split.

    A group spanning two splits means the same key is present on both sides of
    a split boundary. Also returns the pairwise counts so "train<->test" can be
    distinguished from "train<->val".
    """
    n = len(cols[0])
    if n == 0:
        return {"n_rows": 0, "n_groups": 0, "n_groups_spanning_splits": 0,
                "n_rows_in_spanning_groups": 0, "pct_rows_in_spanning_groups": 0.0,
                "n_groups_by_split_pair": {"train_val": 0, "train_test": 0, "val_test": 0}}
    order, gid_sorted, sizes = group_ids(cols)
    split_sorted = np.asarray(split)[order]
    n_groups = int(sizes.size)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(gid_sorted) != 0) + 1])
    smin = np.minimum.reduceat(split_sorted, starts)
    smax = np.maximum.reduceat(split_sorted, starts)
    spanning = smin != smax
    rows_in_spanning = int(sizes[spanning].sum())
    pairs = {}
    for a, sa in SPLITS:
        for b, sb in SPLITS:
            if sa >= sb:
                continue
            has_a = np.zeros(n_groups, bool)
            has_b = np.zeros(n_groups, bool)
            has_a[gid_sorted[split_sorted == sa]] = True
            has_b[gid_sorted[split_sorted == sb]] = True
            pairs["%s_%s" % (a, b)] = int((has_a & has_b).sum())
    return {
        "n_rows": int(n),
        "n_groups": n_groups,
        "n_groups_spanning_splits": int(spanning.sum()),
        "n_rows_in_spanning_groups": rows_in_spanning,
        "pct_rows_in_spanning_groups": 100.0 * rows_in_spanning / n,
        "n_groups_by_split_pair": pairs,
    }


# ------------------------------------------------------------------- the run

def load_columns(h5_path):
    with h5py.File(h5_path, "r") as f:
        trunk = f["trunk_y"][:]                 # float32, exact bits, no cast
        date = f["date"][:]
        split = f["split_id"][:].astype(np.int64)
    if not np.isfinite(trunk).all():
        raise SystemExit("ABORT: %s has non-finite trunk_y; equality grouping is "
                         "undefined for NaN." % h5_path)
    return date, trunk, split


def key_columns(name, date, trunk):
    cols = [trunk[:, i] for i in range(4)]
    return [date] + cols if name == "input_key" else cols


def analyse(option_type, args, prov, expected_test_pct):
    date, trunk, split = load_columns(prov["h5_path"])
    res = {"option_type": option_type, "provenance": prov,
           "n_rows_total": int(len(split)),
           "rows_by_split": {nm: int((split == sid).sum()) for nm, sid in SPLITS},
           "keys": {}}
    for key_name in KEY_NAMES:
        cols = key_columns(key_name, date, trunk)
        per_split = {}
        for nm, sid in SPLITS:
            m = split == sid
            per_split[nm] = duplicate_stats([c[m] for c in cols])
            print("  [%s/%s] %-5s %10d rows  dup rows %.6f%%"
                  % (option_type, key_name, nm, per_split[nm]["n_rows"],
                     per_split[nm]["pct_rows_in_dup_groups"]))
        full = duplicate_stats(cols)
        print("  [%s/%s] %-5s %10d rows  dup rows %.6f%%"
              % (option_type, key_name, "FULL", full["n_rows"],
                 full["pct_rows_in_dup_groups"]))
        cross = cross_split_stats(cols, split)
        print("  [%s/%s] cross-split: %d groups span >1 split (%.6f%% of rows)"
              % (option_type, key_name, cross["n_groups_spanning_splits"],
                 cross["pct_rows_in_spanning_groups"]))
        res["keys"][key_name] = {"key": KEY_NAMES[key_name], "by_split": per_split,
                                 "full_dataset": full, "cross_split": cross}

    got = res["keys"]["input_key"]["by_split"]["test"]["pct_rows_in_dup_groups"]
    res["reproduces_validation"] = {
        "source": "wrds_data_2020-2025/VALIDATION_%s.json duplicate_input_pct_test"
                  % args.dataset_version,
        "expected_pct": expected_test_pct, "recomputed_pct": got,
        "abs_diff": None if expected_test_pct is None else abs(got - expected_test_pct),
        "tolerance_pct": args.tol_pct,
    }
    if expected_test_pct is None:
        print("  [%s] WARNING: no duplicate_input_pct_test found to reproduce" % option_type)
    elif abs(got - expected_test_pct) > args.tol_pct:
        raise SystemExit(
            "ABORT: %s test-split duplicate rate recomputed as %.8f%% but "
            "VALIDATION_%s.json records %.8f%%. This script must reproduce the "
            "published check before its train/val numbers may be cited."
            % (option_type, got, args.dataset_version, expected_test_pct))
    else:
        print("  [%s] test-split reproduces VALIDATION_%s.json (%.8f%% vs %.8f%%)"
              % (option_type, args.dataset_version, got, expected_test_pct))
    return res


def validation_test_pct(version, option_type):
    """`duplicate_input_pct_test` for one option type, or None if unavailable."""
    p = PROJECT_ROOT / "wrds_data_2020-2025" / ("VALIDATION_%s.json" % version)
    if not p.exists():
        return None
    for rec in json.loads(p.read_text()).get("files", []):
        if rec.get("option_type") == option_type and "duplicate_input_pct_test" in rec:
            return float(rec["duplicate_input_pct_test"])
    return None


# ------------------------------------------------------------------ markdown

def markdown(results, meta):
    out = ["# Duplicate-input rates over all splits",
           "",
           "Generated %s. CPU only, read-only, no model. Dataset %s, script "
           "`%s`, git HEAD `%s`%s."
           % (meta["generated"], meta["dataset_version"], meta["script"],
              meta["git_head"][:12],
              " (DIRTY WORKTREE)" if meta["git_worktree_dirty"] else ""),
           "",
           "The published duplicate-input check (`validate_v4.py` section 8, "
           "`analysis/data_quality.py`) is computed on the **test split only**. "
           "This table extends it to train, validation and the full file, and adds "
           "cross-split collisions.",
           "",
           "## Keys", "",
           "| key | columns | meaning |", "|---|---|---|",
           "| `input_key` | %s | the full model input for a row. Two rows sharing it "
           "are the same question asked twice; a target disagreement inside such a group "
           "is unfittable by construction. This is the published statistic's key. |"
           % ", ".join("`%s`" % c for c in KEY_NAMES["input_key"]),
           "| `query_key` | %s | `input_key` with the date dropped. **Diagnostic only.** "
           "The model is conditioned on the market state, which is a function of the "
           "date, so the same query on two dates is not a duplicate input -- the branch "
           "tensor differs and different prices are correct. A high collision rate here "
           "reflects the listed strike/maturity schedule repeating, not leakage. |"
           % ", ".join("`%s`" % c for c in KEY_NAMES["query_key"]),
           "",
           "Grouping is on the **exact float32 bits** stored in the HDF5 -- no rounding "
           "and no tolerance. `validate_v4.py` upcasts to float64 first, which is exact "
           "for float32 input, so the groups are identical.",
           ""]
    for key_name in KEY_NAMES:
        out += ["## `%s`" % key_name, "",
                "| option | split | rows | groups | dup groups | rows in dup groups | "
                "% rows in dup groups | max group |",
                "|---|---|---|---|---|---|---|---|"]
        for res in results:
            k = res["keys"][key_name]
            for label in ("train", "val", "test", "FULL"):
                s = k["full_dataset"] if label == "FULL" else k["by_split"][label]
                out.append("| %s | %s | %s | %s | %s | %s | %.6f | %d |"
                           % (res["option_type"], label, f"{s['n_rows']:,}",
                              f"{s['n_groups']:,}", f"{s['n_dup_groups']:,}",
                              f"{s['n_rows_in_dup_groups']:,}",
                              s["pct_rows_in_dup_groups"], s["max_group_size"]))
        out += ["", "Cross-split (groups formed on the full file whose rows do not all "
                "lie in one split):", "",
                "| option | groups spanning >1 split | rows in them | % of rows | "
                "train&val | train&test | val&test |",
                "|---|---|---|---|---|---|---|"]
        for res in results:
            c = res["keys"][key_name]["cross_split"]
            p = c["n_groups_by_split_pair"]
            out.append("| %s | %s | %s | %.6f | %s | %s | %s |"
                       % (res["option_type"], f"{c['n_groups_spanning_splits']:,}",
                          f"{c['n_rows_in_spanning_groups']:,}",
                          c["pct_rows_in_spanning_groups"],
                          f"{p['train_val']:,}", f"{p['train_test']:,}",
                          f"{p['val_test']:,}"))
        out.append("")
        if key_name == "input_key":
            out += ["`input_key` cross-split collisions are zero **by construction**: the "
                    "splits are cut by date and the date is part of the key. The row is "
                    "reported so the construction is checked rather than assumed.", ""]
        else:
            spans = sum(r["keys"]["query_key"]["cross_split"]["n_groups_spanning_splits"]
                        for r in results)
            out += ["A `query_key` collision carries no information about leakage even "
                    "when it happens: the branch tensor differs, so the model sees "
                    "different inputs and different prices are correct. "
                    + ("In this sample there are none at all -- %d groups span more than "
                       "one split. A listed strike does recur day after day, but the query "
                       "does not: log-moneyness is log(S/K) and so carries the day's spot "
                       "level, and the settlement-aware maturity is continuous in years, so "
                       "the same contract on two dates lands on a different `(log_m, T)`."
                       % spans if spans == 0 else
                       "Here %d groups span more than one split; see the counts above."
                       % spans), ""]
    out += ["## Reproduction check", ""]
    for res in results:
        v = res["reproduces_validation"]
        out.append("- **%s**: test-split `input_key` rate %.8f%% vs %s = %s (tolerance %g)."
                   % (res["option_type"], v["recomputed_pct"], v["source"],
                      "unavailable" if v["expected_pct"] is None
                      else "%.8f%%" % v["expected_pct"], v["tolerance_pct"]))
    out.append("")
    return "\n".join(out)


def git_head():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        return h, bool(d)
    except Exception as exc:
        return "unavailable (%s)" % exc, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--option-type", choices=["call", "put", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--h5-dir", default=None)
    ap.add_argument("--tol-pct", type=float, default=1e-9,
                    help="max allowed |recomputed - VALIDATION| test-split percentage")
    ap.add_argument("--skip-hash", action="store_true",
                    help="do not re-hash the HDF5 (stamps the artifact 'not verified'; "
                         "never use for a publishable run)")
    add_dataset_arg(ap)
    args = ap.parse_args()

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
            print("[%s] sha256 %s... (%.0fs)" % (ot, prov["observed_sha256"][:12],
                                                 time.time() - t0))
            if prov["observed_sha256"] != prov.get("expected_sha256"):
                raise SystemExit(
                    "ABORT: %s hashes to %s but the manifest declares %s -- this is not "
                    "the canonical %s sample." % (prov["h5_path"], prov["observed_sha256"],
                                                  prov.get("expected_sha256"),
                                                  args.dataset_version))
            prov["sha256_verified"] = True
        results.append(analyse(ot, args, prov,
                               validation_test_pct(args.dataset_version, ot)))

    meta = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "analysis/duplicate_inputs_all_splits.py",
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "git_head": head,
        "git_worktree_dirty": dirty,
        "dataset_version": args.dataset_version,
        "device": "cpu",
        "keys": KEY_NAMES,
        "float_precision": ("exact float32 bits as stored in trunk_y; no rounding, no "
                            "tolerance. validate_v4.py upcasts the same columns to "
                            "float64, which is exact for float32 input, so the grouping "
                            "is bit-identical."),
        "extends": ("wrds_data_2020-2025/validate_v4.py section 8 and "
                    "analysis/data_quality.py duplicate_report, both of which compute "
                    "the rate on split_id == 2 only"),
        "query_key_caveat": ("a query_key collision across dates is NOT an input "
                             "duplicate for a market-state-conditioned model: the branch "
                             "tensor differs, so the model sees different inputs. It is "
                             "reported as a diagnostic only."),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / ("duplicate_inputs" + output_tag(args.dataset_version))
    stem.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(results, meta), encoding="utf-8")
    print("wrote %s.{json,md}" % stem)


if __name__ == "__main__":
    main()
