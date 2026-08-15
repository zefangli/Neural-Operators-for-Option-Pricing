"""
Sweep all `train_model_v3/*/results_*/metrics.json` (written by eval_to_json.py)
for ONE dataset version and emit:
  - results/all_metrics_<ver>.csv   (tidy, one row per canonical run)
  - results/table_T1_<ver>.md       (markdown main-results table for report.md)
  - results/table_T1_<ver>.tex      (LaTeX booktabs version)
  - results/replication_seeds_<ver>.{csv,md}   (only if non-seed-42 runs exist)

This makes the T1 table in report.md / PROGRESS reproducible instead of
hand-maintained, so it never drifts from the underlying metrics.json files.

The dataset version is REQUIRED and there is no default: a silent default is how
a v4 number reaches a v5 table. Canonical T1 must be exactly the 12 seed-42
cells (2 option types x 3 branches x 2 architectures) of one dataset version --
missing, duplicated, off-seed, off-version, VVIX or smoke rows are all refused
by name.

`dataset_version` (the SAMPLE) and `schema_version` (the tensor layout) are
independent: v5 is the quote-filtered sample and still carries schema "v4".
Which version is canonical is read from wrds_data_2020-2025/DATASET_MANIFEST.json,
never hard-coded here.

Usage:  python analysis/aggregate_results.py --dataset-version v5
Run `eval_to_json.py --all --dataset-version v5` first so every run has a metrics.json.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from _common import PROJECT_ROOT, canonical_version

OUT_DIR = PROJECT_ROOT / "results"
# VVIX (not VIX) trained runs were quarantined 2026-08-11: the vix_history branch was fed
# CBOE VVIX (secid 152892), not VIX. Their weights are preserved under
# train_model_v3/_vix_is_vvix_LEGACY/ but must never be aggregated as VIX results.
QUARANTINE_DIR = PROJECT_ROOT / "train_model_v3" / "_vix_is_vvix_LEGACY"

COLS = [
    ("option_type", "option"),
    ("branch_key", "branch"),
    ("arch", "arch"),
    ("r2_price", "R2(price)"),
    ("r2_log_filtered", "R2(log,T>1d)"),
    ("r2_log_full", "R2(log,full)"),
    ("rmse_log", "RMSE(log)"),
    ("rmse_price", "RMSE(price)"),
    ("mae_price", "MAE(price)"),
]
BRANCH_ORDER = {"branch_u": 0, "spot_history": 1, "vix_history": 2}
BRANCH_LABEL = {"branch_u": "vol_surface", "spot_history": "spot_history", "vix_history": "vix_history"}

CANONICAL_SEED = 42
# The 12 cells a main-results table must contain, no more and no fewer.
EXPECTED_CELLS = {(o, b, a) for o in ("call", "put") for b in BRANCH_ORDER for a in ("A", "B")}
REPL_METRICS = ["r2_price", "r2_log_filtered", "r2_log_full", "rmse_log"]


def is_quarantined(path):
    """True only for runs physically under _vix_is_vvix_LEGACY/.

    Deliberately NOT a `results_vix_history*` name prefix match: that also caught
    the corrected `results_vix_history_v4/` dirs, silently dropping the retrained
    VIX runs from every aggregated table. Quarantine is a location, not a name --
    the four VVIX runs were moved into _vix_is_vvix_LEGACY/, so location is exact.
    """
    return "_vix_is_vvix_LEGACY" in path.parts


def is_smoke(path):
    """True for runs written by a 1-batch smoke run (`results_*_smoke/`).

    Same lesson as is_quarantined: be precise. Matches only a `results_`-prefixed
    directory part whose name ENDS with `_smoke`, so `results_vol_surface_v4_smoke`
    is caught while `results_vol_surface_v4`, `..._v4_seed43` and the file
    `metrics_smoke.json` are not. Smoke weights are meaningless (1 batch, 1 epoch)
    and must never reach a manuscript table.
    """
    return any(p.startswith("results_") and p.endswith("_smoke") for p in path.parts)


def warn_smoke(paths):
    """Visible warning (not silent): smoke runs exist and are excluded on purpose."""
    if not paths:
        return
    print("WARNING: smoke run(s) detected (results dir ending in _smoke) —")
    print("  these are 1-batch sanity runs, NOT trained results, and are excluded.")
    for p in sorted(paths):
        print(f"  - {p.relative_to(PROJECT_ROOT)}")


def warn_quarantine():
    """Visible warning (not silent): VVIX artifacts exist and are excluded on purpose."""
    if not QUARANTINE_DIR.is_dir():
        return
    excluded = sorted(p.name for p in QUARANTINE_DIR.iterdir() if p.is_dir())
    print("WARNING: VVIX-trained run(s) detected in train_model_v3/_vix_is_vvix_LEGACY/ —")
    print("  these are EXCLUDED VVIX artifacts (input was CBOE VVIX, secid 152892, not VIX)")
    print("  and are intentionally not aggregated. Corrected canonical CBOE-VIX v5")
    print("  vix_history runs are complete and ARE included in this table.")
    for name in excluded:
        print(f"  - {name}")


def cell(m):
    """(option_type, branch_key, arch) -- the identity of a T1 table cell."""
    return (m.get("option_type", "?"), m.get("branch_key", "?"), m.get("arch", "?"))


def cell_name(c):
    return f"{c[0]}/{BRANCH_LABEL.get(c[1], c[1])}/{c[2]}"


def warn_dropped(dropped):
    """Visible warning (not silent): rows that exist but do not belong in THIS table."""
    if not dropped:
        return
    print(f"WARNING: {len(dropped)} metrics.json excluded from this dataset version —")
    for path, why in sorted(dropped):
        print(f"  - {path}  ({why})")


def collect(version, dropped=None):
    """Rows for `version` only. Everything else is dropped LOUDLY, never silently.

    Pass a list as `dropped` to also get the exclusions back (they are named in
    the validation error, so "MISSING call/vol_surface/A" is accompanied by the
    reason it went missing -- e.g. "dataset_version=v4, want v5").
    """
    rows, skipped_smoke = [], []
    dropped = [] if dropped is None else dropped
    for mj in sorted((PROJECT_ROOT / "train_model_v3").glob("*/results_*/metrics.json")):
        if is_quarantined(mj):
            continue
        if is_smoke(mj):
            skipped_smoke.append(mj.parent)
            continue
        m = json.loads(mj.read_text())
        m.setdefault("option_type", "?")
        m.setdefault("arch", "?")
        rel = str(mj.relative_to(PROJECT_ROOT))
        m["metrics_path"] = rel
        got = m.get("dataset_version")
        if not got:
            # Written by a pre-provenance eval_to_json.py -- unknowable sample.
            dropped.append((rel, "no dataset_version recorded; re-run eval_to_json.py"))
        elif got != version:
            dropped.append((rel, f"dataset_version={got}, want {version}"))
        elif m.get("canonical_dataset") is False:
            dropped.append((rel, f"non-canonical {got} HDF5 (sha256 mismatch, e.g. smoke build)"))
        elif m.get("seed") is None:
            dropped.append((rel, "no seed recorded; re-run eval_to_json.py"))
        else:
            rows.append(m)
    warn_smoke(skipped_smoke)
    warn_dropped(dropped)
    rows.sort(key=lambda m: (m.get("option_type", ""),
                             BRANCH_ORDER.get(m.get("branch_key"), 9),
                             m.get("arch", ""), m.get("seed", 0)))
    return rows


def validate_canonical(rows, version, dropped=()):
    """Refuse anything that is not exactly the 12 seed-42 cells of one sample.

    Errors NAME the offending cells/files: "validation failed" at 2am is useless.
    """
    errs = []
    seen = {}
    for m in rows:
        seen.setdefault(cell(m), []).append(m["metrics_path"])

    missing = EXPECTED_CELLS - set(seen)
    if missing:
        errs.append("MISSING cell(s) — no seed-%d %s run has a metrics.json:\n%s"
                    % (CANONICAL_SEED, version,
                       "\n".join(f"    - {cell_name(c)}" for c in sorted(missing))))
    extra = set(seen) - EXPECTED_CELLS
    if extra:
        errs.append("UNEXPECTED cell(s) (not one of the 12):\n%s" % "\n".join(
            f"    - {cell_name(c)}: " + ", ".join(seen[c]) for c in sorted(extra)))
    dup = {c: p for c, p in seen.items() if len(p) > 1}
    if dup:
        errs.append("DUPLICATE cell(s) — one cell, several metrics.json:\n%s" % "\n".join(
            f"    - {cell_name(c)}: " + ", ".join(sorted(p)) for c, p in sorted(dup.items())))

    schemas = {m.get("schema_version") for m in rows}
    if len(schemas) > 1:
        errs.append(f"MIXED schema_version across rows: {sorted(map(str, schemas))}")
    for opt in ("call", "put"):
        hashes = {m.get("h5_sha256") for m in rows if m.get("option_type") == opt}
        if len(hashes) > 1:
            errs.append(f"MIXED {opt} HDF5 sha256 across rows: "
                        + ", ".join(sorted(str(h)[:12] + "..." for h in hashes)))
    if errs:
        if dropped:
            errs.append("EXCLUDED row(s) — why the table is short:\n%s" % "\n".join(
                f"    - {p}  ({why})" for p, why in sorted(dropped)))
        raise SystemExit("REFUSING TO WRITE %s tables:\n  %s" % (version, "\n  ".join(errs)))


def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def write_csv(rows, path):
    keys = sorted({k for m in rows for k in m})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def display_row(m):
    out = []
    for key, _ in COLS:
        v = m.get(key)
        if key == "branch_key":
            out.append(BRANCH_LABEL.get(v, str(v)))
        elif isinstance(v, float):
            out.append(fmt(v))
        else:
            out.append(str(v))
    return out


def write_markdown(rows, path):
    header = [label for _, label in COLS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for m in rows:
        lines.append("| " + " | ".join(display_row(m)) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_replication(all_rows, version):
    """Seed replication summary: mean +/- sample stdev, vol_surface cells only.

    Kept in its own file on purpose -- averaging replication seeds into T1 would
    silently change what the headline number means.

    Restricted to branch_key=='branch_u' (vol_surface): those are the only cells
    with a deliberate multi-seed replication plan (seeds 42/43/44). Every
    spot_history/vix_history cell still has exactly one seed (42) -- including
    them here would report n_seeds=1 rows, and a naive std=0.0 for those would
    misrepresent an UNMEASURED dispersion as a MEASURED zero. Dropped entirely
    rather than shown with a fake number.

    Defense in depth: even within vol_surface, any cell that somehow ends up
    with fewer than 2 seeds gets std=None ("N/A" in the table), never 0.0.
    """
    by_cell = {}
    for m in all_rows:
        c = cell(m)
        if c[1] != "branch_u":   # vol_surface only -- see docstring
            continue
        by_cell.setdefault(c, []).append(m)
    out = []
    for c in sorted(by_cell, key=lambda c: (c[0], BRANCH_ORDER.get(c[1], 9), c[2])):
        ms = sorted(by_cell[c], key=lambda m: m["seed"])
        row = {"option_type": c[0], "branch_key": c[1], "arch": c[2],
               "n_seeds": len(ms), "seeds": ",".join(str(m["seed"]) for m in ms)}
        for k in REPL_METRICS:
            vals = [m[k] for m in ms if isinstance(m.get(k), float)]
            row[k + "_mean"] = statistics.mean(vals) if vals else float("nan")
            row[k + "_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        out.append(row)

    write_csv(out, OUT_DIR / f"replication_seeds_{version}.csv")
    header = ["option", "branch", "arch", "seeds"] + [k + " mean+/-sd" for k in REPL_METRICS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in out:
        cells = [r["option_type"], BRANCH_LABEL.get(r["branch_key"], r["branch_key"]),
                 r["arch"], r["seeds"]]
        for k in REPL_METRICS:
            sd = r[k + "_std"]
            sd_str = "N/A" if sd is None else f"{sd:.6f}"
            cells.append(f"{r[k + '_mean']:.6f} +/- {sd_str}")
        lines.append("| " + " | ".join(cells) + " |")
    (OUT_DIR / f"replication_seeds_{version}.md").write_text("\n".join(lines) + "\n")
    return out


def write_latex(rows, path):
    header = [label.replace("%", r"\%") for _, label in COLS]
    lines = [r"\begin{tabular}{" + "l" * 3 + "r" * (len(COLS) - 3) + "}",
             r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for m in rows:
        lines.append(" & ".join(display_row(m)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-run metrics.json into T1")
    ap.add_argument("--dataset-version", required=True, choices=("v3", "v4", "v5"),
                    help="REQUIRED, no default: which SAMPLE's runs to aggregate. "
                         "Read from each metrics.json's recorded dataset_version "
                         "(not its schema_version, and not the dir name).")
    ap.add_argument("--allow-partial", action="store_true",
                    help="skip the exactly-12-seed-42-cells check (exploratory / "
                         "historical v3 sets only — NEVER for a manuscript table)")
    args = ap.parse_args()
    version = args.dataset_version

    OUT_DIR.mkdir(exist_ok=True)
    warn_quarantine()
    canon_ver = canonical_version()
    if version != canon_ver:
        print(f"NOTE: dataset_version={version} is NOT the canonical sample "
              f"({canon_ver} is, per DATASET_MANIFEST.json). These tables are a "
              f"robustness/historical set, not the manuscript's main results.")
    dropped = []
    rows = collect(version, dropped)
    if not rows:
        raise SystemExit(f"No {version} metrics.json found. Run: "
                         f"python analysis/eval_to_json.py --all --dataset-version {version}"
                         + ("".join(f"\n  excluded: {p} ({w})" for p, w in sorted(dropped))))

    canonical = [m for m in rows if m["seed"] == CANONICAL_SEED]
    extra_seeds = [m for m in rows if m["seed"] != CANONICAL_SEED]
    if not args.allow_partial:
        validate_canonical(canonical, version, dropped)

    write_csv(canonical, OUT_DIR / f"all_metrics_{version}.csv")
    write_markdown(canonical, OUT_DIR / f"table_T1_{version}.md")
    write_latex(canonical, OUT_DIR / f"table_T1_{version}.tex")
    print(f"Aggregated {len(canonical)} seed-{CANONICAL_SEED} {version} runs -> "
          f"results/all_metrics_{version}.csv, table_T1_{version}.md, table_T1_{version}.tex")
    print((OUT_DIR / f"table_T1_{version}.md").read_text())

    if extra_seeds:
        n = len(write_replication(rows, version))
        print(f"{len(extra_seeds)} replication run(s) (seed != {CANONICAL_SEED}) kept OUT of T1 "
              f"-> results/replication_seeds_{version}.{{csv,md}} ({n} cells)")


if __name__ == "__main__":
    main()
