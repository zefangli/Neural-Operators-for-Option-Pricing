"""
Sweep all `train_model_v3/*/results_*/metrics.json` (written by eval_to_json.py),
join with each run's `config.json`, and emit:
  - results/all_metrics.csv         (tidy, one row per run)
  - results/table_T1.md             (markdown main-results table for report.md)
  - results/table_T1.tex            (LaTeX booktabs version)

This makes the T1 table in report.md / PROGRESS reproducible instead of
hand-maintained, so it never drifts from the underlying metrics.json files.

Usage:  python analysis/aggregate_results.py
Run eval_to_json.py --all first so every run has a metrics.json.
"""

import csv
import json
from pathlib import Path

from _common import PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "results"
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


def collect():
    rows = []
    for mj in sorted((PROJECT_ROOT / "train_model_v3").glob("*/results_*/metrics.json")):
        m = json.loads(mj.read_text())
        m.setdefault("option_type", "?")
        m.setdefault("arch", "?")
        rows.append(m)
    rows.sort(key=lambda m: (m.get("option_type", ""),
                             BRANCH_ORDER.get(m.get("branch_key"), 9),
                             m.get("arch", "")))
    return rows


def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def write_csv(rows):
    keys = sorted({k for m in rows for k in m})
    with open(OUT_DIR / "all_metrics.csv", "w", newline="") as f:
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


def write_markdown(rows):
    header = [label for _, label in COLS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for m in rows:
        lines.append("| " + " | ".join(display_row(m)) + " |")
    (OUT_DIR / "table_T1.md").write_text("\n".join(lines) + "\n")


def write_latex(rows):
    header = [label.replace("%", r"\%") for _, label in COLS]
    lines = [r"\begin{tabular}{" + "l" * 3 + "r" * (len(COLS) - 3) + "}",
             r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for m in rows:
        lines.append(" & ".join(display_row(m)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT_DIR / "table_T1.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = collect()
    if not rows:
        raise SystemExit("No metrics.json found. Run: python analysis/eval_to_json.py --all")
    write_csv(rows)
    write_markdown(rows)
    write_latex(rows)
    print(f"Aggregated {len(rows)} runs -> results/all_metrics.csv, table_T1.md, table_T1.tex")
    print((OUT_DIR / "table_T1.md").read_text())


if __name__ == "__main__":
    main()
