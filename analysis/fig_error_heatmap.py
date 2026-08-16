"""
F3 -- error heatmap: where the model's error lives on the (moneyness x maturity) grid.

Two panels per option type, both over the SAME fixed grid:

  * RMSE of log(V/K) -- the repo's headline error scale (CLAUDE.md: the paper
    reports R2(price), R2(log, full) and R2(log, T>1day) together, with the log
    metric as headline).
  * RMSE of V/K itself -- the same rows on the price scale, for the reader who
    wants to know that a large log error in the deep wings is a tiny price error.

BINNING RULE (fixed in advance, no data-dependent edges, no cherry-picking) --
identical for calls and puts so the two figures are directly comparable:

  * MATURITY (calendar days, half-open (lo, hi]) -- F2's three bands, plus a
    short head and a long tail so the grid covers the whole v5 test range
    (1.73 .. 2158.7 days):
        (1, 7]  (7, 30]  (30, 90]  (90, 365]  (365, inf)
  * LOG-MONEYNESS log(S/K) -- nine fixed, symmetric-about-zero edges:
        -inf, -0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20, +inf
    Chosen from the option-pricing convention (a tight ATM band, then widening
    wings), NOT from the error distribution and NOT from quantiles of either
    sample -- quantile edges would put a different grid under calls and puts.

NEAR-EXPIRY TAIL (CLAUDE.md "two recurring numerical facts", fact 1): the exact
`T = 0` decoder degeneracy that motivates the headline `R2(log, T > 1 day)`
filter is a **v3** artifact. v5 is built with `--min-maturity-days 1.0` on the
settlement-aware maturity, so its observed minimum is 1.729 days and the test
split contains ZERO rows with T <= 1 day -- the headline filter is a no-op here.
The script ASSERTS that (`assert (T_days <= 1).sum() == 0`) rather than trusting
the claim, and reports the count in the caption. Every test row is therefore
plotted; nothing is silently dropped and no cell is misleading.

Cells are computed over the FULL v5 test split (819,342 call / 1,380,379 put
rows) -- no subsampling. Empty cells are drawn blank and reported as n = 0.

Provenance: the run must declare `dataset_version == "v5"`, `model_variant ==
"per_query"` and carry the manifest's v5 sha256 for its option type; the shared
`eval_to_json.run_provenance` gate then re-checks the config against the HDF5
actually read.

Usage (conda env `dl_new`):
    python analysis/fig_error_heatmap.py
Outputs:
    results/fig_F3_error_heatmap_{call,put}_v5.png
    results/fig_F3_error_heatmap_v5.md      (caption + binning rule + per-cell stats)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from _common import (PROJECT_ROOT, canonical_version, load_run, load_test_split,  # noqa: E402
                     manifest_entry, model_predict_logv, resolve_h5_path)
from eval_to_json import run_provenance  # noqa: E402

PER_QUERY_DIR = "train_model_v3/%s/results_vol_surface_v5"

# Half-open (lo, hi] in calendar days. F2's three bands + head + tail.
T_EDGES = [1.0, 7.0, 30.0, 90.0, 365.0, np.inf]
T_LABELS = ["1-7d", "7-30d", "30-90d", "90-365d", ">365d"]
# Half-open (lo, hi] in log(S/K). Symmetric, fixed, sample-independent.
M_EDGES = [-np.inf, -0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.20, np.inf]
M_LABELS = ["≤-.20", "-.20/-.10", "-.10/-.05", "-.05/-.02", "-.02/.02",
            ".02/.05", ".05/.10", ".10/.20", ">.20"]


# ------------------------------------------------------------------- provenance

def load_per_query(option_type, device="cpu"):
    """(config, model, h5_path) for the canonical per-query run, gate applied.

    Shared with F5 (`fig_pred_vs_market.py`) -- both figures must read exactly
    the same checkpoint under exactly the same provenance checks.
    """
    run_dir = PROJECT_ROOT / (PER_QUERY_DIR % option_type)
    cfg, _arch, model = load_run(run_dir, device=device)
    want = canonical_version()
    want_sha = manifest_entry(want)["files"][option_type]["sha256"]
    if cfg.get("model_variant") != "per_query":
        raise SystemExit(f"{run_dir}: model_variant={cfg.get('model_variant')!r}, want 'per_query'")
    if cfg.get("option_type") != option_type:
        raise SystemExit(f"{run_dir}: option_type={cfg.get('option_type')!r}")
    if cfg.get("dataset_version") != want:
        raise SystemExit(f"{run_dir}: dataset_version={cfg.get('dataset_version')!r}, "
                         f"want the canonical {want!r}")
    if cfg.get("h5_sha256") != want_sha:
        raise SystemExit(f"{run_dir}: h5_sha256={str(cfg.get('h5_sha256'))[:12]}... is not the "
                         f"manifest's {want} {option_type} file")
    h5_path = resolve_h5_path(cfg)
    run_provenance(cfg, h5_path)
    return cfg, model, h5_path


# ----------------------------------------------------------------------- stats

def cell_stats(log_pred, log_target, T_days, log_m):
    """-> (rmse_log, rmse_price, counts) arrays of shape (len(T_LABELS), len(M_LABELS))."""
    # np.digitize with right=True gives half-open (lo, hi]; edges exclude the outer
    # +-inf sentinels, so index 0 is "<= first finite edge".
    ti = np.digitize(T_days, T_EDGES[1:-1], right=True)
    mi = np.digitize(log_m, M_EDGES[1:-1], right=True)
    shape = (len(T_LABELS), len(M_LABELS))
    resid2 = (log_pred - log_target) ** 2
    perr2 = (np.exp(log_pred) - np.exp(log_target)) ** 2
    flat = ti * len(M_LABELS) + mi
    n = np.bincount(flat, minlength=shape[0] * shape[1]).astype(np.float64)
    s_log = np.bincount(flat, weights=resid2, minlength=n.size)
    s_price = np.bincount(flat, weights=perr2, minlength=n.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse_log = np.where(n > 0, np.sqrt(s_log / n), np.nan)
        rmse_price = np.where(n > 0, np.sqrt(s_price / n), np.nan)
    return (rmse_log.reshape(shape), rmse_price.reshape(shape),
            n.reshape(shape).astype(np.int64))


# ------------------------------------------------------------------------ plot

def _panel(ax, grid, counts, title, cbar_label, fmt):
    finite = grid[np.isfinite(grid) & (grid > 0)]
    norm = LogNorm(vmin=finite.min(), vmax=finite.max()) if finite.size else None
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", norm=norm)
    ax.set_xticks(range(len(M_LABELS)), M_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(T_LABELS)), T_LABELS, fontsize=8)
    ax.set_xlabel("log-moneyness  $\\log(S/K)$", fontsize=9)
    ax.set_ylabel("maturity (calendar days)", fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not counts[i, j]:
                ax.text(j, i, "n=0", ha="center", va="center", fontsize=6, color="0.4")
                continue
            # White on the dark end of viridis, black on the light end.
            rel = (np.log(grid[i, j]) - np.log(norm.vmin)) / (np.log(norm.vmax) - np.log(norm.vmin))
            ax.text(j, i, format(grid[i, j], fmt), ha="center", va="center", fontsize=6,
                    color="white" if rel < 0.6 else "black")
    plt.colorbar(im, ax=ax, label=cbar_label)


def make_figure(option_type, out_png, max_contracts=None, device="cpu"):
    """Build one option type's F3 figure. Returns a stats dict."""
    cfg, model, h5_path = load_per_query(option_type, device=device)
    data = load_test_split(h5_path, cfg["branch_key"], max_contracts=max_contracts)
    data.pop("branch_u", None)          # duplicate of `branch` here; ~0.6-1 GB saved
    log_pred, _sigma = model_predict_logv(model, data, device=device)

    log_target = np.asarray(data["target_v_log"], dtype=np.float64).ravel()
    T_days = np.asarray(data["T"], dtype=np.float64).ravel() * 365.0
    log_m = np.asarray(data["log_m"], dtype=np.float64).ravel()

    # The headline R2(log, T>1day) filter is a NO-OP on v5 by construction; assert
    # it rather than assume it, so a future sample that reintroduces expiry-day
    # rows fails here instead of quietly producing a misleading bottom row.
    n_le_1day = int((T_days <= 1.0).sum())
    assert n_le_1day == 0, (
        f"{option_type}: {n_le_1day} test rows have T <= 1 day. v5 is built with "
        "--min-maturity-days 1.0; the bottom maturity bucket would be the documented "
        "T->0 decoder degeneracy, not model error. Exclude them before plotting.")

    rmse_log, rmse_price, counts = cell_stats(log_pred, log_target, T_days, log_m)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    _panel(axes[0], rmse_log, counts, "RMSE of $\\log(V/K)$  (headline scale)",
           "RMSE $\\log(V/K)$", ".2f")
    _panel(axes[1], rmse_price, counts, "RMSE of $V/K$  (price scale)",
           "RMSE $V/K$", ".1e")
    fig.suptitle(f"F3 — error heatmap, {option_type}s (v5 test split, n={counts.sum():,}): "
                 f"per-query volatility model", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {"option_type": option_type, "n": int(counts.sum()), "n_T_le_1day": n_le_1day,
            "rmse_log": rmse_log, "rmse_price": rmse_price, "counts": counts,
            "T_days_range": [float(T_days.min()), float(T_days.max())],
            "log_m_range": [float(log_m.min()), float(log_m.max())],
            "rmse_log_overall": float(np.sqrt(np.mean((log_pred - log_target) ** 2))),
            "rmse_price_overall": float(np.sqrt(np.mean(
                (np.exp(log_pred) - np.exp(log_target)) ** 2)))}


CAPTION = """# F3 — Error heatmap, moneyness × maturity (v5 test split)

![call](fig_F3_error_heatmap_call_v5.png)
![put](fig_F3_error_heatmap_put_v5.png)

**Files:** `results/fig_F3_error_heatmap_call_v5.png`, `results/fig_F3_error_heatmap_put_v5.png`
(generated by `analysis/fig_error_heatmap.py`; canonical sample **v5**, run
`train_model_v3/{call,put}/results_vol_surface_v5` with `model_variant="per_query"`).

## Binning rule (fixed in advance, no cherry-picking)

Both option types use the **same** grid, so the two figures are directly comparable. Neither axis
is quantile- or error-derived — data-dependent edges would put a different grid under calls and
puts and would let the picture be tuned.

* **Maturity (rows)** — calendar days, half-open `(lo, hi]`:
  **(1, 7]**, **(7, 30]**, **(30, 90]**, **(90, 365]**, **(365, ∞)**. The middle three are F2's
  bands verbatim; the head and tail were added because the v5 test split runs from
  **1.73 to 2158.7 days**, which F2's `(7, 365]` span does not cover.
* **Log-moneyness (columns)** — `log(S/K)`, nine fixed symmetric bands with edges
  **−0.20, −0.10, −0.05, −0.02, +0.02, +0.05, +0.10, +0.20** (outer bands unbounded). A tight ATM
  band widening into the wings, per the usual pricing convention.
* **Rows used** — the **entire** v5 test split, no subsampling. An empty cell is drawn blank and
  labelled `n=0`, never merged into a neighbour.

## Statistics shown

| panel | statistic |
|---|---|
| left | `RMSE(log(V/K)) = sqrt(mean((log V̂/K − log V/K)²))` — the repo's headline error scale |
| right | `RMSE(V/K) = sqrt(mean((V̂/K − V/K)²))` — the same rows on the price scale |

Both colour scales are logarithmic (`LogNorm`), because cell RMSE spans orders of magnitude; the
numeric value is printed in each cell so the figure does not rely on colour alone.

## The near-expiry caveat, and why no bucket is excluded here

CLAUDE.md's fact 1 documents a decoder degeneracy at **exact `T = 0`**: v3's preprocessing set
`T = calendar_days/365`, so expiry-day quotes got literally `T = 0`, where the Black–Scholes
decoder returns intrinsic value for *any* σ̂ — structurally unfittable rows, which is why the
headline metric is `R²(log, T > 1 day)`.

That does **not** apply to v5. v5 is built with `--min-maturity-days 1.0` on the settlement-aware
maturity, so its observed minimum is **1.729 days** and the test split contains **zero** rows with
`T ≤ 1 day`. The `T > 1 day` filter is therefore a no-op on this sample (which is why the headline
and full-domain `R²(log)` coincide on v4/v5), and every test row is plotted. The script does not
take this on trust: it asserts `(T_days <= 1).sum() == 0` and hard-fails if a future sample
reintroduces expiry-day rows, rather than quietly rendering a misleading bottom row.

The `(1, 7]` row is genuinely short-dated, not degenerate — but note CLAUDE.md's fact 1(b): for
small positive `T`, `∂log(price)/∂σ` really does blow up near ATM, so that row's elevated log RMSE
is partly the ill-conditioning of the inverse problem rather than a model defect. The price-scale
panel is the check on that reading.

## Per-cell statistics

"""


def write_caption(out_md, stats_by_type):
    lines = [CAPTION]
    for s in stats_by_type.values():
        lines += [
            f"### {s['option_type']}s — n = {s['n']:,}, T ∈ [{s['T_days_range'][0]:.2f}, "
            f"{s['T_days_range'][1]:.2f}] days, log(S/K) ∈ [{s['log_m_range'][0]:.3f}, "
            f"{s['log_m_range'][1]:.3f}], rows with T ≤ 1 day: **{s['n_T_le_1day']}**.",
            f"Overall RMSE(log) = {s['rmse_log_overall']:.4f}, "
            f"RMSE(price) = {s['rmse_price_overall']:.3e}.",
            "",
            "| maturity | " + " | ".join(M_LABELS) + " |",
            "|---" * (len(M_LABELS) + 1) + "|",
        ]
        for i, tl in enumerate(T_LABELS):
            cells = []
            for j in range(len(M_LABELS)):
                if not s["counts"][i, j]:
                    cells.append("—")
                    continue
                cells.append(f"{s['rmse_log'][i, j]:.3f}<br>{s['rmse_price'][i, j]:.1e}"
                             f"<br>n={s['counts'][i, j]:,}")
            lines.append(f"| **{tl}** | " + " | ".join(cells) + " |")
        lines += ["", "Cell = RMSE(log V/K) / RMSE(V/K) / row count.", ""]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="F3 error-heatmap figure")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--max-contracts", type=int, default=None,
                    help="subsample the test split (smoke tests only; the published "
                         "figure uses the full split)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    out = Path(args.out_dir)
    stats_by_type = {}
    for option_type in ("call", "put"):
        png = out / f"fig_F3_error_heatmap_{option_type}_v5.png"
        s = make_figure(option_type, png, max_contracts=args.max_contracts, device=device)
        stats_by_type[option_type] = s
        print(f"wrote {png}")
        fl, fp = s["rmse_log"], s["rmse_price"]
        print(f"  n={s['n']:,}  T<=1day rows={s['n_T_le_1day']}  "
              f"cells populated={int((s['counts'] > 0).sum())}/{s['counts'].size}")
        print(f"  RMSE(log)  overall={s['rmse_log_overall']:.4f}  "
              f"cell range=[{np.nanmin(fl):.4f}, {np.nanmax(fl):.4f}]")
        print(f"  RMSE(price) overall={s['rmse_price_overall']:.3e}  "
              f"cell range=[{np.nanmin(fp):.3e}, {np.nanmax(fp):.3e}]")
    md = out / "fig_F3_error_heatmap_v5.md"
    write_caption(md, stats_by_type)
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
