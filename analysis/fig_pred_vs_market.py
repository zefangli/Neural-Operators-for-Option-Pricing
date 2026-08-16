"""
F5 -- predicted vs market normalized price, on the v5 test split.

One log-log scatter per option type: the model's normalized price V̂/K (σ̂ pushed
through the analytical Black-Scholes decoder, i.e. exp of the same log(V/K) the
loss is computed on) against the market's V/K (`target_v_log` = log(mid/K)),
with the y = x reference line. Points are coloured by maturity bucket using F3's
maturity edges, so the two figures read on the same grid.

SAMPLING RULE (fixed, reproducible, no cherry-picking):
    `_common.load_test_split(..., max_contracts=SAMPLE_N, seed=42)`
i.e. `np.random.default_rng(42).choice(test_row_indices, SAMPLE_N, replace=False)`
then sorted -- a uniform draw over the whole test split with a pinned seed, the
same helper and the same seed every other analysis in this repo subsamples with.
SAMPLE_N = 50,000 (of 819,342 call / 1,380,379 put rows): enough that the wings
are populated, few enough that the marks stay individually legible instead of
saturating into a black band. Nothing is filtered -- no outlier trimming, no
price floor, no moneyness window.

SCALE: log-log. Normalized prices span ~2e-6 to ~31 (calls) / ~4.6e-6 to ~0.57
(puts) -- five to seven decades -- so a linear scatter would collapse everything
but the deep-ITM calls onto the axes. Both axes share one range so y = x is the
diagonal. The decoder's training-time clamp, log(price) >= log(1e-8), is drawn as
a dashed floor: predictions sitting on it are the "clamp hits" reported in
`metrics.json`.

Provenance: identical gate to F3 -- shared `load_per_query()` (dataset_version
"v5", model_variant "per_query", manifest sha256, then `run_provenance`).

Usage (conda env `dl_new`):
    python analysis/fig_pred_vs_market.py
Outputs:
    results/fig_F5_pred_vs_market_{call,put}_v5.png
    results/fig_F5_pred_vs_market_v5.md      (caption + sampling rule + stats)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from _common import (CLAMP_LOG, PROJECT_ROOT, load_test_split,  # noqa: E402
                     model_predict_logv)
# Same provenance gate and the same maturity grid as F3 -- the two figures must
# read the same checkpoint and bucket maturity identically.
from fig_error_heatmap import T_EDGES, T_LABELS, load_per_query  # noqa: E402

SAMPLE_N = 50_000
SAMPLE_SEED = 42
COLORS = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]


def make_figure(option_type, out_png, sample_n=SAMPLE_N, device="cpu"):
    """Build one option type's F5 scatter. Returns a stats dict."""
    cfg, model, h5_path = load_per_query(option_type, device=device)
    data = load_test_split(h5_path, cfg["branch_key"], max_contracts=sample_n,
                           seed=SAMPLE_SEED)
    data.pop("branch_u", None)
    log_pred, _sigma = model_predict_logv(model, data, device=device)

    log_target = np.asarray(data["target_v_log"], dtype=np.float64).ravel()
    T_days = np.asarray(data["T"], dtype=np.float64).ravel() * 365.0
    market = np.exp(log_target)
    pred = np.exp(log_pred)

    # `normalized_price` is the stored mid/K; target_v_log is its log. They must
    # agree, or the scatter's x-axis is not the quantity the model was fit to.
    stored = np.asarray(data["normalized_price"], dtype=np.float64).ravel()
    rel = np.abs(market - stored) / np.maximum(stored, 1e-12)
    assert np.nanmax(rel) < 1e-4, (
        f"{option_type}: exp(target_v_log) disagrees with `normalized_price` by "
        f"{np.nanmax(rel):.2e} — the axes are not the same quantity")

    lo = float(min(pred.min(), market.min())) * 0.7
    hi = float(max(pred.max(), market.max())) * 1.4
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ti = np.digitize(T_days, T_EDGES[1:-1], right=True)
    for k, label in enumerate(T_LABELS):
        m = ti == k
        if not m.any():
            continue
        ax.scatter(market[m], pred[m], s=2, alpha=0.25, linewidths=0,
                   color=COLORS[k % len(COLORS)], label=f"{label}  (n={int(m.sum()):,})",
                   rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k-", lw=1.0, label="$y = x$")
    # Only drawn when it is actually on-screen: with no clamp hits the floor sits
    # decades below the data and a legend entry for an invisible line misleads.
    if np.exp(CLAMP_LOG) > lo:
        ax.axhline(np.exp(CLAMP_LOG), color="0.4", ls="--", lw=0.9,
                   label="decoder clamp $10^{-8}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlabel("market normalized price  $V/K$", fontsize=10)
    ax.set_ylabel("predicted normalized price  $\\hat V/K$", fontsize=10)
    leg = ax.legend(fontsize=8, loc="upper left", markerscale=4)
    for h in leg.legend_handles:
        try:
            h.set_alpha(1.0)
        except AttributeError:
            pass

    resid = log_pred - log_target
    stats = {
        "option_type": option_type, "n": int(pred.size),
        "pearson_log10": float(np.corrcoef(np.log10(market), np.log10(pred))[0, 1]),
        "pearson_level": float(np.corrcoef(market, pred)[0, 1]),
        "r2_price": float(1.0 - np.sum((market - pred) ** 2)
                          / np.sum((market - market.mean()) ** 2)),
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "median_abs_rel_err": float(np.median(np.abs(pred - market) / market)),
        "frac_within_1pct": float(np.mean(np.abs(pred / market - 1.0) <= 0.01)),
        "frac_within_10pct": float(np.mean(np.abs(pred / market - 1.0) <= 0.10)),
        "n_clamp_hits": int((log_pred <= CLAMP_LOG + 1e-9).sum()),
        "market_range": [float(market.min()), float(market.max())],
        "pred_range": [float(pred.min()), float(pred.max())],
        "min_price_nonneg": bool(pred.min() > 0 and market.min() > 0),
    }
    ax.set_title(f"F5 — predicted vs market price, {option_type}s (v5 test split)\n"
                 f"n={stats['n']:,} uniform sample (seed {SAMPLE_SEED}); "
                 f"$R^2$(price)={stats['r2_price']:.5f}, "
                 f"median |rel err|={stats['median_abs_rel_err']:.3%}", fontsize=10)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return stats


CAPTION = """# F5 — Predicted vs market normalized price (v5 test split)

![call](fig_F5_pred_vs_market_call_v5.png)
![put](fig_F5_pred_vs_market_put_v5.png)

**Files:** `results/fig_F5_pred_vs_market_call_v5.png`, `results/fig_F5_pred_vs_market_put_v5.png`
(generated by `analysis/fig_pred_vs_market.py`; canonical sample **v5**, run
`train_model_v3/{call,put}/results_vol_surface_v5` with `model_variant="per_query"`).

## Sampling rule (fixed, reproducible, no cherry-picking)

`_common.load_test_split(..., max_contracts=50000, seed=42)` — i.e.
`np.random.default_rng(42).choice(test_row_indices, 50000, replace=False)`, sorted. A **uniform**
draw over the whole test split (819,342 call / 1,380,379 put rows) with the repo's pinned seed 42,
via the same helper every other analysis subsamples with. **Nothing is filtered**: no outlier
trimming, no price floor, no moneyness or maturity window. 50,000 is enough to populate the wings
and few enough that individual marks stay legible instead of saturating into a black band.

## Axes and scale

* x = market normalized price `V/K` = `exp(target_v_log)` = mid/K. The script asserts this agrees
  with the stored `normalized_price` column to <1e-4 relative, so the axis is provably the quantity
  the model was fit to.
* y = predicted normalized price `V̂/K` — σ̂ from the per-query head pushed through the analytical
  Black–Scholes decoder, `exp` of the log-price the loss is computed on.
* **Log-log**, both axes on a shared range so `y = x` is the diagonal. Normalized prices span five
  to seven decades (≈2e-6 … 31 for calls, ≈4.6e-6 … 0.57 for puts); on a linear scale everything
  except the deep-ITM calls would collapse onto the axes.
* Colour = maturity bucket, using F3's maturity edges verbatim ((1,7], (7,30], (30,90], (90,365],
  >365 days), so the two figures read on the same grid.
* The dashed line at `1e-8` is the decoder's training-time price clamp, `log(clamp(price, 1e-8))`.
  Predictions resting on it are the `clamp_hit_count` reported in each run's `metrics.json`.

## Statistics (on the 50,000-row sample)

"""


def write_caption(out_md, stats_by_type):
    lines = [CAPTION,
             "| option | n | R²(price) | Pearson r (log₁₀) | Pearson r (level) | RMSE(log V/K) | "
             "median &#124;rel err&#124; | within ±1% | within ±10% | clamp hits | market range | "
             "predicted range |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in stats_by_type.values():
        lines.append(
            "| {o} | {n:,} | {r2:.6f} | {pl:.6f} | {pv:.6f} | {rl:.4f} | {mre:.3%} | {w1:.2%} | "
            "{w10:.2%} | {ch:,} | [{m0:.2e}, {m1:.2e}] | [{p0:.2e}, {p1:.2e}] |".format(
                o=s["option_type"], n=s["n"], r2=s["r2_price"], pl=s["pearson_log10"],
                pv=s["pearson_level"], rl=s["rmse_log"], mre=s["median_abs_rel_err"],
                w1=s["frac_within_1pct"], w10=s["frac_within_10pct"], ch=s["n_clamp_hits"],
                m0=s["market_range"][0], m1=s["market_range"][1],
                p0=s["pred_range"][0], p1=s["pred_range"][1]))
    lines += ["",
              "Both predicted and market normalized prices are strictly positive throughout "
              "(a Black–Scholes price at a positive σ̂ cannot be negative), which is why the "
              "log-log axes lose no rows.", ""]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="F5 predicted-vs-market price scatter")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--sample-n", type=int, default=SAMPLE_N)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    out = Path(args.out_dir)
    stats_by_type = {}
    for option_type in ("call", "put"):
        png = out / f"fig_F5_pred_vs_market_{option_type}_v5.png"
        s = make_figure(option_type, png, sample_n=args.sample_n, device=device)
        stats_by_type[option_type] = s
        print(f"wrote {png}")
        print("  n={n:,}  R2(price)={r2:.6f}  r(log10)={pl:.6f}  RMSE(log)={rl:.4f}  "
              "median|rel|={mre:.3%}  within10%={w10:.2%}  clamp hits={ch}".format(
                  n=s["n"], r2=s["r2_price"], pl=s["pearson_log10"], rl=s["rmse_log"],
                  mre=s["median_abs_rel_err"], w10=s["frac_within_10pct"],
                  ch=s["n_clamp_hits"]))
    md = out / "fig_F5_pred_vs_market_v5.md"
    write_caption(md, stats_by_type)
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
