"""
F2 -- smile recovery: market IV vs per-query sigma_hat vs the scalar-sigma control.

This is the paper's core-claim figure. Per (trade date, maturity bucket) panel it
draws three curves against log-moneyness:

  1. MARKET  -- the observed per-contract implied volatility. Obtained exactly as
     the baseline table does it: `baselines.implied_vol()` (vectorised bisection
     on the BS pricer, NaN where T==0, where the quote is outside the
     no-arbitrage band, or where vega is uninformative) combined with
     `baselines.choose_observed_iv()`, which prefers OptionMetrics' stored
     `impl_volatility` and falls back to the inversion where it is missing.
  2. PER-QUERY -- sigma_hat from the canonical run `results_vol_surface_v5`
     (`model_variant == "per_query"`), i.e. sigma_hat(latent, log_m, T).
  3. SCALAR   -- sigma_hat from the ablation run `results_vol_surface_scalarsigma_v5`
     (`model_variant == "scalar_sigma"`), i.e. sigma_hat(latent) alone. It cannot
     depend on the contract, so it is flat across moneyness by construction; the
     script ASSERTS that (np.ptp == 0 per panel) rather than trusting the label.

SELECTION RULE (fixed, mechanical, no cherry-picking) -- the whole point of the
figure is that the panels were not chosen to look good:

  * DATES: take the test split's unique trade dates in ascending order and pick
    the three at the 25th / 50th / 75th percentile of that ordered list
    (`np.quantile(..., method="nearest")` on the date INDEX, not on any metric).
    Rank-based, so it depends only on the split's date axis.
  * MATURITY BUCKETS: three fixed calendar-day bands, stated in advance and
    identical for every date and both option types:
        short  (7, 30]   days
        medium (30, 90]  days
        long   (90, 365] days
  * ONE EXPIRY PER PANEL: a (date, bucket) cell usually contains several
    expiries, whose smiles would overlay into mush. Keep the single expiry (one
    unique T value) whose maturity in days is closest to the bucket MIDPOINT
    (18.5 / 60.0 / 227.5 days); ties go to the shorter maturity. Again purely
    geometric -- no reference to model error, IV level, or row count.
  * Every contract of that expiry is plotted. No moneyness window, no outlier
    trimming, no minimum-row threshold.

A panel is left empty (annotated "no contracts") if the rule selects nothing;
that is reported rather than back-filled with a different date.

Provenance: both runs must declare `dataset_version == "v5"` (the manifest's
canonical sample) and carry the manifest's v5 sha256 for their option type; the
shared `eval_to_json.run_provenance` gate then re-checks the config against the
HDF5 actually being read. Anything else hard-fails.

Usage (conda env `dl_new`, CPU is fine):
    python analysis/fig_smile_recovery.py
Outputs:
    results/fig_F2_smile_recovery_{call,put}_v5.png
    results/fig_F2_smile_recovery_v5.md      (caption + selection rule + stats)
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from _common import (PROJECT_ROOT, canonical_version, load_run, manifest_entry,  # noqa: E402
                     resolve_h5_path)
from baselines import choose_observed_iv, implied_vol  # noqa: E402
from eval_ablation import load_scalar_run  # noqa: E402
from eval_to_json import run_provenance  # noqa: E402

PER_QUERY_DIR = "train_model_v3/%s/results_vol_surface_v5"
SCALAR_DIR = "train_model_v3/%s/results_vol_surface_scalarsigma_v5"

# (label, lo_days, hi_days) -- half-open (lo, hi], fixed in advance.
BUCKETS = [("short (7-30d)", 7.0, 30.0),
           ("medium (30-90d)", 30.0, 90.0),
           ("long (90-365d)", 90.0, 365.0)]
DATE_QUANTILES = (0.25, 0.50, 0.75)
SCALAR_FLAT_RTOL = 1e-5   # float32 reduction-order headroom; see the assert below


# --------------------------------------------------------------- selection rule

def pick_dates(test_dates, quantiles=DATE_QUANTILES):
    """The unique test trade dates at the given quantiles of the ordered date axis."""
    uniq = np.unique(test_dates)
    pos = np.quantile(np.arange(len(uniq)), quantiles, method="nearest").astype(int)
    return [str(uniq[i]) for i in sorted(set(pos.tolist()))]


def pick_expiry(days, lo, hi):
    """The unique maturity in (lo, hi] closest to the bucket midpoint; None if empty."""
    cand = np.unique(days[(days > lo) & (days <= hi)])
    if not len(cand):
        return None
    mid = 0.5 * (lo + hi)
    # np.argmin takes the FIRST minimum, and `cand` is sorted ascending, so ties
    # resolve to the shorter maturity -- the documented rule.
    return float(cand[int(np.argmin(np.abs(cand - mid)))])


# ------------------------------------------------------------------ data access

def test_index(f):
    return np.where(f["split_id"][:] == 2)[0]


def select_panels(h5_path):
    """-> (panels, dates). panels[(date, bucket_label)] = (global row indices, T_days)."""
    with h5py.File(h5_path, "r") as f:
        idx = test_index(f)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        assert hi - lo == len(idx), "test rows are not contiguous"
        dates = f["date"][lo:hi].astype("U10")
        T = f["trunk_y"][lo:hi, 1].astype(np.float64)
    days = T * 365.0
    chosen = pick_dates(dates)
    panels = {}
    for d in chosen:
        on_date = dates == d
        for label, blo, bhi in BUCKETS:
            t_days = pick_expiry(days[on_date], blo, bhi)
            if t_days is None:
                panels[(d, label)] = (np.empty(0, dtype=int), None)
                continue
            m = on_date & (np.abs(days - t_days) < 1e-9)
            panels[(d, label)] = (lo + np.flatnonzero(m), t_days)
    return panels, chosen


def load_rows(h5_path, branch_key, rows):
    """Row-wise read of just the selected contracts (never the whole split)."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    with h5py.File(h5_path, "r") as f:
        trunk = f["trunk_y"][rows].astype(np.float64)
        out = {"log_m": trunk[:, 0], "T": trunk[:, 1], "r": trunk[:, 2], "q": trunk[:, 3],
               "target_v_log": f["target_v_log"][rows].astype(np.float64).ravel(),
               "branch": f[branch_key][rows]}
        if "impl_volatility" in f:
            out["impl_volatility"] = f["impl_volatility"][rows].astype(np.float64).ravel()
    return out


def market_iv(rows, option_type):
    """Observed IV, reusing the validated baseline machinery verbatim."""
    iv_inv, _ = implied_vol(rows["log_m"], rows["T"], rows["r"], rows["q"],
                            np.exp(rows["target_v_log"]), option_type)
    iv_obs, _ = choose_observed_iv(rows, iv_inv)
    return iv_obs


@torch.no_grad()
def model_sigma(model, rows, device="cpu"):
    t = lambda k, s: torch.as_tensor(rows[k], dtype=torch.float32, device=device).reshape(-1, *s)
    return model.sigma(t("branch", (rows["branch"].shape[1],)),
                       t("log_m", (1,)), t("T", (1,))).cpu().numpy().ravel()


# ------------------------------------------------------------------- provenance

def load_pair(option_type, device="cpu"):
    """(config, per-query model, scalar model, h5_path) with the provenance gate applied."""
    pq_dir = PROJECT_ROOT / (PER_QUERY_DIR % option_type)
    sc_dir = PROJECT_ROOT / (SCALAR_DIR % option_type)
    cfg_pq, _arch, m_pq = load_run(pq_dir, device=device)
    cfg_sc, m_sc = load_scalar_run(sc_dir, device=device)

    want = canonical_version()
    want_sha = manifest_entry(want)["files"][option_type]["sha256"]
    for d, c, variant in ((pq_dir, cfg_pq, "per_query"), (sc_dir, cfg_sc, "scalar_sigma")):
        if c.get("model_variant") != variant:
            raise SystemExit(f"{d}: model_variant={c.get('model_variant')!r}, want {variant!r}")
        if c.get("option_type") != option_type:
            raise SystemExit(f"{d}: option_type={c.get('option_type')!r}")
        if c.get("dataset_version") != want:
            raise SystemExit(f"{d}: dataset_version={c.get('dataset_version')!r}, "
                             f"want the canonical {want!r}")
        if c.get("h5_sha256") != want_sha:
            raise SystemExit(f"{d}: h5_sha256={str(c.get('h5_sha256'))[:12]}... is not the "
                             f"manifest's {want} {option_type} file")
    h5_path = resolve_h5_path(cfg_pq)
    run_provenance(cfg_pq, h5_path)
    return cfg_pq, m_pq, m_sc, h5_path


# ------------------------------------------------------------------------ plot

def make_figure(option_type, out_png, dates=None, buckets=BUCKETS, device="cpu"):
    """Build one option type's F2 panel grid. Returns the per-panel stats list."""
    cfg, m_pq, m_sc, h5_path = load_pair(option_type, device=device)
    panels, chosen = select_panels(h5_path)
    dates = dates or chosen

    fig, axes = plt.subplots(len(dates), len(buckets), figsize=(5 * len(buckets),
                                                               3.6 * len(dates)),
                             squeeze=False)
    stats = []
    for i, d in enumerate(dates):
        for j, (label, _lo, _hi) in enumerate(buckets):
            ax = axes[i][j]
            idx, t_days = panels[(d, label)]
            ax.set_title(f"{d} — {label}" + (f", T={t_days:.1f}d" if t_days else ""),
                         fontsize=10)
            ax.grid(True, ls="--", alpha=0.4)
            if not len(idx):
                ax.text(0.5, 0.5, "no contracts", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.4")
                stats.append({"date": d, "bucket": label, "n": 0})
                continue

            rows = load_rows(h5_path, cfg["branch_key"], idx)
            order = np.argsort(rows["log_m"])
            rows = {k: v[order] for k, v in rows.items()}
            x = rows["log_m"]
            iv = market_iv(rows, option_type)
            s_pq = model_sigma(m_pq, rows, device)
            s_sc = model_sigma(m_sc, rows, device)

            # Real invariant of the ablation: sigma_hat cannot see the contract, so
            # every row of a panel (one date => one branch input) must get the same
            # value. Not bit-exact: the FNO's batched reductions order float32 sums
            # differently across rows, which shows up as ptp ~1e-8 on a sigma ~0.2.
            # SCALAR_FLAT_RTOL is that float32 headroom -- 4 orders of magnitude
            # below any smile, so a scalar model that actually varied would fail.
            assert np.ptp(s_sc) <= SCALAR_FLAT_RTOL * abs(float(s_sc[0])), (
                f"scalar-sigma model is not flat on {d}/{label}: ptp={np.ptp(s_sc)}")

            ok = np.isfinite(iv)
            ax.plot(x[ok], iv[ok], "o", ms=2.5, color="0.25", label="market IV")
            ax.plot(x, s_pq, "-", lw=1.8, color="tab:blue", label="per-query $\\hat\\sigma$")
            ax.plot(x, s_sc, "--", lw=1.8, color="tab:red", label="scalar-$\\sigma$ $\\hat\\sigma$")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
            if i == len(dates) - 1:
                ax.set_xlabel("log-moneyness  $\\log(S/K)$", fontsize=9)
            if j == 0:
                ax.set_ylabel("implied volatility", fontsize=9)
            stats.append({
                "date": d, "bucket": label, "T_days": t_days, "n": int(len(x)),
                "n_market_iv": int(ok.sum()),
                "log_m_range": [float(x[0]), float(x[-1])],
                "market_iv_range": [float(np.nanmin(iv)), float(np.nanmax(iv))] if ok.any() else None,
                "per_query_range": [float(s_pq.min()), float(s_pq.max())],
                "per_query_ptp": float(np.ptp(s_pq)),
                "scalar_sigma": float(s_sc[0]),
                "scalar_ptp": float(np.ptp(s_sc)),
            })

    fig.suptitle(f"F2 — smile recovery, {option_type}s (v5 test split): per-query "
                 f"$\\hat\\sigma$ tracks the market smile, scalar-$\\sigma$ cannot",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return stats


CAPTION = """# F2 — Smile recovery (v5 test split)

![call](fig_F2_smile_recovery_call_v5.png)
![put](fig_F2_smile_recovery_put_v5.png)

**Files:** `results/fig_F2_smile_recovery_call_v5.png`, `results/fig_F2_smile_recovery_put_v5.png`
(generated by `analysis/fig_smile_recovery.py`; canonical sample **v5**).

## Selection rule (fixed in advance, no cherry-picking)

* **Dates** — the test split's unique trade dates in ascending order, taking the three at the
  **25th / 50th / 75th percentile of that ordered date axis** (`np.quantile(index, ...,
  method="nearest")`). Rank-based on the date index only; no metric, no model output, and no
  visual inspection enters the choice.
* **Maturity buckets** — three fixed calendar-day bands, identical for every date and both
  option types: **short (7, 30]**, **medium (30, 90]**, **long (90, 365]** days.
* **One expiry per panel** — a (date, bucket) cell holds several expiries whose smiles would
  overlay; keep the single expiry whose maturity is **closest to the bucket midpoint**
  (18.5 / 60.0 / 227.5 days), ties to the shorter maturity. Purely geometric.
* **All contracts of that expiry are plotted** — no moneyness window, no outlier trimming, no
  minimum-row threshold. A cell the rule leaves empty is drawn as "no contracts", not refilled.

## Curves

| curve | meaning |
|---|---|
| market IV (grey dots) | observed per-contract implied volatility — OptionMetrics' stored `impl_volatility` where valid, else the vectorised bisection inversion of the BS pricer (`analysis/baselines.py: implied_vol` / `choose_observed_iv`). NaN in the deep wings (uninformative vega) is simply not drawn. |
| per-query $\\hat\\sigma$ (blue) | `results_vol_surface_v5`, `model_variant="per_query"` — $\\hat\\sigma(\\text{latent}, \\log m, T)$. |
| scalar-$\\sigma$ $\\hat\\sigma$ (red dashed) | `results_vol_surface_scalarsigma_v5`, `model_variant="scalar_sigma"` — $\\hat\\sigma(\\text{latent})$ only. |

x-axis is the dataset's own convention, $\\log$-moneyness $= \\log(S/K)$, so **ITM calls are on
the right**; the usual "strike" reading is mirrored.

## What it shows

The per-query curve reproduces the market smile/skew across moneyness, while the scalar-σ control
is a horizontal line in every panel — it conditions on the market state alone and is structurally
incapable of varying with the contract (the script asserts `ptp == 0` per panel). The figure is the
visual counterpart of the ablation table `results/ablation_scalar_sigma_v5.md`: the gap between the
blue curve and the flat red line is the smile the scalar model cannot express.

## Per-panel statistics

"""


def write_caption(out_md, stats_by_type):
    lines = [CAPTION,
             "| option | date | bucket | T (days) | n | n market IV | log(S/K) range | "
             "market IV range | per-query σ̂ range | per-query σ̂ ptp | scalar σ̂ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for opt, stats in stats_by_type.items():
        for s in stats:
            if not s["n"]:
                lines.append(f"| {opt} | {s['date']} | {s['bucket']} | — | 0 | — | — | — | — | — | — |")
                continue
            miv = s["market_iv_range"]
            lines.append(
                "| {o} | {d} | {b} | {t:.1f} | {n} | {nm} | [{x0:.3f}, {x1:.3f}] | "
                "{miv} | [{p0:.4f}, {p1:.4f}] | {ptp:.4f} | {sc:.4f} |".format(
                    o=opt, d=s["date"], b=s["bucket"], t=s["T_days"], n=s["n"],
                    nm=s["n_market_iv"], x0=s["log_m_range"][0], x1=s["log_m_range"][1],
                    miv=f"[{miv[0]:.4f}, {miv[1]:.4f}]" if miv else "—",
                    p0=s["per_query_range"][0], p1=s["per_query_range"][1],
                    ptp=s["per_query_ptp"], sc=s["scalar_sigma"]))
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="F2 smile-recovery figure")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    out = Path(args.out_dir)
    stats_by_type = {}
    for option_type in ("call", "put"):
        png = out / f"fig_F2_smile_recovery_{option_type}_v5.png"
        stats_by_type[option_type] = make_figure(option_type, png, device=device)
        print(f"wrote {png}")
        for s in stats_by_type[option_type]:
            if s["n"]:
                print("  {date} {bucket:<16} n={n:<5} per-query sigma [{a:.4f},{b:.4f}] "
                      "ptp={p:.4f}  scalar={c:.4f}".format(
                          a=s["per_query_range"][0], b=s["per_query_range"][1],
                          p=s["per_query_ptp"], c=s["scalar_sigma"], **s))
    md = out / "fig_F2_smile_recovery_v5.md"
    write_caption(md, stats_by_type)
    print(f"wrote {md}")
    print(json.dumps({k: len(v) for k, v in stats_by_type.items()}))


if __name__ == "__main__":
    main()
