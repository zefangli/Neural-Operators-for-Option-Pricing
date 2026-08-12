# Phase 2 v3 — Progress Snapshot

Last updated: 2026-06-11. This file describes the **current state of `train_model_v3/`** only.

## What the model is

Per-contract implied-volatility estimation feeding an analytical Black-Scholes pricer. The
network never predicts price directly: it emits a **per-query σ̂** for each contract
`(log_moneyness, T)` against an encoded market state, and `σ̂` is pushed through closed-form BS to
get `V/K`. The model predicts `log(V/K)`; the loss is plain data MSE on `log(V/K)` (no PDE /
arbitrage / BS-anchor terms).

Two architectures are implemented for every branch/option combo, sharing the same FNO encoder and
the same hyperparameters — they differ only in how `(market latent, query)` produces σ̂:

- **MLP-head** (`train_*.py`): FNO encoder → latent; `ConditionalVolHead` MLP on
  `[latent, log_m, T]` → softplus → σ̂.
- **DeepONet** (`train_*_don.py`): FNO **branch** → basis vector; MLP **trunk** on `(log_m, T)` →
  basis vector; `σ̂ = softplus(⟨branch, trunk⟩ + sigma_bias) + sigma_floor`. `sigma_bias` is
  initialized to −1.50 so σ̂ ≈ 0.20 at init.

## Current scripts (all 12)

Each MLP-head/DeepONet pair shares an identical model and hyperparameters; pairs and branches
differ only by the config triple `branch_key` / grid / modes (+ `option_type`, h5, `results_dir`).

```
train_model_v3/
  call/   (deeponet_tensors_call.h5, option_type="call")
    train_vol_surface.py    train_vol_surface_don.py     branch_u      grid 11x17  modes 4x8
    train_spot_history.py   train_spot_history_don.py    spot_history  grid 21x5   modes 6x2
    train_vix_history.py    train_vix_history_don.py      vix_history   grid 21x4   modes 6x2
  put/    (deeponet_tensors_put.h5,  option_type="put")
    train_vol_surface.py    train_vol_surface_don.py     branch_u      grid 11x17  modes 4x8
    train_spot_history.py   train_spot_history_don.py    spot_history  grid 21x5   modes 6x2
    train_vix_history.py    train_vix_history_don.py      vix_history   grid 21x4   modes 6x2
```

MLP-head runs write to `results_<branch>/`, DeepONet runs to `results_<branch>_don/`. Each run dir
holds `config.json`, `best_model_phase1.pth`, `best_model.pth` (best across both phases — use for
eval), `final_model.pth`, `loss_history.txt`, `loss_plot.png`.

## Shared configuration (identical across all 12)

- FNO encoder/branch: 4 `SpectralConv2d` blocks (+ 1×1-conv skips, SiLU), `fno_width=32`, then
  flat → 128 → latent/basis. Latent/`p_dim` = 128, head/`trunk_hidden` = 128. DeepONet adds
  `sigma_floor=1e-6` and the scalar `sigma_bias`.
- Training: two-phase Adam (`epochs_adam=100`, `lr=1e-3`) then fine-tune (`epochs_finetune=50`,
  `finetune_lr=1e-5`), `batch_size=256`, `grad_clip=1.0`, `patience_adam=patience_finetune=5`,
  `min_finetune_epochs=10`, seed 42. Early stop on val MSE; `ReduceLROnPlateau` in phase 1.
- Loss: `compute_loss` has **MSE active** with `F.huber_loss(delta=1.0)` commented one line above —
  a deliberate one-line toggle (MSE is the default).
- Param counts at the 128/128 config: FNO branch ≈ 1.05M (11×17 grid) / 0.55M (21×5 grid); MLP head
  +33,409; DeepONet trunk +49,920 +1 (`sigma_bias`). Branch is identical between the A and B
  variant of the same branch (matched capacity).

## Split / data contract

Each `deeponet_tensors_{call,put}.h5` holds row-aligned datasets: `branch_u`, `spot_history`,
`vix_history`, `trunk_y = [log_moneyness, T_years, r, q]`, `target_v_log = log(mid/K)`, plus
`moneyness`, `normalized_price`, `date`, `split_id` (0=train, 1=val, 2=test). **The split is by
unique trade date (80/10/10), not by row** — a deliberate no-leakage time-ordered split; preserve it.

## Two numerical facts baked into the eval

1. **Near-expiry / near-ATM log tail.** ~1.5% of contracts (T→0, ATM) carry ~93% of the log-space
   SSE. The inverse problem `log(price) → σ` is genuinely ill-conditioned as T→0
   (`∂log(price)/∂σ` blows up), so log-perfection there is unattainable while the *absolute* price
   error is ~1e-6 (economically nil). Hence R²(price) ≈ 0.9998 while full-domain R²(log) ≈ 0.89.
   The eval reports both, and the **headline log metric is `R²(log, T > 1/365)`** (≈0.9924 on the
   trained DeepONet vol_surface weights). A per-sample log-error percentile + tail diagnostic block
   in the test-eval characterizes this cluster (count, share of SSE, target/pred ranges, clamp-hit
   fraction, log_m / T ranges).
2. **Two pricing modes.** Default training/eval uses `log(clamp(price, 1e-8))`. The `--stable-log`
   flag swaps in `bs_log_normalized_price` (`log_ndtr` + `log(-expm1(...))`), exact but with a
   gradient that explodes near ATM. **`--stable-log` is eval-only** (it NaNs out training because
   the global `clip_grad_norm_` then crushes the rest of the batch). It confirms the clamp masks a
   real model error on the tail; it does not create one.

## CLI flags (every script)

`--epochs-adam`, `--epochs-finetune`, `--max-train-batches` (smoke), `--batch-size`, `--lr`,
`--finetune-lr`, `--seed`, `--eval-only` (load `best_model.pth`, run test eval + diagnostic, write
nothing), `--stable-log` (eval-only diagnostic pricer). MLP-head also: `--latent-dim`,
`--head-hidden`. DeepONet also: `--p-dim`, `--trunk-hidden`, `--sigma-floor`.

## Results we have

**All 12 runs are now trained** (`{call,put} × {vol_surface, spot_history, vix_history} × {MLP-head
(A), DeepONet (B)}`). Metrics below are read from the bottom of each run's `loss_history.txt`
(test split). Price metrics are on the **normalized price `V/K`**, not raw dollars.

| option | branch | arch | R²(price) | R²(log, T>1day) | R²(log, full) | RMSE(log) | RMSE(price) | MAE(price) |
|---|---|---|---|---|---|---|---|---|
| call | vol_surface  | A | 0.999851 | 0.992481 | 0.894105 | 0.835967 | 0.009895 | 0.000757 |
| call | vol_surface  | B | 0.999854 | 0.992413 | 0.894330 | 0.835075 | 0.009795 | 0.000657 |
| call | spot_history | A | 0.999834 | 0.948926 | 0.858569 | 0.966102 | 0.010428 | 0.002325 |
| call | spot_history | B | 0.998842 | 0.950056 | 0.859608 | 0.962546 | 0.027581 | 0.003303 |
| call | vix_history  | A | 0.999811 | 0.905300 | 0.822470 | 1.082397 | 0.011135 | 0.003280 |
| call | vix_history  | B | 0.999730 | 0.915567 | 0.830493 | 1.057656 | 0.013315 | 0.002758 |
| put  | vol_surface  | A | 0.996177 | 0.979764 | 0.832595 | 0.905997 | 0.001501 | 0.000626 |
| put  | vol_surface  | B | 0.997241 | 0.981660 | 0.833637 | 0.903173 | 0.001275 | 0.000539 |
| put  | spot_history | A | 0.963103 | 0.909478 | 0.776980 | 1.045718 | 0.004664 | 0.002555 |
| put  | spot_history | B | 0.836896 | 0.887795 | 0.760506 | 1.083651 | 0.009807 | 0.003808 |
| put  | vix_history  | A | 0.964931 | 0.902385 | 0.770127 | 1.061662 | 0.004547 | 0.002523 |
| put  | vix_history  | B | 0.948355 | 0.900832 | 0.769931 | 1.062114 | 0.005518 | 0.002779 |

What the matrix says:
- **Branch ranking is consistent: `vol_surface` ≫ `spot_history` > `vix_history`** on the headline
  R²(log, T>1day), for both call and put. The IV-surface input prices best by a wide margin (call:
  .992 vs .949/.950 vs .905/.916). This answers the live research question — the gridded surface
  carries far more pricing-relevant signal than SPX or VIX history alone.
- **A vs B tie on `vol_surface`** (within seed noise, both option types). On the weaker branches B
  is mixed: slightly better on call/vix_history, but **notably unstable on put/spot_history (B
  R²(price) = 0.837**, a clear outlier vs A's 0.963) — likely patience-limited / an unstable run;
  flag for a re-run before publication.
- **Put R²(price) sits below call** across the board (vol_surface .996 vs .9999). Put normalized
  prices are smaller, so the variance denominator differs; R²(log, T>1day) remains strong (.980).

## Next steps

> **Publication-artifacts work has started — see `PROGRESS_paper_artifacts.md`.**
> The `analysis/` package (metrics-JSON dump, results aggregation, greeks check)
> is written but **not yet run** (GPU was occupied). Run the verification gates in
> that doc §2 first. Baselines, the scalar-σ̂ ablation, and figures are specced
> there (§4) with the resolved data findings (branch_u is a tenor×delta surface;
> no per-contract observed-IV column → BS-invert for the smile figure).

The core 12-run matrix is **done** (✅), and the two comparison questions it was meant to answer are
resolved (branch ranking + A-vs-B tie above). The live work now shifts to the publication artifacts
laid out in `pub_plan.md`; none of these exist yet:

1. ~~Train each of the 6 branches under both A and B (12 runs).~~ **Done** — see table above.
2. **Re-run the put/spot_history DeepONet (B)** — its R²(price)=0.837 is an outlier; confirm
   whether it is patience-limited / seed-unstable before it goes in a paper table.
3. **External baselines** (`analysis/baselines.py`, new) — BS constant-σ (ATM IV of day, VIX) and
   classic IV interpolation (spline / NN / SVI) over the same HDF5 test split, same metrics schema.
   Establishes the value added by the learned per-query surface.
4. **Per-query σ̂ vs single-scalar σ̂ ablation** — the headline ablation (smile recovery); quantify
   the R²(log, T>1day) gap and produce the smile figure (F2).
5. **Greeks check** (`analysis/greeks_check.py`, new) — autograd ∂price/∂S and vega through the BS
   decoder vs closed-form, expect ~1e-7 agreement; substantiates the "exact greeks" claim.
6. **Aggregation + figures** — dump per-run `metrics.json`, aggregate to LaTeX tables T1–T4, and
   build figures F1–F6 (F7 training curves already exist per run as `loss_plot.png`).
7. **Draft `paper/`** (fresh, not `final_report.tex`). A paper-skeleton draft now lives in
   `report.md`.
8. Optional, applied uniformly if pursued: exclude/down-weight `T < 1/365` during training
   (consistent with the filtered eval); train longer if runs look patience-limited.

## How to run

```
conda activate dl_new

# Fresh train (MLP-head / DeepONet):
python train_model_v3/call/train_vol_surface.py
python train_model_v3/call/train_vol_surface_don.py

# Eval only (no writes):
python train_model_v3/call/train_vol_surface.py --eval-only

# Diagnostic eval with the stable log_ndtr pricer (read-only; huge tail log-errors expected):
python train_model_v3/call/train_vol_surface.py --eval-only --stable-log

# Smoke test:
python train_model_v3/call/train_spot_history.py --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0

# Matched-capacity overrides:
#   MLP-head:  --latent-dim 128 --head-hidden 128
#   DeepONet:  --p-dim 128 --trunk-hidden 128

# Switch loss to Huber: in compute_loss, uncomment the F.huber_loss line, comment the F.mse_loss line.
```

Each `train_*.py` script is **self-contained** — it redefines the full stack (SpectralConv2d, FNO
encoder, head/trunk, BS pricer, dataset, train loop). To change the architecture, edit each script;
there is no shared library.
