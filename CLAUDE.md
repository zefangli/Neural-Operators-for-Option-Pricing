# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Deep-learning option pricing on SPX (OptionMetrics / WRDS data). The model
learns to price options by estimating a per-contract implied volatility and
feeding it through an **analytical Black-Scholes formula** — the network never
predicts price directly. This is a research / experiment codebase (no package,
no test suite, no CI); progress is tracked in markdown narrative files, not git
(the repo is not even a git repository).

## Environment & commands

There is no build system. Everything is run as standalone Python scripts under
the conda env `dl_new` (deps: torch, polars, numpy, h5py, matplotlib).

```bash
conda activate dl_new
```

**Data pipeline** (run from `wrds_data_2020-2025/`, the current dataset):
```bash
python pre_process_1_data.py            # raw CSVs -> deeponet_training_data_parts/*.parquet
python pre_process_2_hdf5.py --type call # parquet -> deeponet_tensors_call.h5
python pre_process_2_hdf5.py --type put  # parquet -> deeponet_tensors_put.h5
python pre_process_3_print_dataset.py    # inspect the HDF5
```
Phase 1 skips already-written yearly parts unless `--force`. Phase 2 falls back
to `deeponet_training_data_parts/` if the single parquet is absent.

**Training** (the active work is in `train_model_v3/`):
```bash
# Fresh train (MLP-head, the Pilot A architecture):
python train_model_v3/call/train_vol_surface.py
# DeepONet variant (Pilot B):
python train_model_v3/call/train_vol_surface_don.py

# Eval only — loads best_model.pth, runs test eval + tail diagnostic, writes nothing:
python train_model_v3/call/train_vol_surface.py --eval-only

# Diagnostic eval with the numerically-stable log_ndtr pricer (read-only;
# expect huge log errors on the near-expiry tail — that is expected, see below):
python train_model_v3/call/train_vol_surface.py --eval-only --stable-log
```
Common overrides: `--epochs-adam`, `--epochs-finetune`, `--max-train-batches`
(smoke test), `--batch-size`, `--lr`, `--finetune-lr`, `--seed`. The MLP-head
script also takes `--latent-dim`/`--head-hidden`; the DeepONet script takes
`--p-dim`/`--trunk-hidden`/`--sigma-floor`. There is no "run a single test"
because there are no unit tests.

A run writes into its `results_*/` dir: `config.json`, `best_model_phase1.pth`,
`best_model.pth` (best across both phases — use this for eval), `final_model.pth`,
`loss_history.txt`, `loss_plot.png`. `--eval-only` deliberately does **not**
overwrite these.

## Architecture (the big picture)

Every `train_*.py` script is **self-contained** — it redefines the full model
stack (SpectralConv2d, FNO encoder, head, BS pricer, dataset, train loop) rather
than importing shared modules. To change the architecture you edit each script;
there is no central library. The scripts differ only in a handful of config
values, not in structure.

The forward pass (v3):
1. **FNO market-state encoder** — a 2D Fourier Neural Operator (4 SpectralConv2d
   blocks with 1x1-conv skips, SiLU) over a market-state grid → latent vector.
2. **Conditional vol head** — `(latent, log_moneyness, T_years) → sigma_hat`
   (softplus-positive). This is the key v3.1 idea: σ̂ is *per contract query*, not
   one scalar per market state, so the model can fit the volatility smile/skew.
   - MLP-head variant (`train_vol_surface.py`): MLP on the concatenation.
   - DeepONet variant (`train_vol_surface_don.py`): branch⋅trunk dot product
     readout, `σ̂ = softplus(⟨b,t⟩ + bias) + floor`, bias init ≈ −1.5 so σ̂≈0.20.
3. **Analytical Black-Scholes** — `bs_normalized_price(log_m, T, r, q, σ̂)` gives
   V/K; the model predicts `log(V/K)`. There is **no** Network 2 / learned pricer
   (that was the v1/v2 design, removed in v3), and the loss is plain data MSE on
   `log(V/K)` — no PDE / arbitrage / BS-anchor terms.

**Three branch inputs**, one script each, selected by the config triple
`branch_key` / `grid_h` × `grid_w` / `modes1` × `modes2`. The 1-D HDF5 vector is
reshaped to the 2-D grid inside the FNO encoder:
| script | branch_key | grid | flat dim |
|---|---|---|---|
| `train_vol_surface.py` | `branch_u` (implied-vol surface) | 11×17 | 187 |
| `train_spot_history.py` | `spot_history` (21d SPX OHLCV) | 21×5 | 105 |
| `train_vix_history.py` | `vix_history` (21d VIX OHLC) | 21×4 | 84 |

`call/` and `put/` are parallel copies pointing at `deeponet_tensors_call.h5` /
`deeponet_tensors_put.h5` and passing `option_type` to the BS formula.

**Data contract (HDF5).** Each `deeponet_tensors_{call,put}.h5` holds aligned
row-indexed datasets: `branch_u`, `spot_history`, `vix_history` (the three branch
inputs), `trunk_y` = `[log_moneyness, T_years, r, q]`, `target_v_log` =
`log(mid/K)`, plus `moneyness`, `normalized_price`, `date`, and `split_id`
(0=train, 1=val, 2=test). **The split is by unique trade date (80/10/10), not by
row** — this is a deliberate no-leakage time-ordered split; preserve it.

## Two recurring numerical facts you must know

These dominate the project's metrics and have already been investigated at length
(see `PROGRESS_phase2_v3.md`). Do not re-discover them as "bugs":

1. **The near-expiry / near-ATM log tail.** ~1.5% of contracts (T→0, ATM) carry
   ~93% of the log-space SSE. The inverse problem `log(price) → σ` is genuinely
   ill-conditioned as T→0 (`∂log(price)/∂σ` blows up), so log-perfection there is
   unattainable, while the *absolute* price error is ~1e-6 (economically nil).
   This is why `R²(price) ≈ 0.9998` while full-domain `R²(log) ≈ 0.89`. The
   **headline log metric is `R²(log, T > 1/365)` (≈0.9924)** — log accuracy on
   contracts with more than ~1 trading day to expiry. Report both.
2. **The pricing path has two modes.** Default training/eval uses
   `log(clamp(price, 1e-8))`. `--stable-log` swaps in `bs_log_normalized_price`
   (`log_ndtr` + `log(-expm1(...))`), which is exact but whose gradient explodes
   near ATM. **`--stable-log` is eval-only**; dropping it into training NaNs out
   because the global `clip_grad_norm_` then crushes the rest of the batch. The
   stable-log eval *confirmed* the clamp masks a real model error on the tail, it
   does not create one.

`compute_loss` ships with **MSE active and `F.huber_loss(delta=1.0)` commented
out** — a deliberate one-line toggle. Huber was tried and reverted (it stopped
earlier and hurt R²(price) without moving R²(log)); leave MSE as the default
unless a future dataset has heavier tails.

## Status, layout, and history

- **`PROGRESS_phase2_v3.md` is the source of truth for current state** — read it
  before resuming. As of the last session: the call/vol_surface pilots are
  considered done; the next step is choosing MLP-head vs DeepONet and propagating
  the per-query architecture to the 5 still-on-v3-baseline scripts
  (`call/{spot_history,vix_history}`, all three `put/*`).
- `plan_phase1.md`, `plan_phase2{,_v2,_v3}.md` are design docs; `plan_phase2_v3.md`
  predates the v3.1/v3.2 changes (per-query σ̂, filtered eval, `--eval-only`/
  `--stable-log`) — trust `PROGRESS_phase2_v3.md` over it where they disagree.
- `train_model/` (v1) and `train_model_v2/` are superseded; v2 still had the
  learned two-network design with `pretrain_net2.py`. `OLD_*/` and `archive/` and
  `wrds_data/` (the older 2020-2025-named-but-different dataset) are legacy —
  `wrds_data_2020-2025/` is the current data.
- `final_report.tex` / `DLforPhysicalSystems_project.pdf` are the writeup.
