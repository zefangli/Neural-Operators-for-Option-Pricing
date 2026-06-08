# Phase 2 v3 — Progress Snapshot

Last updated: 2026-06-01. Resume from here.

## TL;DR of where we are

v3 removed Network 2 (analytical BS replaces it). v3.1 added **per-query σ̂** (DeepONet branch/trunk), lifting R²(log) from 0.71 → 0.89 and saturating R²(price) at ~0.9998. We then diagnosed the remaining log-space tail: it's a **single structural cluster of near-expiry (T ≈ 0), near-ATM contracts** carrying 93% of the log-SSE, where log-price is ill-conditioned (small σ̂ error → huge log error). The `--stable-log` (log_ndtr) eval proved the clamp was *masking* a genuine model error there, not creating one — the inverse problem `log(price) → σ` is unattainably sharp as T→0. **Huber loss was tried and reverted: it stopped earlier and slightly hurt R²(price) without moving R²(log).** Current canonical setup: **MSE training** + **near-expiry-filtered R²(log, T>1day) = 0.9924** as the honest "meaningful-domain" metric, paired with full-domain **R²(price) = 0.999854**.

## What's done

### 1. v3 baseline (analytical BS, single σ̂ per market state) — COMPLETE
- `plan_phase2_v3.md` rewritten to describe the analytical-BS plan (this doc supersedes parts of it; see "Plan file status" below).
- `train_model_v3/{call,put}/pretrain_net2.py` — **deleted** (no Net 2 to pretrain).
- All 6 training scripts in `train_model_v3/{call,put}/train_{vol_surface,spot_history,vix_history}.py` rewritten:
  - `MLP_Predictor` class removed.
  - `bs_normalized_price` + `_norm_cdf` helpers added.
  - `TwoNetworkModel` → `VolEstimatorModel`. Forward: `sigma_hat = net1(branch); price = BS(...); log_v_hat = log(clamp(price, 1e-8))`.
  - Dropped `pretrained_net2_path`, `hidden_dim`, `num_layers` from config + CLI.
  - Optimizer/grad-clip use `model.parameters()`.
- After first user run on `call/train_vol_surface.py`: price metrics great (R² ≈ 0.9998), log metrics bad (R² ≈ 0.02, MSE ≈ 6.5).
- Added a per-sample log-error percentile + clamp-hit diagnostic block to the test-eval section. Confirmed thin-tail pathology: ~3.9% sq.err > 10, ~2.7% hit the `1e-8` clamp.

### 2. v3.1 — per-query σ̂, two pilots on `call/train_vol_surface*.py` only

Both pilots replace the constant-σ̂ head with a function that emits σ̂ given the per-contract query `(log_m, T)` against the market-state encoding. Both:
- Keep the FNO encoder structurally the same (4 SpectralConv2d blocks + skips + flat → 128 → latent), with the **final scalar head removed**.
- Keep the BS pricing step unchanged.
- Use the `1e-8` log clamp.

**Pilot A — Latent + MLP head** → `train_model_v3/call/train_vol_surface.py`
- Encoder produces a 64-d latent.
- `ConditionalVolHead`: `[latent, log_m, T]` → 64 → 64 → 1 (SiLU), softplus output.
- Config gains `latent_dim`, `head_hidden` + CLI `--latent-dim` / `--head-hidden`.
- v3.2 infrastructure ported from the DeepONet pilot: MSE loss (Huber commented), filtered R²(log, T>1day), tail diagnostic, `--eval-only` / `--stable-log` flags. **Model architecture unchanged from v3.1.**

**Pilot B — DeepONet (user-authored, then cheap-fixed)** → `train_model_v3/call/train_vol_surface_don.py`
- Branch (FNO): outputs `p_dim=64` basis vector (linear final layer).
- Trunk (MLP): `(log_m, T)` → `p_dim=64` basis vector (linear final layer).
- Readout: `σ̂ = softplus(⟨branch, trunk⟩ + sigma_bias) + sigma_floor`.
- Cheap-fix applied: `sigma_bias = nn.Parameter(torch.tensor([-1.50]))` (was `torch.zeros(1)`) so that softplus at init gives σ̂ ≈ 0.20, near typical equity IV, avoiding wasted early epochs. See lines ~198–203 in `train_vol_surface_don.py`.
- CLI: `--p-dim`, `--trunk-hidden`, `--sigma-floor`.

### 2b. v3.1 pilot results (this session)

Both pilots ran end-to-end. They land at essentially the same numbers, ~R²(log) 0.89 / R²(price) 0.9998.

**Pilot B (DeepONet) — best run (after user fine-tuned `p_dim=128`, `trunk_hidden=128`, deeper trunk):**

| Metric | Value |
|---|---|
| Test MSE (log)        | 0.6974 |
| Test RMSE (log)       | 0.8351 |
| Test MSE (price)      | 9.59e-5 |
| Test RMSE (price)     | 0.00980 |
| Test MAE (price)      | 0.000657 |
| R²(log)               | 0.8943 |
| R²(price)             | 0.999854 |
| log-error p50 (sq)    | 0.001 |
| log-error p90         | 0.085 |
| log-error p99         | 36.59 |
| log-error p99.9       | 46.58 |
| log-error p100        | 148.25 |
| sq.err > 10           | 14,191 / 905,029 (1.57%) |
| clamp hits            | 14,128 (effectively the same set) |

**Pilot A (MLP-head)** landed at the same R² (~0.892 / 0.999845) on the v3.1 architecture.

### 2c. Tail diagnostic + the `--stable-log` experiment

Added two read-only debug tools to `train_vol_surface_don.py`:
- A **tail diagnostic block** in the test-eval that characterizes the `sq.err > 10` rows: how much of total log-SSE they carry, their target/pred ranges, fraction that hit the clamp, and the `log_m` / `T_years` ranges they cover.
- An `--eval-only` flag that skips training, loads the existing `best_model.pth`, runs the test eval + diagnostic, and **does not overwrite** any training artifacts.
- A `--stable-log` flag that swaps the pricing forward to `log_ndtr` + `log(-expm1(...))` (no `clamp(price, 1e-8)` floor). Defaults off. The unsafe-training caveat is in the docstring.

Diagnostic findings on the best DeepONet weights:
- **1.57% of samples carry 93.4% of the log-SSE.** The other 98.43% are essentially perfect (p90 sq-err ≈ 0.085, p50 ≈ 0.001).
- The tail's true `target_v_log` ∈ [−13.1, −6.2] — real, representable prices (~2e-6 to 2e-3 in `V/K`). **Zero** targets below the 1e-8 floor.
- The tail's `T_years` median is ~0 (near-expiry), `log_moneyness` ~−0.02 (near-ATM).
- 99.6% of the tail predictions are clamp-hits (pred ≤ −18.42), mean signed log-error ≈ −6.4 (model predicts ~6 nats *lower* than truth).

`--stable-log` eval on the same weights: tail predictions came out as `pred_v_log ≈ −4e7`. That is **not** a numerical bug — `log_ndtr` faithfully reported that the model's `sigma_hat` produces an analytic BS price of essentially exact zero on that cluster. The clamp was masking a real model error, not creating one. So:

- Not numerics. Not noise. The model's `sigma_hat` is genuinely wrong on this cluster.
- It's wrong because (a) training never gave gradient there (clamp dead-zone) and (b) the inverse problem `log(price) → σ` is ill-conditioned as T→0 — `∂ log(price)/∂σ` blows up, so even with gradient, log-perfection is essentially unattainable.
- This explains why R²(price) = 0.9998 is untouched — the absolute price error on the tail is ~1e-6, economically meaningless.

### 2d. v3.2 changes to `train_vol_surface_don.py` (this session, kept)

1. **Training loss = MSE (current active line).** Huber (smooth-L1, `delta=1.0`) is kept commented out in `compute_loss` for easy switch-back. See "Huber experiment" below for why MSE is the canonical setup again.
2. **Near-expiry-filtered R²(log)** added to the eval block (and `loss_history.txt`), reporting `R²(log, T > 1/365)` alongside the full-domain value. Threshold = ~1 trading day. **This is the headline log metric** going forward.
3. **Tail diagnostic block** in the test-eval that characterizes the `sq.err > 10` rows (count, share of SSE, target/pred ranges, fraction clamp-hit, log_m / T_years ranges, plus reference target percentiles).
4. `--eval-only` and `--stable-log` flags retained (default off). `--stable-log` is for diagnostic use only — not safe for training (the `1/(-expm1(diff))` gradient blows up near ATM and global `clip_grad_norm_` then crushes the rest of the batch).

What is **not** changed: validation / early-stopping uses MSE inside `evaluate()` (matches the training loss).

### 2da. Huber experiment (tried, reverted)

Switched `compute_loss` to `F.huber_loss(..., delta=1.0)`, retrained end-to-end. Result:

| Metric | MSE-trained | Huber-trained |
|---|---|---|
| R²(log)                | 0.8943 | 0.8942 |
| R²(log, T>1day)        | 0.9924 *(also under MSE)* | 0.9924 |
| R²(price)              | 0.999854 | 0.999593 |
| Test MSE (price)       | 9.59e-5 | 2.68e-4 |
| Test MSE (log, T>1day) | — | 0.04284 |

Huber stopped earlier (smaller gradient magnitudes triggered the patience-based stop sooner), slightly hurt R²(price) and price MSE, and did not move R²(log) — that metric is computed on raw squared errors, so the tail still counts fully in the score; Huber only modulated *training* gradients. **Reverted to MSE.** The Huber line stays commented in `compute_loss` so it's a one-line toggle if a future dataset has heavier tails.

### 2e. False starts in this session (reverted)

- v3.2 MLP-head pilot: tried stable-log forward + linear final latent + σ̂ init in one go. NaN'd, then on a fix attempt landed at R²(log) = −7.8 (gradient explosion via global grad-clip rescaling the whole batch when the near-ATM `1/(-expm1(diff))` term spiked). **All edits reverted**; `train_vol_surface.py` is back to the v3.1 pilot state.
- v3.2 DeepONet readout MLP: replaced the rank-1 dot product with `[b·t, b, t] → MLP → 1`. Slightly worse than rank-1 in user testing. **Reverted.**
- Lesson: stable-log is a valid eval tool but unsafe to drop into training without per-sample gradient handling.

### 3. Files untouched at v3.1 stage (still on constant-σ̂ baseline)

These are the 5 scripts that need the per-query architecture once we pick a winner:
- `train_model_v3/call/train_spot_history.py`
- `train_model_v3/call/train_vix_history.py`
- `train_model_v3/put/train_vol_surface.py`
- `train_model_v3/put/train_spot_history.py`
- `train_model_v3/put/train_vix_history.py`

User/linter touched several of these recently — changes are limited to clamp value (`1e-12` → `1e-8`), patience bumps, and the diagnostic block. No architectural changes.

## Test results we have so far

Baseline v3 (single σ̂, `call/train_vol_surface.py`, 44 epochs total before early stop):

| Metric | Value |
|---|---|
| Test MSE (log) | 1.8855 |
| Test RMSE (log) | 1.373 |
| Test MSE (price) | 1.24e-4 |
| Test RMSE (price) | 0.01116 |
| Test MAE (price) | 0.00266 |
| R²(log) | 0.7143 |
| R²(price) | 0.9998 |
| log-error p50 (sq) | 0.0234 |
| log-error p90 | 1.999 |
| log-error p99 | 45.43 |
| log-error p100 | 148.25 |
| sq.err > 10 | 35,469 / 905,029 (3.9%) |
| clamp hits (pred ≤ log(1.0001e-8)) | 24,287 / 905,029 (2.7%) |

DeepONet pilot is the canonical v3.2 setup: MSE training + filtered eval + diagnostic flags. Best weights = MSE-trained `best_model.pth` in `results_vol_surface_don/` (R²(price) 0.999854, R²(log, T>1day) 0.9924).

## Next steps (resume here)

1. **Call the call/vol_surface pilots done.** Both now have identical v3.2 eval infrastructure. Reported metrics on DeepONet: `R²(price) = 0.9998` and `R²(log, T > 1/365) = 0.9924`. The MLP-head variant should hit similar numbers.
2. **Re-run the MLP-head pilot** `train_model_v3/call/train_vol_surface.py` with a full train and capture the filtered-R². Compare against the DeepONet baseline above.
3. **Pick Pilot A (MLP-head) or Pilot B (DeepONet)** for propagation. Both were within noise on v3.1; v3.2 gave DeepONet a deeper trunk (`p_dim=128`, `trunk_hidden=128`) so rerun MLP-head with `--latent-dim 128 --head-hidden 128` for a fair comparison. Discriminator: filtered R²(log), throughput, or code simplicity.
4. **Propagate the winning architecture** to the 5 untouched scripts:
   - `train_model_v3/call/train_spot_history.py`
   - `train_model_v3/call/train_vix_history.py`
   - `train_model_v3/put/train_vol_surface.py`
   - `train_model_v3/put/train_spot_history.py`
   - `train_model_v3/put/train_vix_history.py`

   The diff per file is mechanical: swap encoder + head/trunk per chosen architecture, port `bs_log_normalized_price`, port flags + tail diagnostic + filtered R² in eval, port the MSE-with-Huber-commented `compute_loss`.
   - Train longer: bump `patience_adam` / `patience_finetune` and rerun under MSE. Earlier runs may have been patience-limited.
   - Filter or down-weight short-T contracts during training (e.g. exclude `T < 1/365`). Explicit, defensible, consistent with the filtered eval.
   - Stable-log *training* with per-sample grad clipping or a soft `diff`-clamp (replacing global `clip_grad_norm_(…, 1.0)`). High-risk; tail's absolute price errors are ~1e-6, so probably chasing economically negligible accuracy.
   - Add CLI args for `latent_dim` / `head_hidden` in the MLP-head version.

## How to resume / re-run

Both call/vol_surface scripts now share the same eval infrastructure (`--eval-only`, `--stable-log`, tail diagnostic, filtered R²(log, T>1day), MSE-with-Huber-commented). They differ only in model architecture.

- **Eval only (DeepONet):**
  ```
  python train_model_v3/call/train_vol_surface_don.py --eval-only
  ```
- **Eval only (MLP-head):**
  ```
  python train_model_v3/call/train_vol_surface.py --eval-only
  ```
- **Diagnostic eval with stable-log forward** (read-only; expect huge log errors on the near-expiry tail):
  ```
  python train_model_v3/call/train_vol_surface.py --eval-only --stable-log
  ```
- **Switch back to Huber** in either script: uncomment the `F.huber_loss` line in `compute_loss` and comment the `F.mse_loss` line.
- **Fresh training (MSE):**
  ```
  python train_model_v3/call/train_vol_surface.py          # MLP-head
  python train_model_v3/call/train_vol_surface_don.py      # DeepONet
  ```
- **Override hyperparams:** add flags like `--latent-dim 128 --head-hidden 128` (MLP-head) or `--p-dim 128 --trunk-hidden 128` (DeepONet).

## Plan file status

`plan_phase2_v3.md` documents the v3 design (delete Network 2, use analytical BS). It does **not** yet describe v3.1 (per-query σ̂) or v3.2 (Huber loss, filtered eval, `--eval-only` / `--stable-log` flags). Update it once the architecture is decided. The pre-rewrite v3 description still applies to the BS-formula choice and the diagnostic block; only the "single σ̂ per market state" part is superseded.

## File map at end of session

```
train_model_v3/
  call/
    pretrain_net2.py                 [DELETED]
    train_vol_surface.py             [v3.2 — MLP-head + MSE (Huber commented) + filtered eval + --eval-only/--stable-log]
    train_vol_surface_don.py         [v3.2 — DeepONet + MSE (Huber commented) + filtered eval + --eval-only/--stable-log]
    train_spot_history.py            [v3 baseline — constant σ̂]
    train_vix_history.py             [v3 baseline — constant σ̂]
    results_vol_surface_don/         [DeepONet best_model.pth = MSE-trained; R²(price)=0.999854, R²(log,T>1day)=0.9924]
    results_*/                       [v3 baseline run artifacts]
  put/
    pretrain_net2.py                 [DELETED]
    train_vol_surface.py             [v3 baseline — constant σ̂]
    train_spot_history.py            [v3 baseline — constant σ̂]
    train_vix_history.py             [v3 baseline — constant σ̂]
    results_*/                       [v3 baseline run artifacts]
plan_phase2_v3.md                    [v3 design; needs v3.1/v3.2 update]
PROGRESS_phase2_v3.md                [this file]
```
