# Train Model Phase 2 Plan

## Summary

Build six standalone training scripts under `train_model/`: three for calls and three for puts. Each script trains the same two-network architecture but uses a different market-state input for Network 1:

- `branch_u`: implied volatility surface, reshaped to `11 x 17`
- `spot_history`: SPX OHLCV lookback, reshaped to `21 x 5`
- `vix_history`: VIX OHLC lookback, reshaped to `21 x 4`

Network 1 will be an FNO-based market-state-only volatility estimator. Unlike `OLD_Final_model_evaluation/training.py`, it will not take moneyness or time-to-expiration. Network 2 will take `[log_moneyness, T_years, r, q, sigma_hat]` and predict `log(V/K)`. Other parts of the models will keep the same structure and training losses.

## Key Changes

- Create these standalone files:
  - `train_model/call/train_vol_surface.py`
  - `train_model/call/train_spot_history.py`
  - `train_model/call/train_vix_history.py`
  - `train_model/put/train_vol_surface.py`
  - `train_model/put/train_spot_history.py`
  - `train_model/put/train_vix_history.py`
- Each file should:
  - Load the matching HDF5 file from `wrds_data_2020-2025/deeponet_tensors_call.h5` or `wrds_data_2020-2025/deeponet_tensors_put.h5`.
  - Use `split_id` for train/validation/test splits instead of row-index splitting.
  - Select exactly one branch dataset: `branch_u`, `spot_history`, or `vix_history`.
  - Load `trunk_y = [log_moneyness, T_years, r, q]`.
  - Train against `target_v_log`.
  - Not apply per-feature normalization — the preprocessing pipeline already handles that.
- Network 1:
  - Use a 2D FNO similar to `OLD_Final_model_evaluation/training.py`.
  - Reshape the selected branch input to its fixed grid shape.
  - Pass through four FNO blocks, then MLP projection layers.
  - Output a positive scalar `sigma_hat` using `Softplus`.
  - Do not use `log_moneyness`, `moneyness`, or `T_years`.
  - FNO width = 32 (internal channel count, consistent with old code).
  - Fourier mode counts, set per grid:
    - `branch_u` (11×17): `modes1=4`, `modes2=8`
    - `spot_history` (21×5): `modes1=6`, `modes2=2`
    - `vix_history` (21×4): `modes1=6`, `modes2=2`
  - MLP projection after flattening:
    - `Linear(width * H * W, 128)` → `Linear(128, 64)` → `Linear(64, 1)` → `Softplus`
- Network 2:
  - Use an MLP with `hidden_dim=128`, `num_layers=4` (consistent with old code).
  - Input dimension is `5`: `[log_moneyness, T_years, r, q, sigma_hat]`.
  - Output one unconstrained scalar: predicted `log(V/K)`.
  - Convert to price space with `v_hat = exp(log_v_hat)` only for PDE, arbitrage, BS-anchor, and evaluation metrics.

## Losses

- Use data loss on log price:
  - `MSE(log_v_hat, target_v_log)`
- Use Black-Scholes PDE loss in normalized price space:
  - Let `x = log_moneyness`, `M = exp(x)`, and `v = exp(log_v_hat)`.
  - Compute derivatives with autograd (`torch.autograd.grad`, `create_graph=True`) against raw `x` and `T_years`.
  - Convert derivatives:
    - `dv/dM = (dv/dx) / M`
    - `d2v/dM2 = (d2v/dx2 - dv/dx) / M^2`
  - Residual (verified correct against standard BS PDE):
    - `-dv/dT_years + 0.5 * sigma_hat^2 * M^2 * d2v/dM2 + (r - q) * M * dv/dM - r * v`
- Use option-type-specific arbitrage loss:
  - Calls: `F.relu(-dv_dM) + F.relu(-d2v_dM2)` (positive delta, positive gamma)
  - Puts: `F.relu(dv_dM) + F.relu(-d2v_dM2)` (negative delta, positive gamma)
- Include Black-Scholes anchoring loss for Network 2:
  - Sample synthetic inputs from independent uniform distributions over financially meaningful ranges:
    - `log_moneyness ∼ Uniform(-1.5, 1.5)` — covers M ≈ 0.22 to 4.5
    - `T_years ∼ Uniform(1/365, 5.0)` — one day to five years
    - `r ∼ Uniform(0.0, 0.10)` — negative to high rate environments
    - `q ∼ Uniform(0.0, 0.05)` — typical dividend yield range
    - `sigma ∼ Uniform(0.05, 1.00)` — low vol to very high vol
  - Synthetic `sigma` is passed directly to Network 2 as the 5th input (bypassing Network 1), so the anchor loss isolates Network 2's training.
  - Compute analytical BS normalized call/put price for the sampled `(log_moneyness, T_years, r, q, sigma)`.
  - Train Network 2's `log_v_hat` output to match `log(analytical_price)` via MSE.
  - The synthetic batch is independent of the real data batch — no need for collocation pairing.
  - Batch size controlled by config parameter `bs_anchor_batch_size` (default: 256).
- Default loss weights:
  - `data = 1.0`
  - `pde = 0.5`
  - `arb = 0.5`
  - `bs_anchor = 0.5`
  - `terminal/lower/upper boundary = 0.0` (disabled by default, can be re-enabled if needed)

## Training Procedure

Two-stage training, matching the pattern from `OLD_Final_model_evaluation/training.py`:

### Stage 1: Adam

- `epochs_adam = 100` (default; early stopping with `patience_adam = 3`)
- `batch_size = 256`
- `lr = 1e-3`
- `grad_clip = 1.0`
- `ReduceLROnPlateau` scheduler (factor 0.5, patience 2)
- Save `best_model_phase1.pth` when validation loss improves.
- Collocation points not used (PDE and BS-anchor provide the regularization).

### Stage 2: Fine-tuning

- `epochs_finetune = 50` (default; early stopping with `patience_finetune = 3`, minimum 10 epochs)
- `finetune_lr = 1e-5`
- Save `best_model.pth` when validation loss improves.
- At end, save `final_model.pth`.

## Training Outputs

Each script writes to a feature-specific output directory, for example:

- `train_model/call/results_vol_surface/`
- `train_model/call/results_spot_history/`
- `train_model/call/results_vix_history/`

Each run should save:

- `best_model_phase1.pth`
- `best_model.pth`
- `final_model.pth`
- `loss_history.txt`
- `loss_plot.png`
- `config.json`
- final train/validation/test metrics

Primary metrics:

- test MSE on `log(V/K)`
- test RMSE on `log(V/K)`
- price-space MSE/MAE after `exp(log_v_hat)`
- optional `R^2` in log space and price space

## Test Plan

- Static smoke checks:
  - All six files import successfully.
  - Each file resolves the correct HDF5 file path.
  - Each selected branch dataset reshapes to the expected FNO grid.
- Data checks:
  - Confirm selected HDF5 datasets exist.
  - Confirm `split_id` contains only `{0, 1, 2}`.
  - Confirm `trunk_y.shape[1] == 4`.
  - Confirm `target_v_log.shape[1] == 1`.
- Model checks:
  - Run one forward pass for each feature type.
  - Confirm `sigma_hat > 0`.
  - Confirm `log_v_hat` shape is `(batch, 1)`.
  - Confirm PDE and BS-anchor losses are finite.
- Training smoke test:
  - Run each script with a tiny debug setting, such as `--epochs-adam 1 --epochs-finetune 0 --max-train-batches 2`.
  - Confirm checkpoints, logs, and loss plot are produced.

## Running the Code

All six scripts are run from the project root after activating the `dl_new` conda environment:

```bash
conda activate dl_new
cd <project_root>
```

### Training / Evaluation (full run)

Each script loads data, trains, evaluates on the test set, and saves outputs — all in one invocation.

```bash
# --- Calls ---
python train_model/call/train_vol_surface.py
python train_model/call/train_spot_history.py
python train_model/call/train_vix_history.py

# --- Puts ---
python train_model/put/train_vol_surface.py
python train_model/put/train_spot_history.py
python train_model/put/train_vix_history.py
```

Each script writes its results to a feature-specific output directory:

| Script | Output Directory |
|---|---|
| `call/train_vol_surface.py` | `train_model/call/results_vol_surface/` |
| `call/train_spot_history.py` | `train_model/call/results_spot_history/` |
| `call/train_vix_history.py` | `train_model/call/results_vix_history/` |
| `put/train_vol_surface.py` | `train_model/put/results_vol_surface/` |
| `put/train_spot_history.py` | `train_model/put/results_spot_history/` |
| `put/train_vix_history.py` | `train_model/put/results_vix_history/` |

### CLI flags (all six scripts accept the same subset)

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--epochs-adam` | int | 100 | Override Adam stage epochs |
| `--epochs-finetune` | int | 50 | Override fine-tune stage epochs |
| `--max-train-batches` | int | None (all) | Limit batches per epoch (for smoke tests) |
| `--batch-size` | int | 256 | Override batch size |
| `--lr` | float | 1e-3 | Override Adam learning rate |
| `--finetune-lr` | float | 1e-5 | Override fine-tune learning rate |
| `--seed` | int | 42 | Override random seed |

### Smoke test

Run a quick 2-batch sanity check to verify the script compiles, loads data, runs one forward/backward pass, and saves outputs:

```bash
python train_model/call/train_vol_surface.py --epochs-adam 1 --epochs-finetune 0 --max-train-batches 2
```

Check that `train_model/call/results_vol_surface/` contains: `config.json`, `best_model_phase1.pth`, `best_model.pth`, `final_model.pth`, `loss_history.txt`, `loss_plot.png`.

### Monitoring during training

Each epoch prints loss component breakdowns and validation MSE. The `loss_plot.png` in the results directory shows the training curve (log scale) with a vertical dashed line marking the transition from Adam to fine-tuning. For real-time monitoring, watch the terminal output for early stopping messages (triggered after 3 epochs without validation improvement).

## Assumptions

- The implementation should keep the six training files standalone, with duplicated architecture/loss/training code rather than shared utility modules.
- SPX and VIX history inputs should use 2D FNO grids, not a 1D temporal FNO.
- Boundary condition losses remain disabled by default, following the best result discussed in `final_report.tex`.
- The target remains `log(V/K)`, while PDE and arbitrage constraints are evaluated on `V/K = exp(log_v_hat)`.
- Calls and puts are trained separately and are not combined into one model.
- No per-feature normalization is applied in the training scripts — the preprocessing pipeline already normalizes `spot_history` and `vix_history` by each series' final-day value, and `log_moneyness` is already a log-ratio.

