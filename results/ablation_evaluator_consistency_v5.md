# Scalar-sigma ablation: evaluator-consistency diagnostic

Generated 2026-09-07 19:11:39. git HEAD `394bec038d66`. torch 2.10.0+cu130 (cuda 13.0, cuDNN 91200) on NVIDIA GeForce RTX 3060.

No training, no optimizer step, no checkpoint written. Every path runs the EXISTING `best_model.pth` under `torch.no_grad()` with `model.eval()`, after hashing it against `results/ablation_scalar_sigma_v5.json`. **`results/ablation_scalar_sigma_v5.*` is unchanged and the CPU evaluator (`cpu_bs4096`) remains canonical.**

| path | device | batch | pipeline |
|---|---|---|---|
| `cpu_bs512` | CPU | 512 | `analysis/_common.model_predict_logv` |
| `cpu_bs4096` | CPU | 4096 | `analysis/_common.model_predict_logv` (canonical evaluator) |
| `gpu_bs512` | GPU | 512 | the training script's own `H5BranchDataset` / `make_loader` / model class -- the path that wrote the `loss_history.txt` tail |

The August 2026 training environment was never snapshotted, so `gpu_bs512` reproduces the training-time *code path* on today's GPU, driver and CUDA runtime, not the original hardware stack.

## call / scalar_sigma

819,342 test rows, checkpoint `train_model_v3\call\results_vol_surface_scalarsigma_v5\best_model.pth` (`960fae1f4f14...`).

Training-time tail (`loss_history.txt`, for reference): R2(log) 0.800342, RMSE(log) 1.047691, R2(price) 0.999825.

| path | R2(log) | RMSE(log) | R2(price) | R2(log,T>1d) | rows at clamp | s |
|---|---|---|---|---|---|---|
| `cpu_bs512` | 0.800106 | 1.048310 | 0.999825 | 0.800106 | 7,306 | 149 |
| `cpu_bs4096` | 0.800103 | 1.048319 | 0.999825 | 0.800103 | 7,307 | 153 |
| `gpu_bs512` | 0.800342 | 1.047691 | 0.999825 | 0.800342 | 7,297 | 30 |

Same sigma_hat, re-decoded at higher precision (this is the isolation step -- the model output is held fixed and only the decoder changes):

| path | decoder | R2(log) | RMSE(log) | R2(price) |
|---|---|---|---|---|
| `cpu_bs512` | float32 naive (as produced) | 0.800106 | 1.048310 | 0.999825 |
| `cpu_bs512` | float64 naive | 0.798491 | 1.052537 | 0.999825 |
| `cpu_bs512` | float64 stable log_ndtr | -6.034411 | 6.218761 | 0.999825 |
| `cpu_bs4096` | float32 naive (as produced) | 0.800103 | 1.048319 | 0.999825 |
| `cpu_bs4096` | float64 naive | 0.798491 | 1.052537 | 0.999825 |
| `cpu_bs4096` | float64 stable log_ndtr | -6.034411 | 6.218761 | 0.999825 |
| `gpu_bs512` | float32 naive (as produced) | 0.800342 | 1.047691 | 0.999825 |
| `gpu_bs512` | float64 naive | 0.798491 | 1.052537 | 0.999825 |
| `gpu_bs512` | float64 stable log_ndtr | -6.034407 | 6.218759 | 0.999825 |

The float64 `log_ndtr` decoder is reported alongside the other two but it is **not** a drop-in replacement here: it has no 1e-8 floor, so wherever the scalar-sigma model prices a contract far below 1e-8 it returns the true (very negative) log-price instead of -18.42, and the log-space metric moves accordingly. That is the floor being removed, not a defect in either evaluator, and it is why the stable-decoder R2(log) row must not be read as "the correct number".

Pairwise path comparison:

| pair | max Δσ̂ | p99 Δσ̂ | max Δlog-price | rows Δlog-price > 0.1 | ΔR2(log) native | ΔR2(log) float64 | ΔR2(log) stable | ΔR2(price) native |
|---|---|---|---|---|---|---|---|---|
| `cpu_bs512` vs `cpu_bs4096` | 7.45e-08 | 1.49e-08 | 1.19 | 1 (0.0001%) | 3.58e-06 | 3.94e-11 | 4.29e-09 | -2.97e-13 |
| `cpu_bs512` vs `gpu_bs512` | 1.19e-07 | 1.04e-07 | 1.38 | 3,267 (0.3987%) | -0.000236 | -1.35e-09 | -4.51e-06 | -1.05e-10 |
| `cpu_bs4096` vs `gpu_bs512` | 1.19e-07 | 1.04e-07 | 1.38 | 3,268 (0.3989%) | -0.000239 | -1.39e-09 | -4.51e-06 | -1.04e-10 |

- `cpu_bs512__vs__cpu_bs4096`: the 1 moved rows have log-moneyness in [-0.062, -0.062] (median -0.062), maturity 4.00-4.00 days (median 4.00), true log-price in [-11.11, -11.11]; 0 have a true price below the 1e-8 clamp, 1 sit on the clamp on one path and off it on the other.
- `cpu_bs512__vs__gpu_bs512`: the 3,267 moved rows have log-moneyness in [-0.799, -0.031] (median -0.088), maturity 1.73-490.73 days (median 6.73), true log-price in [-13.08, -9.11]; 0 have a true price below the 1e-8 clamp, 1177 sit on the clamp on one path and off it on the other.
- `cpu_bs4096__vs__gpu_bs512`: the 3,268 moved rows have log-moneyness in [-0.799, -0.031] (median -0.088), maturity 1.73-490.73 days (median 6.73), true log-price in [-13.08, -9.11]; 0 have a true price below the 1e-8 clamp, 1178 sit on the clamp on one path and off it on the other.

## put / scalar_sigma

1,380,379 test rows, checkpoint `train_model_v3\put\results_vol_surface_scalarsigma_v5\best_model.pth` (`ef4510e40d5a...`).

Training-time tail (`loss_history.txt`, for reference): R2(log) -0.313834, RMSE(log) 2.267795, R2(price) -3.090190.

| path | R2(log) | RMSE(log) | R2(price) | R2(log,T>1d) | rows at clamp | s |
|---|---|---|---|---|---|---|
| `cpu_bs512` | -0.316871 | 2.270415 | -3.090190 | -0.316871 | 26,893 | 258 |
| `cpu_bs4096` | -0.316871 | 2.270415 | -3.090190 | -0.316871 | 26,893 | 261 |
| `gpu_bs512` | -0.313834 | 2.267795 | -3.090190 | -0.313834 | 26,628 | 46 |

Same sigma_hat, re-decoded at higher precision (this is the isolation step -- the model output is held fixed and only the decoder changes):

| path | decoder | R2(log) | RMSE(log) | R2(price) |
|---|---|---|---|---|
| `cpu_bs512` | float32 naive (as produced) | -0.316871 | 2.270415 | -3.090190 |
| `cpu_bs512` | float64 naive | -0.327938 | 2.279935 | -3.090190 |
| `cpu_bs512` | float64 stable log_ndtr | -32.775219 | 11.498273 | -3.090190 |
| `cpu_bs4096` | float32 naive (as produced) | -0.316871 | 2.270415 | -3.090190 |
| `cpu_bs4096` | float64 naive | -0.327938 | 2.279935 | -3.090190 |
| `cpu_bs4096` | float64 stable log_ndtr | -32.775219 | 11.498273 | -3.090190 |
| `gpu_bs512` | float32 naive (as produced) | -0.313834 | 2.267795 | -3.090190 |
| `gpu_bs512` | float64 naive | -0.327938 | 2.279935 | -3.090190 |
| `gpu_bs512` | float64 stable log_ndtr | -32.775219 | 11.498273 | -3.090190 |

The float64 `log_ndtr` decoder is reported alongside the other two but it is **not** a drop-in replacement here: it has no 1e-8 floor, so wherever the scalar-sigma model prices a contract far below 1e-8 it returns the true (very negative) log-price instead of -18.42, and the log-space metric moves accordingly. That is the floor being removed, not a defect in either evaluator, and it is why the stable-decoder R2(log) row must not be read as "the correct number".

Pairwise path comparison:

| pair | max Δσ̂ | p99 Δσ̂ | max Δlog-price | rows Δlog-price > 0.1 | ΔR2(log) native | ΔR2(log) float64 | ΔR2(log) stable | ΔR2(price) native |
|---|---|---|---|---|---|---|---|---|
| `cpu_bs512` vs `cpu_bs4096` | 1.19e-07 | 0 | 0.00874 | 0 (0.0000%) | 7.36e-09 | -5.68e-11 | 7.72e-08 | -3.35e-09 |
| `cpu_bs512` vs `gpu_bs512` | 3.58e-07 | 1.79e-07 | 4.15 | 8,921 (0.6463%) | -0.00304 | 4.31e-09 | -5.39e-08 | -3.34e-07 |
| `cpu_bs4096` vs `gpu_bs512` | 3.58e-07 | 1.79e-07 | 4.15 | 8,921 (0.6463%) | -0.00304 | 4.36e-09 | -1.31e-07 | -3.3e-07 |

- `cpu_bs512__vs__cpu_bs4096`: no row moved by more than 0.1 in log-price.
- `cpu_bs512__vs__gpu_bs512`: the 8,921 moved rows have log-moneyness in [0.105, 3.401] (median 0.430), maturity 1.73-840.73 days (median 14.73), true log-price in [-12.29, -6.21]; 0 have a true price below the 1e-8 clamp, 2573 sit on the clamp on one path and off it on the other.
- `cpu_bs4096__vs__gpu_bs512`: the 8,921 moved rows have log-moneyness in [0.105, 3.401] (median 0.430), maturity 1.73-840.73 days (median 14.73), true log-price in [-12.29, -6.21]; 0 have a true price below the 1e-8 clamp, 2573 sit on the clamp on one path and off it on the other.

## call / per_query

819,342 test rows, checkpoint `train_model_v3\call\results_vol_surface_v5\best_model.pth` (`82e4f7b101ff...`).

| path | R2(log) | RMSE(log) | R2(price) | R2(log,T>1d) | rows at clamp | s |
|---|---|---|---|---|---|---|
| `cpu_bs4096` | 0.991044 | 0.221896 | 0.999977 | 0.991044 | 0 | 155 |
| `gpu_bs512` | 0.991044 | 0.221894 | 0.999977 | 0.991044 | 0 | 27 |

Same sigma_hat, re-decoded at higher precision (this is the isolation step -- the model output is held fixed and only the decoder changes):

| path | decoder | R2(log) | RMSE(log) | R2(price) |
|---|---|---|---|---|
| `cpu_bs4096` | float32 naive (as produced) | 0.991044 | 0.221896 | 0.999977 |
| `cpu_bs4096` | float64 naive | 0.991044 | 0.221895 | 0.999977 |
| `cpu_bs4096` | float64 stable log_ndtr | 0.991044 | 0.221895 | 0.999977 |
| `gpu_bs512` | float32 naive (as produced) | 0.991044 | 0.221894 | 0.999977 |
| `gpu_bs512` | float64 naive | 0.991044 | 0.221895 | 0.999977 |
| `gpu_bs512` | float64 stable log_ndtr | 0.991044 | 0.221895 | 0.999977 |

The float64 `log_ndtr` decoder is reported alongside the other two but it is **not** a drop-in replacement here: it has no 1e-8 floor, so wherever the scalar-sigma model prices a contract far below 1e-8 it returns the true (very negative) log-price instead of -18.42, and the log-space metric moves accordingly. That is the floor being removed, not a defect in either evaluator, and it is why the stable-decoder R2(log) row must not be read as "the correct number".

Pairwise path comparison:

| pair | max Δσ̂ | p99 Δσ̂ | max Δlog-price | rows Δlog-price > 0.1 | ΔR2(log) native | ΔR2(log) float64 | ΔR2(log) stable | ΔR2(price) native |
|---|---|---|---|---|---|---|---|---|
| `cpu_bs4096` vs `gpu_bs512` | 4.23e-06 | 4.02e-07 | 0.0178 | 0 (0.0000%) | -1.2e-07 | 2.16e-09 | 2.16e-09 | -1.05e-10 |

- `cpu_bs4096__vs__gpu_bs512`: no row moved by more than 0.1 in log-price.

## put / per_query

1,380,379 test rows, checkpoint `train_model_v3\put\results_vol_surface_v5\best_model.pth` (`9c87fa602de6...`).

| path | R2(log) | RMSE(log) | R2(price) | R2(log,T>1d) | rows at clamp | s |
|---|---|---|---|---|---|---|
| `cpu_bs4096` | 0.977899 | 0.294127 | 0.992553 | 0.977899 | 0 | 258 |
| `gpu_bs512` | 0.977898 | 0.294136 | 0.992553 | 0.977898 | 0 | 48 |

Same sigma_hat, re-decoded at higher precision (this is the isolation step -- the model output is held fixed and only the decoder changes):

| path | decoder | R2(log) | RMSE(log) | R2(price) |
|---|---|---|---|---|
| `cpu_bs4096` | float32 naive (as produced) | 0.977899 | 0.294127 | 0.992553 |
| `cpu_bs4096` | float64 naive | 0.977900 | 0.294127 | 0.992553 |
| `cpu_bs4096` | float64 stable log_ndtr | 0.977900 | 0.294127 | 0.992553 |
| `gpu_bs512` | float32 naive (as produced) | 0.977898 | 0.294136 | 0.992553 |
| `gpu_bs512` | float64 naive | 0.977900 | 0.294127 | 0.992553 |
| `gpu_bs512` | float64 stable log_ndtr | 0.977900 | 0.294127 | 0.992553 |

The float64 `log_ndtr` decoder is reported alongside the other two but it is **not** a drop-in replacement here: it has no 1e-8 floor, so wherever the scalar-sigma model prices a contract far below 1e-8 it returns the true (very negative) log-price instead of -18.42, and the log-space metric moves accordingly. That is the floor being removed, not a defect in either evaluator, and it is why the stable-decoder R2(log) row must not be read as "the correct number".

Pairwise path comparison:

| pair | max Δσ̂ | p99 Δσ̂ | max Δlog-price | rows Δlog-price > 0.1 | ΔR2(log) native | ΔR2(log) float64 | ΔR2(log) stable | ΔR2(price) native |
|---|---|---|---|---|---|---|---|---|
| `cpu_bs4096` vs `gpu_bs512` | 3.81e-06 | 1.79e-07 | 0.0206 | 0 (0.0000%) | 1.36e-06 | -7.84e-09 | -7.84e-09 | -1.89e-09 |

- `cpu_bs4096__vs__gpu_bs512`: no row moved by more than 0.1 in log-price.

## Conclusion

**Conclusion: no implementation error was found -- the discrepancy is a device / float32-precision path difference on rows the model prices at the 1e-8 clamp floor.** Batch size alone is not the cause: holding the device fixed and moving from batch 512 to batch 4096 on the CPU changes R2(log) by 3.6e-06. Changing the DEVICE changes it by 0.00304, which is the size of the published discrepancy. The model output is not what differs -- sigma_hat agrees between CPU and GPU to 3.58e-07 in the worst row and 1.79e-07 at the 99th percentile, i.e. to float32 rounding. What differs is the decoder. For calls, 3,268 of 819,342 test rows (0.3989%) decode to a log-price that differs by more than 0.1 between the canonical CPU evaluator and the GPU training path, by up to 1.38; 1,178 of them sit on the 1e-8 clamp on one path and off it on the other, and the per-path clamp-hit count itself differs (7,306 vs 7,307 vs 7,297). Those rows are deep out-of-the-money and short-dated (median maturity 6.7 days, median log-moneyness -0.088) -- exactly where M e^{-qT} N(d1) - e^{-rT} N(d2) is a difference of two nearly equal float32 numbers. For puts, 8,921 of 1,380,379 test rows (0.6463%) decode to a log-price that differs by more than 0.1 between the canonical CPU evaluator and the GPU training path, by up to 4.15; 2,573 of them sit on the 1e-8 clamp on one path and off it on the other, and the per-path clamp-hit count itself differs (26,893 vs 26,893 vs 26,628). Those rows are deep out-of-the-money and short-dated (median maturity 14.7 days, median log-moneyness 0.430) -- exactly where M e^{-qT} N(d1) - e^{-rT} N(d2) is a difference of two nearly equal float32 numbers. Holding each path's own float32 sigma_hat fixed and re-decoding it in float64 with the same formula collapses the R2(log) difference from 0.00304 to 4.36e-09, so essentially all of the gap is float32 arithmetic in the decoder rather than a different prediction. R2(price) never differs by more than 3.34e-07 on any comparison, because a row worth 1e-8 contributes nothing in price space. The per-query checkpoints, run through the same two paths as a positive control, move by at most 1.4e-06 in R2(log) -- 2230x smaller -- and hit the clamp floor on 0 rows, which is the whole difference: the per-query model does not price contracts at zero. The float64 `log_ndtr` decoder is reported alongside the other two but it is **not** a drop-in replacement here: it has no 1e-8 floor, so wherever the scalar-sigma model prices a contract far below 1e-8 it returns the true (very negative) log-price instead of -18.42, and the log-space metric moves accordingly. That is the floor being removed, not a defect in either evaluator, and it is why the stable-decoder R2(log) row must not be read as "the correct number". The practical consequence: the scalar-sigma log-space metrics are reproducible across devices only to about 3e-03 in R2, so the six-decimal figures in results/ablation_scalar_sigma_v5.* should be read as ~3e-03-precise; the price-space figures, which agree to 3e-07, carry their printed precision. The CPU evaluator at batch 4096 remains canonical and nothing in results/ablation_scalar_sigma_v5.* was changed.
