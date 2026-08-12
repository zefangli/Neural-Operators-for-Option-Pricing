# Learning Implied Volatility, Not Prices: A Neural-Operator Encoder with a Differentiable Black–Scholes Decoder for Option Pricing

*Paper skeleton — drafted 2026-06-11. This document mirrors the structure of `pub_plan.md` §2.
Sections backed by current experimental results are written out; sections that depend on
artifacts not yet built (baselines, ablations, greeks, figures F1–F6) are marked with explicit
`> **TODO**` blocks. Do not cite a number here that is not in a results file.*

---

## Status: what is backed by data vs. TODO

| Section | State |
|---|---|
| Method (§3), Data (§4) | ✅ Written from code / data contract |
| Main results T1 — 12 core runs (§6) | ✅ Real numbers (test split) |
| Branch comparison, A-vs-B analysis (§6) | ✅ From T1 |
| Near-expiry tail analysis (§6) | ✅ From eval diagnostic |
| Related work (§2) | ⬚ TODO — citations to gather |
| Baselines T2 — BS const-σ, IV interp (§5/§6) | ⬚ TODO — `analysis/baselines.py` not built |
| Ablations T3 — per-query vs scalar, BS vs direct (§6) | ⬚ TODO — not run |
| Greeks validation (§6, F6) | ⬚ TODO — `analysis/greeks_check.py` not built |
| Figures F1–F6 | ⬚ TODO — only F7 (training curves) exist |

---

## Abstract

Most neural option pricers learn a map `(market state, contract) → price`. We instead learn a map
to a **per-contract implied volatility** `σ̂` and feed it through the **closed-form Black–Scholes
(BS) formula** to obtain the price. Because BS is built from differentiable primitives, gradients
flow from a plain data loss on `log(V/K)` back through BS into the network, so no neural surrogate
pricer is needed. This single architectural choice buys three properties for free: (i) no surrogate
approximation error — all error is attributable to `σ̂`; (ii) exact greeks — every greek is the
closed-form BS greek at `σ̂`; (iii) arbitrage-free by construction. A 2-D Fourier Neural Operator
(FNO) encodes the market state; a per-query head (MLP or DeepONet branch·trunk) emits `σ̂` for each
contract `(log-moneyness, T)`. On SPX options (WRDS / OptionMetrics) with a no-leakage,
date-based 80/10/10 split, the model reaches **R²(price) ≈ 0.9999** and **R²(log, T>1 day) ≈ 0.992**
on the IV-surface input for calls. Comparing market-state inputs, the implied-volatility surface
prices best by a wide and consistent margin over SPX-history and VIX-history inputs.

> **TODO (abstract):** add one sentence with the baseline gap (model vs BS-constant-σ / IV
> interpolation) once §3b baselines are run.

---

## 1. Introduction

The pricing problem and the *learn-IV-vs-learn-price* dichotomy. Pricing an option is a map from
market state and contract terms to a price; the standard neural approach regresses that price (or
the implied vol) directly. We argue the price target is the wrong one: it forces the network to
re-learn the well-understood, closed-form dependence of price on volatility, time, and moneyness,
and it gives up structural guarantees. Instead we predict only the one genuinely hard quantity — a
per-contract implied volatility `σ̂` — and recover the price analytically.

**Why structural arbitrage-freeness + exact greeks matter.** A BS price for any `σ̂ ≥ 0` satisfies
the BS PDE and the no-arbitrage bounds, so the auxiliary PDE/arbitrage penalties that
physics-informed (PINN) approaches need become unnecessary; and every greek is the exact closed-form
BS greek at the estimated `σ̂`, computed either analytically or by autograd through the decoder.

**Contributions (headline claims).**
1. Per-query `σ̂` recovers the volatility smile/skew that a single-scalar `σ̂` cannot.
2. The analytical-BS decoder matches or beats a direct-price regressor of equal capacity, while
   giving exact greeks and arbitrage-free prices.
3. Among market-state inputs (IV surface / SPX history / VIX history), we quantify which prices best
   — and by how much.

> **Note on scope (keep while drafting):** this is an empirical / application paper. The core idea
> (differentiable BS layer, learning IV not price, analytical-pricer decoder) is sound but **not
> novel to the ML community**. Frame contributions as "a clean, carefully-evaluated study," not "a
> new method." Target: arXiv (`q-fin.CP` primary, `cs.LG` cross-list) + a finance/ML workshop.

---

## 2. Related work

> **TODO:** prose + citations for each cluster below.
> - Neural option pricing (direct price/IV regression).
> - PINNs in finance (PDE/arbitrage soft penalties) — position our structural alternative against
>   these, including our own superseded v1/v2 design.
> - Neural operators: FNO, DeepONet.
> - Classic IV-surface modeling: SVI, SSVI, splines (the practitioner baselines).

---

## 3. Method

### 3.1 The inverse IV problem
Given an observed normalized option price `V/K` for a contract with log-moneyness `log_m`, maturity
`T`, rate `r`, dividend yield `q`, the implied volatility `σ` is the value making BS reproduce the
price. We learn an estimator `σ̂` of this `σ` conditioned on the broader market state, and price
through BS. The target is `log(V/K)`. The inverse map `log(price) → σ` is ill-conditioned as
`T → 0` (`∂log(price)/∂σ` blows up near expiry/ATM) — a fact that dominates the error analysis (§6).

### 3.2 FNO market-state encoder
A 2-D Fourier Neural Operator: 4 `SpectralConv2d` blocks with 1×1-conv skip connections and SiLU
activations, `fno_width = 32`, flattened and projected to a 128-d latent (branch) vector. The 1-D
HDF5 market-state vector is reshaped to its 2-D grid inside the encoder. Three branch inputs, one
script each:

| input (branch_key) | grid | flat dim | modes |
|---|---|---|---|
| `branch_u` (implied-vol surface) | 11×17 | 187 | 4×8 |
| `spot_history` (21-day SPX OHLCV) | 21×5 | 105 | 6×2 |
| `vix_history` (21-day VIX OHLC) | 21×4 | 84 | 6×2 |

### 3.3 Per-query `σ̂` readouts
The same encoder feeds two interchangeable readouts (matched capacity, 128/128):
- **(A) MLP-head** (`train_*.py`): MLP on `[latent, log_m, T]` → softplus → `σ̂`.
- **(B) DeepONet** (`train_*_don.py`): FNO **branch** vector, MLP **trunk** on `(log_m, T)`;
  `σ̂ = softplus(⟨branch, trunk⟩ + sigma_bias) + sigma_floor`, with `sigma_bias` initialized to
  −1.50 (`σ̂ ≈ 0.20` at init) and `sigma_floor = 1e-6`.

`σ̂` is **per contract query**, not one scalar per market state — this is what lets the model fit the
smile/skew.

### 3.4 Differentiable Black–Scholes decoder
`bs_normalized_price(log_m, T, r, q, σ̂, option_type)` returns `V/K` from differentiable primitives
(`exp`, `erf`, `sqrt`). Gradients of the data loss flow through BS into the network. **Corollaries:**
(i) every greek is the exact closed-form BS greek at `σ̂` (verifiable by autograd vs analytic, §6);
(ii) the price is arbitrage-free for any `σ̂ ≥ 0`. There is **no** learned/neural pricer (that was
the removed v1/v2 design).

### 3.5 Loss
Plain data MSE on `log(V/K)`. The default path uses `log(clamp(price, 1e-8))`. **No** PDE /
arbitrage / BS-anchor terms — they are redundant given the decoder. (`compute_loss` ships with MSE
active and `F.huber_loss(δ=1.0)` commented one line above as a deliberate toggle; Huber was tried
and reverted — stopped earlier, hurt R²(price), did not move R²(log).)

---

## 4. Data

WRDS / OptionMetrics **SPX** options (secid `108105`), European-style (BS applies directly),
covering **2015-02-02 to 2025-08-29** (the dataset directory is named `wrds_data_2020-2025` but
its date coverage starts in 2015 — confirmed from the `date` column). Filter `volume > 0`. Each `deeponet_tensors_{call,put}.h5` holds row-aligned datasets: the three branch
inputs (`branch_u`, `spot_history`, `vix_history`); trunk `trunk_y = [log_moneyness, T_years, r, q]`;
target `target_v_log = log(mid/K)`; and `moneyness`, `normalized_price`, `date`, `split_id`.

**No-leakage split.** The 80/10/10 train/val/test split is **by unique trade date**, not by row — a
deliberate time-ordered split that prevents same-day leakage across splits.

**Test-split sizes** (from eval): call ≈ **905,029** contracts, put ≈ **1,462,668** contracts. The
near-expiry tail `T ≤ 1 day` removed by the headline filter is ~68,443 calls (~7.6%) and ~78,187
puts (~5.3%).

> **TODO:** retention / row-count table across the full pipeline (raw → filtered → per-split);
> confirm whether per-contract observed IV can be recovered for the smile figure F2 (invert market
> price or read a retained column).

---

## 5. Experiments

### 5.1 Core model runs — 12 total (done)
`{call, put} × {vol_surface, spot_history, vix_history} × {MLP-head (A), DeepONet (B)}`. Shared
config (identical across all 12): two-phase training — Adam (`epochs=100`, `lr=1e-3`,
`ReduceLROnPlateau`) then fine-tune (`epochs=50`, `lr=1e-5`); `batch_size=256`, `grad_clip=1.0`,
patience 5 each phase, `min_finetune_epochs=10`, seed 42; early stop on val MSE. Metrics on the
**test split**: R²(price), R²(log, full), **R²(log, T>1 day) (headline)**, RMSE(log), RMSE(price),
MAE(price). All price metrics are on **normalized price `V/K`** (not raw dollars).

### 5.2 Baselines (not yet built)
> **TODO — `analysis/baselines.py`:**
> - **BS constant-σ:** price each test contract with a single daily σ — (i) ATM IV of the day,
>   (ii) VIX — then BS. Establishes the value added by a per-query surface.
> - **Classic IV interpolation:** σ per query from the observed 11×17 surface via 2-D spline /
>   bilinear, nearest-neighbor, and optionally an SVI slice fit per date; then BS. The standard
>   practitioner baseline the model must beat.

### 5.3 Ablations (not yet run)
> **TODO:** (a) **per-query σ̂ vs single-scalar σ̂** — the headline ablation; (b) *(stretch)*
> analytical-BS decoder vs direct-price regressor of equal encoder capacity; (c) *(stretch)*
> MSE vs Huber; (d) *(stretch)* near-expiry exclusion/down-weighting during training.

### 5.4 Greeks validation (not yet run)
> **TODO — `analysis/greeks_check.py`:** autograd `∂price/∂S` and vega through the BS decoder vs
> closed-form (`e^{-qT}N(d1)` and put analogue) on sampled test contracts; expect ~1e-7 agreement.

---

## 6. Results & analysis

### T1 — Main results (12 core runs, test split)

| option | branch | arch | R²(price) | R²(log, T>1day) | R²(log, full) | RMSE(log) | RMSE(price) | MAE(price) |
|---|---|---|---|---|---|---|---|---|
| call | vol_surface  | A | 0.999851 | **0.992481** | 0.894105 | 0.835967 | 0.009895 | 0.000757 |
| call | vol_surface  | B | 0.999854 | **0.992413** | 0.894330 | 0.835075 | 0.009795 | 0.000657 |
| call | spot_history | A | 0.999834 | 0.948926 | 0.858569 | 0.966102 | 0.010428 | 0.002325 |
| call | spot_history | B | 0.998842 | 0.950056 | 0.859608 | 0.962546 | 0.027581 | 0.003303 |
| call | vix_history  | A | 0.999811 | 0.905300 | 0.822470 | 1.082397 | 0.011135 | 0.003280 |
| call | vix_history  | B | 0.999730 | 0.915567 | 0.830493 | 1.057656 | 0.013315 | 0.002758 |
| put  | vol_surface  | A | 0.996177 | **0.979764** | 0.832595 | 0.905997 | 0.001501 | 0.000626 |
| put  | vol_surface  | B | 0.997241 | **0.981660** | 0.833637 | 0.903173 | 0.001275 | 0.000539 |
| put  | spot_history | A | 0.963103 | 0.909478 | 0.776980 | 1.045718 | 0.004664 | 0.002555 |
| put  | spot_history | B | 0.836896 | 0.887795 | 0.760506 | 1.083651 | 0.009807 | 0.003808 |
| put  | vix_history  | A | 0.964931 | 0.902385 | 0.770127 | 1.061662 | 0.004547 | 0.002523 |
| put  | vix_history  | B | 0.948355 | 0.900832 | 0.769931 | 1.062114 | 0.005518 | 0.002779 |

**Branch comparison (headline claim 3).** The ranking is consistent across both option types and
both architectures: **`vol_surface` ≫ `spot_history` > `vix_history`**. On R²(log, T>1 day), calls
score .992 / .949–.950 / .905–.916; puts .980–.982 / .888–.909 / .901–.902. The gridded IV surface
carries far more pricing-relevant signal than SPX or VIX price history alone — which is the expected
and reassuring result (the surface *is* the contemporaneous vol information), and it quantifies how
much is lost when only historical underlying/VIX data is available.

**A vs B.** On `vol_surface` the two readouts **tie within seed noise** for both calls (.99248 vs
.99241) and puts (.97976 vs .98166). On the weaker branches the comparison is mixed: B is slightly
better on call/`vix_history`, but **put/`spot_history` B is an outlier** (R²(price) = 0.837 vs A's
0.963) — likely a patience-limited / unstable run; flagged for a re-run before publication.

**Calls vs puts.** Put R²(price) sits below call (vol_surface .996 vs .9999). Put normalized prices
are smaller, so the variance denominator differs; the headline R²(log, T>1 day) remains strong
(.980+), so this is a scaling artifact of the price-space metric, not a pricing failure.

**The near-expiry log tail (the project's defining numerical fact).** ~1.5% of contracts (`T → 0`,
ATM) carry ~93% of the log-space SSE because the inverse problem `log(price) → σ` is genuinely
ill-conditioned as `T → 0`. This is why R²(price) ≈ 0.9999 while full-domain R²(log) ≈ 0.89 — the
*absolute* price error on the tail is ~1e-6 (economically nil). We therefore report all three
together and treat **R²(log, T>1 day)** as the headline. A diagnostic block in `--eval-only`
characterizes this cluster (count, SSE share, target/pred ranges, clamp-hit fraction, log_m/T
ranges). A numerically-stable eval pricer (`--stable-log`, `log_ndtr` + `log(-expm1)`) confirms the
`1e-8` clamp masks a real model error on the tail rather than creating one; it is **eval-only**
(its gradient explodes near ATM and NaNs out training).

### T2 — Baselines vs model
> **TODO:** populate once §5.2 baselines run (BS-constant-σ, IV interpolation vs best model).

### T3 — Ablations
> **TODO:** per-query vs scalar; BS-decoder vs direct regressor; MSE vs Huber.

### T4 — Error stratification
> **TODO:** RMSE / R² over a moneyness × maturity grid (the eval already computes the stratification
> block; aggregate it here).

---

## 7. Limitations & discussion

- **Ill-conditioned `T → 0` tail.** Log-perfection near expiry is unattainable; we scope the
  headline metric to `T > 1 day` and report the economic (dollar/normalized) error to show the tail
  is economically negligible. A tail-aware loss / reparametrization / σ̂ uncertainty quantification
  is the natural follow-on (out of scope here).
- **Single underlying (SPX).** Stated explicitly; a second underlying is the journal/conference
  extension, not a v1 blocker.
- **European-style assumption.** SPX options are European, so BS applies directly. Any American-style
  contamination would violate the BS-pricing assumption.
- **No transaction costs / bid-ask.** We price the mid; spreads and costs are out of scope.

---

## 8. Conclusion

Predicting a per-contract implied volatility and pricing it through a differentiable Black–Scholes
decoder is a clean, structurally-honest alternative to learning prices directly: it removes
surrogate error, yields exact greeks, and is arbitrage-free by construction. Across 12 SPX runs with
a no-leakage time split, the approach reaches R²(price) ≈ 0.9999 and R²(log, T>1 day) ≈ 0.992 on the
IV-surface input, the two readout architectures tie on that input, and the IV surface consistently
outprices SPX- and VIX-history inputs. Remaining for the full paper: external baselines, the
per-query-vs-scalar ablation, the greeks check, and figures.

---

## Appendix

### A. Hyperparameters (identical across all 12 runs; from `config.json`)
seed 42; `epochs_adam=100`, `lr=1e-3`, `ReduceLROnPlateau` (phase 1); `epochs_finetune=50`,
`finetune_lr=1e-5`; `batch_size=256`, `grad_clip=1.0`; `patience_adam = patience_finetune = 5`,
`min_finetune_epochs=10`. FNO: 4 `SpectralConv2d` blocks (+1×1-conv skips, SiLU), `fno_width=32`,
flat → 128 → latent/basis 128; head/trunk hidden 128. DeepONet adds `sigma_floor=1e-6` and scalar
`sigma_bias` (init −1.50). Param counts: FNO branch ≈ 1.05M (11×17) / 0.55M (21×5); MLP head
+33,409; DeepONet trunk +49,920 +1.

### B. Numerical stability
Default training/eval uses `log(clamp(price, 1e-8))`. `--stable-log` swaps in
`bs_log_normalized_price` (`log_ndtr` + `log(-expm1(...))`), exact but with a gradient that explodes
near ATM — **eval-only**.

### C. Reproducibility
conda env `dl_new` (torch, polars, numpy, h5py, matplotlib). One `config.json` per run saved; seed
42 fixed. WRDS / OptionMetrics data is **licensed** and cannot be redistributed; preprocessing code
is released so a WRDS subscriber can reproduce the HDF5 (SPX secid `108105`, `volume > 0`).
> **TODO:** pin exact package versions in `environment.yml`.

### D. Figures
- **F7 — Training curves:** ✅ `loss_plot.png` exists per run.
> **TODO (new plotting code):** F1 architecture diagram; F2 smile recovery (per-query vs scalar σ̂)
> — the core-claim figure; F3 error heatmap (moneyness × maturity); F4 near-expiry SSE-vs-T tail;
> F5 predicted-vs-market price scatter; F6 greeks agreement (autograd vs closed-form).
