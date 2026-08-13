# Phase 2 v3 — Progress Snapshot

Last updated: 2026-08-13 (v5 dataset build, validation, and all three CPU analyses COMPLETE — v5 is
now the canonical dataset). This file describes the **current state of `train_model_v3/`** only.

> **2026-08-13 — v5 build, validation, and all three CPU analyses are COMPLETE. This supersedes every
> "build in progress" / "not yet built" / "provisional" / "approximate" qualifier attached to v5 in
> the note below and elsewhere in this file.** The v5 HDF5 build finished
> (`wrds_data_2020-2025/deeponet_tensors_{call,put}_v5.h5`) and passed **45/45 validation checks**
> (`VALIDATION_v5.json`: the 31 original v4-style checks plus 7 new v5-specific checks, ×2 option
> types), with **zero remaining static-bound violations, max gap 0.000e+00**. Final row counts
> (confirming the approximate float32-based percentages in the note below to within ~0.001 pp):
>
> | | Call | Put |
> |---|---:|---:|
> | Total rows | 4,846,268 | 7,925,349 |
> | Train | 3,297,722 | 5,323,327 |
> | Validation | 729,204 | 1,221,643 |
> | Test | 819,342 | 1,380,379 |
> | Removed by filter | 2.2061% (109,324) | 0.8222% (65,705) |
>
> HDF5 SHA-256: call `ecf94de2996e3212ac791f944f847142e7a809c640b24cbd5989e8bdba801153`, put
> `46f3ee9d03d3423c6330c2b35a4d09893b0f538bdc046bc1018ebab79861c756`. Split date boundaries are
> **bit-identical to v4** (2128/266/267 dates, 2015-02-02 .. 2025-08-29) — the filter removed no
> trade date, only individual rows. v4 files re-hashed byte-identical to their original values (call
> `5f1c42aa2cc125fa0344b9e3d19fb374278aa629560c420c7ea4c76a6f910546`, put
> `2ba5b13b0beb374d86790b6fc94799ea92dbd057454fca4af86f357a1bafbd29`) — confirming v4 is unchanged and
> retained as the unfiltered robustness sample. `wrds_data_2020-2025/DATASET_MANIFEST.json` now
> exists and is populated as the single source of truth for dataset versions, paths, and hashes.
> `wrds_data_2020-2025/smoke_v5/` exists (`deeponet_tensors_{call,put}_v5_smoke.h5`) and is
> gate-verified — a real training script's `add_provenance()` was called against it directly and
> passed.
>
> **All three v5 CPU analyses are complete**, full test split, no subsampling.
> *Data quality* (`results/data_quality_v5.json`): **zero static-bound violations across every
> maturity bucket (d=2–7 through d>365) and every moneyness bucket (M<0.8 through M>1.2), for both
> option types**; duplicate-input rate remains 0.00% (unchanged from v4). *Diagnostics*
> (`results/diagnostics_tail_v5.json`): T=0 ceiling still not binding — `n_T0=0`,
> `ceiling_binding=False`, `r2_log_full_ceiling=1.0` for both option types (unchanged from v4, since
> the maturity floor is unaffected by the quote filter). *Baselines*
> (`results/baselines_v5.{json,csv}`): best price-accurate baseline (R²(price)>0.99) by R²(log) —
> **call** `vix_level_scaled_pxfit` **+0.7904** (price 0.999809, vs v4's +0.8011/0.999802); **put**
> `surface_spline` **−2.5186** (price 0.996594, vs v4's −2.4688/0.997004). **v5's numbers are
> slightly worse than v4's on both sides, and this is understood and expected, not a data-quality
> regression:** the rows the filter removed carried roughly twice their row-share of total variance
> (call −2.52% of rows / −5.16% of SST; put −0.73% of rows / −1.41% of SST) — removing the most
> extreme mispriced rows shrinks the R² denominator faster than the numerator, so v5 is a genuinely
> harder benchmark on the same metric than v4, not comparable apples-to-apples as a "did it get
> better" question. State this explicitly wherever v4-vs-v5 baseline numbers are compared. The same
> scalar-VIX anti-inference caveat as v4 applies: `vix_level_*` baselines use the contemporaneous
> scalar VIX close, not the `vix_history` neural branch, and cannot rank neural input branches
> against each other.
>
> **Test suite:** 292 tests pass as of this writing (may have grown further from concurrently-running
> provenance-fix and manifest-fix agents — treat as "292+ as of the last full run" if citing later).
> `git diff --check` clean.
>
> **What remains genuinely pending — unchanged by this note.** No neural model has been trained on
> v4 or v5; every R²/RMSE/MAE number in the 12-run table below remains v3-trained, historical, and
> not comparable to v4 or v5. The 12 core training runs (canonical target now v5), the seed-43/44
> replication for the four headline `vol_surface` configs, the `put/spot_history` DeepONet rerun, the
> per-query-vs-scalar ablation, the greeks-check execution, figures F1–F6, and the surface-wide
> arbitrage violation-rate diagnostic are all still pending GPU work. **Only canonical v5 neural
> training remains before the dataset/analysis side of the paper is complete** — not any remaining
> data-quality question.

> **2026-08-12 (later) — zero-tolerance static no-arbitrage quote-quality filter approved; v5
> dataset defined.** Full evidence and derivation are in `results/QUOTE_QUALITY_REPORT.md`
> (analysis-only pass, nothing filtered or rebuilt as of that document). The decision: drop any test
> row whose midpoint `(best_bid + best_offer)/2 / K` falls outside the static no-arbitrage band in
> normalized `V/K` units — calls outside `[max(fwd − disc, 0), fwd]`, puts outside
> `[max(disc − fwd, 0), disc]`, with `fwd = M·e^{−qT}`, `disc = e^{−rT}`, computed in float64 before
> the float32 HDF5 cast. Zero tolerance (`tol = 0`), fully deterministic, no tuned constant.
> Measured on the float32 v4 export (the float64 recomputation for the actual v5 build may shift
> these slightly — treat as **approximate** until the v5 build reports final counts): **call
> ~2.206%** removed (train 1.878% / val 3.324% / test 2.516%), **put ~0.822%** removed (train
> 0.881% / val 0.676% / test 0.726%); train-vs-test differential is small (+0.64 pp call, −0.16 pp
> put), unlike the spread-based rules that were considered and rejected (they shift 1.9–4.4× more of
> train than test — see the quote-quality report §3). Removal is **not** uniform: concentrated in
> deep-ITM calls (`M>1.2`, ~34% of that bucket) and deep-OTM puts (`M<0.8`, ~39%), and in the 2–7-day
> call bucket (~11.6%) — though those buckets are only 2.06%/0.30% of their option type, and the
> largest *absolute* contributors are moderately-ITM calls (42,384 rows) and near-ATM calls (32,065
> rows). A `tol = 1e-3` variant (cuts removal ~5×, call 0.437%, still catches every pathological row)
> was considered and **not** chosen — recorded as the documented fallback.
>
> **This filtered sample is dataset version v5**, distinct from v4: `schema_version` stays `"v4"`
> (the tensor schema/columns are unchanged) but a new independent attribute `dataset_version =
> "v5"` is added (the sample definition changed). Files:
> `wrds_data_2020-2025/deeponet_tensors_{call,put}_v5.h5`. **The unfiltered v4 files are retained**,
> explicitly labeled *"unfiltered robustness sample; noncanonical for training"* — not invalid, kept
> for a robustness check against v5. **All 12 training runs will target v5** once built; a
> `DATASET_MANIFEST.json` (queued) becomes the single source of truth for dataset versions, paths,
> and hashes. The v5 build itself, `VALIDATION_v5.json`, and the v5 re-runs of baselines/diagnostics/
> data-quality are **in progress as of this note** — do not cite v5 row counts or hashes beyond the
> approximate removal percentages above until that build completes.
>
> A **quote-liquidity stratification** (`best_bid > 0`, relative spread ≤ 1) is being added as
> *reporting strata only* — so the manuscript can show results are not driven by the zero-bid/wide-
> spread quotes retained by the bound filter. This is **not** a second filter; adopting it as one
> would need a separate decision.
>
> Precedent for excluding no-arbitrage-violating OptionMetrics observations exists in the literature
> (Buechner & Kelly, NBER w29369; Cohen, Reisinger & Wang, arXiv:2008.09454 — the latter also cautions
> that filtering discards information). Cited as precedent/caveat only; no specific numerical result
> from either is claimed here.

> **2026-08-12 — v4 dataset build completed; v3 in this file is now historical.** The blocker
> described in the 2026-08-11 note below (CPU preprocessing build in progress) is resolved: both
> `wrds_data_2020-2025/deeponet_tensors_{call,put}_v4.h5` exist and passed all 31 validation
> checks. Full detail in the dedicated "2026-08-12: v4 dataset build" section near the bottom of
> this file. **Every table and number in this file that follows (the 12-run matrix, the two
> "numerical facts", the ranking claims) describes v3-trained runs and is retained as the
> historical record** — it is not superseded science, but it is not comparable to v4 either. All
> 12 core runs must be retrained on v4 before any of this file's numbers can be cited as current.

> **2026-08-11 VVIX quarantine (input mislabel).** The `vix_history` branch was trained on **VVIX** —
> the CBOE volatility-of-VIX index (secid `152892`, ticker `VVIX`, from `vvix_2015_2025.csv`, columns
> `open/high/low/close`) — not VIX; the CSV was mislabeled. All four trained run dirs
> (`call|put × results_vix_history[,_don]`) were moved unchanged into
> `train_model_v3/_vix_is_vvix_LEGACY/` (call/put-prefixed; **no weights deleted**) and are excluded
> from `analysis/aggregate_results.py`. The `vix_history` numbers in the table below and any ranking
> claims citing them are **VVIX artifacts, not evidence about VIX** — pending retraining on corrected
> wide-format Cboe VIX (v4 HDF5, built by the preprocessing agent; deferred GPU commands in
> `train_model_v3/_vix_is_vvix_LEGACY/DEFERRED_GPU_COMMANDS.md`, migration record in
> `MIGRATION_NOTE.md`). What remains: preprocess the corrected VIX file into the v4 HDF5, then run
> the four deferred commands when the GPU is free.

> **2026-08-11 correction (documentation pass, no code/training changed):** the "Two numerical
> facts baked into the eval" section below (fact 1, the near-expiry tail) is **superseded** by a
> corrected mechanism. The original text attributed the whole tail to `T→0` ill-conditioning; the
> dominant driver is actually an exact-`T=0` **decoder degeneracy** from `T_years =
> calendar_days_to_expiry / 365` rounding same-day-expiry contracts to exactly zero, which is a
> preprocessing/decoder issue, not ill-conditioning. See the new "2026-08-11: near-expiry tail,
> corrected mechanism" section below for the full corrected account, kept alongside (not
> overwriting) the original text so the earlier reasoning stays visible. The headline
> `R²(log, T>1day)` numbers are unaffected either way. See also `report.md` §6 for the paper-facing
> version of this correction, and `PROGRESS_paper_artifacts.md` for the execution status of the
> `analysis/` package.

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

1. **Near-expiry / near-ATM log tail.** ⚠️ **SUPERSEDED 2026-08-11 — see the dedicated section below
   ("2026-08-11: near-expiry tail, corrected mechanism") for the accurate account.** Original text,
   kept for history: ~~~1.5% of contracts (T→0, ATM) carry ~93% of the log-space SSE. The inverse
   problem `log(price) → σ` is genuinely ill-conditioned as T→0 (`∂log(price)/∂σ` blows up), so
   log-perfection there is unattainable while the *absolute* price error is ~1e-6 (economically
   nil). Hence R²(price) ≈ 0.9998 while full-domain R²(log) ≈ 0.89.~~~ The "ill-conditioning"
   explanation for the *bulk* of the effect was wrong — the dominant driver is an exact-`T=0`
   decoder degeneracy, a preprocessing/decoder issue rather than a numerical-conditioning fact (the
   small-`T`-but-positive ill-conditioning claim itself is still correct, just not the dominant
   term). What does **not** change: the eval reports both R²(log, full) and R²(log, T>1day), and the
   **headline log metric remains `R²(log, T > 1/365)`** (≈0.9924 on the trained DeepONet vol_surface
   weights, and ≈0.9817 on put/vol_surface). A per-sample log-error percentile + tail diagnostic
   block in the test-eval characterizes this cluster (count, share of SSE, target/pred ranges,
   clamp-hit fraction, log_m / T ranges) — that instrumentation was measuring the right symptom
   (clamp-hit fraction = T=0 hit rate); only the diagnosis attached to it was wrong.
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

**All 12 runs are trained** (`{call,put} × {vol_surface, spot_history, vix_history} × {MLP-head
(A), DeepONet (B)}`) — but ⚠️ the four `vix_history` rows below are **VVIX artifacts** (input
mislabel, quarantined 2026-08-11 — see the note at the top of this file); only the 8 non-vix rows
are valid VIX-era results. Metrics below are read from the bottom of each run's `loss_history.txt`
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

*⚠️ All four `vix_history` rows above are VVIX-trained (quarantined 2026-08-11) — kept here only as
the historical record; they are **pending retraining** and must not be cited as VIX results.*

What the matrix says:
- **Branch ranking (8 valid runs): `vol_surface` ≫ `spot_history`** on the headline R²(log,
  T>1day), for both call and put. The IV-surface input prices best by a wide margin (call: .992 vs
  .949/.950). The `vix_history` slot in the ranking (previously .905/.916 call, .901/.902 put) is
  **suspended** — those numbers came from VVIX-trained models; the ranking will be restored when the
  corrected-VIX runs land.
- **A vs B close on `vol_surface`** (both option types). ⚠️ 2026-08-11: downgraded from "tie within
  seed noise" — the runs use **seed 42 only** (verified in every `config.json`), so there is no
  seed-noise evidence backing "tie"; it is close on this one seed and an untested hypothesis pending
  3-seed re-runs. On the remaining completed branch B is mixed: **notably unstable on
  put/spot_history (B R²(price) = 0.837**, a clear outlier vs A's 0.963) — likely
  patience-limited / an unstable run; flag for a re-run before publication.
- **Put R²(price) sits below call** across the board (vol_surface .996 vs .9999). ⚠️ 2026-08-11
  correction: the original framing here ("scaling artifact of the price-space metric, not a pricing
  failure") had it backwards. Verified baselines on the test split: call R²(price) is
  **near-vacuous** — a zero-parameter intrinsic-value baseline already reaches 0.999466, and a
  single constant σ=0.15 reaches 0.999811, beating several of the 12 call runs. Put R²(price) is
  **informative against naive baselines** (0.632539 zero-param → 0.946693 BS-constant-σ(0.20) →
  0.997241 best neural — a real progression) **but not against IV-surface interpolation**, which an
  independent reviewer measured at put R²(price) ≈ 0.997208 — essentially tied with the neural
  0.997241. **Reproduced in-repo 2026-08-12** by `analysis/baselines.py` on the v3 test split
  (put interpolation 0.997201; call interpolation 0.999856, which slightly *exceeds* the best
  neural call 0.999854) → `results/baselines_vix.*`. These are v3-sample numbers and must be
  regenerated on the v4 sample before they are cited as canonical. R²(log, T>1day)
  remains strong for both (.980+ put, .992+ call) and is unaffected by this correction — it is a
  log-space metric, not the price-space R² being discussed here.

## 2026-08-11: near-expiry tail, corrected mechanism

This section supersedes fact 1 under "Two numerical facts baked into the eval" above (kept there
with a strikethrough, not deleted, for history). The mechanism is **two separate phenomena**, not
one, and the dominant one is not ill-conditioning:

**(a) Exact `T=0` is a decoder degeneracy, not ill-conditioning (dominant term).** Preprocessing
computes `T_years = calendar_days_to_expiry / 365`, so every contract quoted on its own expiry date
gets **exactly** `T = 0`. The BS decoder clamps `sqrt_T = sqrt(clamp(T, 1e-10))`, so at `T = 0` it
returns intrinsic value **for any σ̂ whatsoever** — vega underflows to zero and the `1e-8` price
clamp zeroes the gradient a second time. The decoder output is mathematically independent of the
quantity being learned on these rows — they are structurally unfittable, not merely hard. Verified
irreducible error floor on the test split (SSE that no σ̂ could reduce):

| | T=0 rows | irreducible SSE | R²(log, full) ceiling | currently reported R²(log, full) |
|---|---|---|---|---|
| call | 38,765 (4.28%) | 9.86% of SST | **0.9014** | 0.894105 |
| put | 39,260 (2.68%) | 15.02% of SST | **0.8498** | 0.832595 |

Over 99.9% of that irreducible SSE comes from the OTM subset (14,129 calls / 25,062 puts), where the
prediction is pinned at `log(1e-8) = −18.42` against a market target near −12 (median absolute log
error 6.11 calls / 6.70 puts). The trained models already sit within ~0.007 (call) / ~0.017 (put) of
a ceiling **no model can cross** given this preprocessing — so most of the R²(log, full) ≈ 0.89 gap
is a **preprocessing specification bug** (undifferentiated `T=0` from calendar-day rounding), not a
fact about how hard the inverse pricing problem is.

**(b) Genuine small-but-positive-T ill-conditioning still stands (secondary).** Do not overcorrect:
for `0 < T ≤ 1 day`, `∂log(price)/∂σ` really does blow up near ATM and the inverse problem really is
ill-conditioned. This part of the original narrative is correct and is not being retracted — only
its share of the tail is smaller than previously stated, since (a) dominates.

**Headline metric unaffected.** `R²(log, T > 1/365)` already excludes every `T=0` row by
construction, so 0.9924/0.9817 (call/put, DeepONet vol_surface) stand exactly as reported. The
existing `--eval-only` clamp-hit-fraction diagnostic was already measuring the `T=0` symptom
correctly; only the diagnosis attached to it (ill-conditioning) was wrong.

**Remediation (queued, not done — no code changed by this doc pass).** Either (i) exclude
unresolved expiry-day rows from train/eval, or (ii) compute a defensible intraday maturity from
documented quote/settlement timestamps if recoverable from raw OptionMetrics data. **Not** an
epsilon added to `T` — that would fabricate optionality the contract does not have on its expiry
date.

Also see `report.md` §6 for the paper-facing writeup of this same correction (same numbers), and
§1 above for what this means for `CLAUDE.md`'s "Two recurring numerical facts" section (also
corrected in this pass).

## 2026-08-12: v4 dataset build completed

The full v4 HDF5 build (`wrds_data_2020-2025/deeponet_tensors_{call,put}_v4.h5`) finished at
09:14 and passed validation. This section is the canonical record of that build; do not re-derive
these numbers, and do not fabricate v4 **model** metrics — no v4 training has happened yet.

**Counts and splits:**

| | Call | Put |
|---|---:|---:|
| Total rows | 4,955,592 | 7,991,054 |
| Train | 3,360,823 | 5,370,629 |
| Validation | 754,276 | 1,229,955 |
| Test | 840,493 | 1,390,470 |
| Unique dates | 2,661 | 2,661 |

Coverage 2015-02-02 .. 2025-08-29; last train date 2023-07-17; last validation date 2024-08-06.
Built from 43 quarterly parquet parts (8.06 GB call / 13.00 GB put). HDF5 SHA-256 (abbreviated):
call `5f1c42aa…10546`, put `2ba5b13b…bd29` — full hashes in `wrds_data_2020-2025/VALIDATION_v4.json`
(cite the file, do not invent the full hash). Source VIX CSV SHA-256 (full):
`e2f18c516c9d48fb1730c2c0203fce36900c70aa0635b99e64afdde157346301`.

**Validation: all 31 checks passed** (`wrds_data_2020-2025/validate_v4.py` → `VALIDATION_v4.json`):
finite values, time-ordered non-overlapping splits by unique date, zero duplicate model inputs
(down from v3's 17.8%/19.8%), genuine VIX levels (9.14–82.69, not VVIX's 60–200 range), and direct
raw-VIX-to-HDF5 matches.

**Maturity decision, stated precisely.** The canonical cutoff is `--min-maturity-days 1.0` on the
**settlement-aware** maturity: contracts with strictly more than 24 hours remaining to settlement.
Observed minimum maturity is **1.729 days** — expected, not an anomaly: quotes are end-of-day, and
AM-settled (SPX) contracts settle at the 09:30 ET opening print, losing a further 6.5 hours, so a
2-calendar-day AM contract is 2 − 6.5/24 = 1.729 days. PM-settled (SPXW) contracts bottom out at
exactly 2.000 days (a 1-calendar-day PM contract is exactly 1.0 day and is excluded by the strict
`>`). **Important nuance:** v3's headline filter `R²(log, T > 1/365)` did not actually exclude
1-day contracts — `float32(1/365)` rounds above the true value, so it only ever dropped exact
`T = 0`. v4 genuinely excludes everything at or below 24 hours, which was a deliberate choice — do
**not** describe v4 as reproducing the v3 filter, and do not re-cut v4 to imitate the float32
rounding accident; that would weaken the methodology. Consequence: on v4, `T > 1/365` is a no-op,
so "headline" and "full-domain" `R²(log)` become the same number. A separately-named `T > 0`
diagnostic export for near-expiry sensitivity is possible future work but must never replace the
canonical sample.

**The all-12-must-be-retrained rule.** v4 changes three things at once relative to v3: (1) maturity
basis (settlement-aware, not `calendar_days/365`), (2) filtering (strict `>24h`, so the
structurally-unfittable exact-`T=0` rows described in "Two numerical facts" fact 1(a) above are
gone entirely), (3) contract identity (AM/PM settlement retained, collapsing the previously
duplicated SPX/SPXW inputs from 17.8%/19.8% to 0.00%). All three move the metrics independently of
any architecture change. A table mixing v3-trained and v4-trained rows would confound the branch
comparison with a dataset change and make the headline claim uninterpretable — this is why **all
12 core runs are retrained on v4**, not only the 4 quarantined `vix_history` ones. See
`train_model_v3/_vix_is_vvix_LEGACY/DEFERRED_GPU_COMMANDS.md` for the full command matrix and run
order (12 smoke jobs → provenance check + smoke-dir cleanup → the 12-run seed-42 v4 matrix → eval +
aggregate → seed-43/44 decision).

**v3-vs-v4 comparability rule (do not lose this).** All existing v3 neural metrics in this file
(the 12-run table above, the branch-ranking claims, the A-vs-B "close on this seed" framing) are
**historical and cannot be compared directly with v4 results**, for the three reasons just given.
Preserve the v3 numbers as clearly-labelled historical rows; do not delete or relabel them as v4.

**What is still pending on v4:** the 12 core model runs (and the WS4 per-query-vs-scalar ablation).
No v4 model metrics exist yet — none are stated here or should be inferred from the v3 table above.

**Update (2026-08-12, later same day) — the CPU-only analyses are no longer pending.**
`analysis/baselines.py`, `analysis/data_quality.py`, and `analysis/diagnostics.py` have all now run
on the full v4 test split → `results/{baselines,data_quality,data_quality_duplicates,
data_quality_bounds,diagnostics_tail,diagnostics_strata,diagnostics_conditioning}_v4.*`. Headline
results: duplicate-input rows 17.82%/19.81% (v3) → **0.00%/0.00%** (v4); static-bound violations
4.36%/1.45% (v3) → **2.5165%/0.7257%** (v4), zero at `T ≤ 1 day` (v3 had 31–33% there); the exact-T=0
decoder-degeneracy ceiling described in the "2026-08-11: near-expiry tail" section above is
**structurally eliminated** on v4 (`n_T0=0`, `r2_log_full_ceiling=1.0` for both option types, since
v4's `>24h` filter excludes those rows at export). Full detail, including the baseline ranking tables
and the interpretation caveats (scalar-VIX baseline ≠ evidence about the `vix_history` branch; puts
need both log-R² and price-R² reported together) are in `PROGRESS_paper_artifacts.md`'s "2026-08-12
addendum #2" and `report.md` §5.2/§6. **This remains baselines/diagnostics only — no v4 model has
been trained**; do not infer or extrapolate any v4 model metric from these numbers.

## Next steps

> **Publication-artifacts work has started — see `PROGRESS_paper_artifacts.md`.**
> The `analysis/` package's **model-evaluation** half (metrics-JSON dump, results
> aggregation, greeks check) is written but **not yet run** — it needs a
> v4/v5-trained model, which does not exist yet (GPU work still pending). Its
> **data-side** half is done: `analysis/baselines.py`, `analysis/data_quality.py`,
> and `analysis/diagnostics.py` have all been run on v3, v4, and the canonical v5
> test splits (see the 2026-08-13 note at the top of this file). The scalar-σ̂
> ablation and figures are specced in that doc §4 with the resolved data findings
> (branch_u is a tenor×delta surface; no per-contract observed-IV column →
> BS-invert for the smile figure).

The core 12-run matrix is **done** (✅), and the two comparison questions it was meant to answer have
preliminary answers (branch ranking is solid; A-vs-B "tie" is single-seed only, see 2026-08-11 note
above — treat as an untested hypothesis, not a resolved result, until 3-seed re-runs land). The live
work now shifts to the publication artifacts laid out in `pub_plan.md`:

1. ~~Train each of the 6 branches under both A and B (12 runs).~~ **Done** — see table above.
2. **Re-run the put/spot_history DeepONet (B)** — its R²(price)=0.837 is an outlier; confirm
   whether it is patience-limited / seed-unstable before it goes in a paper table.
3. **External baselines** (`analysis/baselines.py`) — ✅ **written and run 2026-08-12** on the full
   v3 test split (`results/baselines_vix.*`), ✅ **on the v4 test split**
   (`results/baselines_v4.*`, also 2026-08-12), and ✅ **on the canonical v5 test split**
   (`results/baselines_v5.*`, 2026-08-12/13) — BS constant-σ (ATM IV of day, VIX-scaled) and
   classic IV interpolation (spline / nearest / bilinear) over the same HDF5 test split, same
   metrics schema. Establishes the value baselines set for a future v5-trained model to beat; see
   `report.md` §5.2/§6, `PROGRESS_paper_artifacts.md`'s "2026-08-12 addendum #2", and the 2026-08-13
   note at the top of this file for the v4/v5 numbers and the interpretation caveats (scalar-VIX
   baseline vs the `vix_history` branch; puts need both log-R² and price-R² reported together; v5's
   slightly-worse-than-v4 numbers are an expected variance-composition effect, not a regression).
4. **Per-query σ̂ vs single-scalar σ̂ ablation** — the headline ablation (smile recovery); quantify
   the R²(log, T>1day) gap and produce the smile figure (F2).
5. **Greeks check** (`analysis/greeks_check.py`) — **written 2026-06-12, not yet executed** —
   autograd ∂price/∂S and vega through the BS decoder vs closed-form, expect ~1e-7 agreement;
   substantiates the "analytic BS partial greeks at fixed σ̂" claim (not "exact greeks" — see the
   script's own header comments at lines 83–87, which already state the caveat correctly: this
   validates the differentiable-decoder claim at the frozen predicted σ̂, not the total
   sticky-strike sensitivity of the learned system).
6. **Aggregation + figures** — dump per-run `metrics.json`, aggregate to LaTeX tables T1–T4, and
   build figures F1–F6 (F7 training curves already exist per run as `loss_plot.png`). As of
   2026-08-11 zero `metrics.json` files exist anywhere in the repo — the `analysis/` scripts that
   would produce them are written but unexecuted.
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
