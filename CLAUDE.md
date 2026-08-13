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

✅ **v5 dataset (2026-08-12/13, built + validated + analyzed — canonical).** A zero-tolerance static
no-arbitrage midpoint filter defines dataset version **v5**: drop any row whose midpoint (in
normalized `V/K` units, `fwd = M·e^{−qT}`, `disc = e^{−rT}`) falls outside `[max(fwd−disc,0), fwd]`
(calls) / `[max(disc−fwd,0), disc]` (puts), computed float64 before the float32 HDF5 cast. The build
is complete: `deeponet_tensors_{call,put}_v5.h5` (call 4,846,268 rows / put 7,925,349 rows; removed
2.2061% call / 0.8222% put), passed **45/45 validation checks** (`VALIDATION_v5.json`) with **zero
remaining static-bound violations**. Split date boundaries are bit-identical to v4 (the filter
removed rows, not trade dates). `schema_version` stays `"v4"`; `dataset_version = "v5"` is new. v4 is
**retained** as the unfiltered robustness sample, not invalidated. `wrds_data_2020-2025/
DATASET_MANIFEST.json` now exists and is the single source of truth for versions/paths/hashes. All
three v5 CPU analyses (`analysis/{data_quality,diagnostics,baselines}.py --dataset-version v5`) are
also complete — see `results/{data_quality,diagnostics_tail,baselines}_v5.*`. Full derivation and
the concentration/precedent caveats: `results/QUOTE_QUALITY_REPORT.md`, `PROGRESS_phase2_v3.md`,
`report.md` §4.1a. **No model has been trained on v4 or v5 yet** — that is the one remaining item
before the dataset/analysis side of the paper is complete.

**Data pipeline — v4 built 2026-08-12; v5 (quote-quality-filtered v4) built + validated + analyzed
2026-08-12/13 and is now canonical (see above). Both run from `wrds_data_2020-2025/`:**
```bash
python pre_process_1_data_v4.py --chunk-months 3   # raw CSVs -> deeponet_training_data_parts_v4/*.parquet
python pre_process_2_hdf5_v4.py --type call         # parquet -> deeponet_tensors_call_v4.h5
python pre_process_2_hdf5_v4.py --type put          # parquet -> deeponet_tensors_put_v4.h5
```
Phase 1 skips already-written parts unless `--force`. `--min-maturity-days` defaults to `1.0` (the
paper sample rule: strictly more than 24 hours to settlement); `--t-basis` defaults to `settlement`
(`calendar` reproduces v3 behavior for comparison only). See `wrds_data_2020-2025/README.txt` for
the full run order, self-tests, and smoke-slice invocation.

The v3 scripts (`pre_process_1_data.py`, `pre_process_2_hdf5.py`, `pre_process_3_print_dataset.py`)
are **legacy** — kept for provenance only, and v3 phase 1 can no longer even read the corrected
`vix_2015_2025.csv` (column-name mismatch: v3 expects `open/high/low/close`, the corrected file has
`vixo/vixh/vixl/vix`). Do not "fix" that by renaming columns; use the v4 scripts.

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

⚠️ **`vix_history` caveat (2026-08-11):** every existing `vix_history` trained
run was trained on **VVIX** — the CBOE volatility-of-VIX index (secid 152892) —
because the source CSV was mislabeled as VIX. The four trained run dirs are
quarantined (not deleted) under `train_model_v3/_vix_is_vvix_LEGACY/` (see its
`MIGRATION_NOTE.md`). Corrected runs on genuine Cboe VIX are **pending GPU
retraining** (deferred commands in `DEFERRED_GPU_COMMANDS.md`); they write to
`results_vix_history_v4/`-style dirs and take `--h5-path`/`--results-dir`
overrides. As of 2026-08-12 the four `vix_history` scripts **default to
`deeponet_tensors_{call,put}_v4.h5`** and validate provenance at startup: they
abort unless the HDF5 has `schema_version == "v4"`, a `phase1_vix_source_series`
naming `CBOE VIX`, and the reviewed VIX CSV's SHA-256. A v3 file is refused, so
a bare invocation can no longer train on VVIX and label the result VIX.

`call/` and `put/` are parallel copies; the v3 scripts point at
`deeponet_tensors_call.h5` / `deeponet_tensors_put.h5`, passing `option_type` to
the BS formula. As of 2026-08-12 the `vix_history` scripts default to the v4
files (`deeponet_tensors_{call,put}_v4.h5`); the other 8 scripts still default
to v3 paths pending their own retraining (§ below and `PROGRESS_phase2_v3.md`).

**Data contract (HDF5) — v4 build facts (now the unfiltered robustness sample;
v5 is canonical, see below).** The v4 build
(`wrds_data_2020-2025/deeponet_tensors_{call,put}_v4.h5`) completed 2026-08-12
09:14 and passed all 31 validation checks
(`wrds_data_2020-2025/VALIDATION_v4.json`):

| | Call | Put |
|---|---:|---:|
| Total rows | 4,955,592 | 7,991,054 |
| Train | 3,360,823 | 5,370,629 |
| Validation | 754,276 | 1,229,955 |
| Test | 840,493 | 1,390,470 |
| Unique dates | 2,661 | 2,661 |

Coverage 2015-02-02 .. 2025-08-29; last train date 2023-07-17; last validation
date 2024-08-06. HDF5 SHA-256 (abbreviated): call `5f1c42aa…10546`, put
`2ba5b13b…bd29` — full hashes are in `VALIDATION_v4.json`, do not invent or
re-quote them from memory. Source VIX CSV SHA-256 (full):
`e2f18c516c9d48fb1730c2c0203fce36900c70aa0635b99e64afdde157346301`.

**v5 (canonical, built + validated + analyzed 2026-08-12/13).** A
zero-tolerance static no-arbitrage midpoint filter applied to v4 defines v5
(`wrds_data_2020-2025/deeponet_tensors_{call,put}_v5.h5`), passed **45/45
validation checks** (`VALIDATION_v5.json`), zero remaining static-bound
violations:

| | Call | Put |
|---|---:|---:|
| Total rows | 4,846,268 | 7,925,349 |
| Train | 3,297,722 | 5,323,327 |
| Validation | 729,204 | 1,221,643 |
| Test | 819,342 | 1,380,379 |
| Removed by filter | 2.2061% (109,324) | 0.8222% (65,705) |

Split date boundaries are bit-identical to v4 (2128/266/267 dates,
2015-02-02..2025-08-29) — the filter removed rows, never a trade date. HDF5
SHA-256: call `ecf94de2996e3212ac791f944f847142e7a809c640b24cbd5989e8bdba801153`,
put `46f3ee9d03d3423c6330c2b35a4d09893b0f538bdc046bc1018ebab79861c756`.
`schema_version` stays `"v4"`; `dataset_version="v5"` is new.
`wrds_data_2020-2025/DATASET_MANIFEST.json` is the single source of truth for
dataset versions/paths/hashes. `wrds_data_2020-2025/smoke_v5/` exists and is
gate-verified against a real training script's `add_provenance()`. All three
v5 CPU analyses (`analysis/{data_quality,diagnostics,baselines}.py`) are also
complete — see `results/{data_quality,diagnostics_tail,baselines}_v5.*`.

**Maturity rule (v4, decided).** `--min-maturity-days 1.0` on the
**settlement-aware** maturity: contracts with strictly more than 24 hours
remaining to settlement. Observed minimum is **1.729 days** (AM-settled SPX
contracts lose 6.5h to the 09:30 ET opening print vs an EOD quote — expected,
not an anomaly). PM-settled SPXW contracts bottom out at exactly 2.000 days.
This is **not** the same filter as v3's `R²(log, T > 1/365)` — that filter only
ever dropped exact `T = 0` because `float32(1/365)` rounds above the true value.
v4's filter genuinely excludes everything ≤24h, which is why on v4 the
`T > 1/365` filter is a no-op (headline and full-domain `R²(log)` coincide). Do
not re-cut v4 to imitate the v3 float32 rounding accident.

v4 keeps every v3 dataset name, shape, dtype, and row alignment — `branch_u`,
`spot_history`, `vix_history` (the three branch inputs), `trunk_y` =
`[log_moneyness, T_years, r, q]`, `target_v_log` = `log(mid/K)`, plus
`moneyness`, `normalized_price`, `date`, and `split_id` (0=train, 1=val,
2=test) — and adds new columns alongside (`symbol`, `optionid`, `root`,
`am_settlement`, `exercise_style`, `impl_volatility`, market delta, `best_bid`,
`best_offer`, `spread_norm`, `half_spread_norm`, `strike`, `T_calendar`,
`T_settlement`, `vix_level`). **The split is by unique trade date (80/10/10),
not by row** — this is a deliberate no-leakage time-ordered split; preserve it.

**v3-vs-v4/v5 comparability rule.** All existing v3-trained neural results
(`PROGRESS_phase2_v3.md`'s 12-run table, `report.md` T1) are **historical and
cannot be compared directly with v4 or v5 results.** v4/v5 change three things
at once relative to v3: (1) maturity basis (settlement-aware, not
`calendar_days/365`), (2) filtering (strict >24h, so the
structurally-unfittable exact-`T=0` rows described below are gone), (3)
contract identity (AM/PM settlement retained, collapsing the previously
duplicated SPX/SPXW inputs from 17.8%/19.8% to 0.00%). This is why **all 12
core runs will be retrained on v5** (the canonical target; v4 is retained only
as the unfiltered robustness sample), not only the 4 quarantined
`vix_history` ones — see
`train_model_v3/_vix_is_vvix_LEGACY/DEFERRED_GPU_COMMANDS.md`.

⚠️ **v3 legacy note.** The v3 (`deeponet_tensors_{call,put}.h5`, no `_v4`
suffix) files still exist and still back every number currently in `report.md`
and `PROGRESS_phase2_v3.md`'s 12-run table — they are not deleted, only
superseded as the canonical dataset. In v3, the `vix_history` dataset was built
from **VVIX**, not VIX (mislabeled CSV, quarantined 2026-08-11 — see above and
`train_model_v3/_vix_is_vvix_LEGACY/MIGRATION_NOTE.md`); v4 stores genuine Cboe
VIX and is the fix for that mislabel as well as the maturity/identity issues.

## Two recurring numerical facts you must know

These dominate the project's metrics and have already been investigated at length
(see `PROGRESS_phase2_v3.md`). Do not re-discover them as "bugs" — but also see the
2026-08-11 correction below on fact 1: the *mechanism* originally written here was
wrong, even though the instrumentation and the headline number were right.

1. **The near-expiry / near-ATM log tail — two distinct mechanisms, not one.**
   The full-domain `R²(log) ≈ 0.89` gap was originally attributed entirely to
   `T→0` ill-conditioning of the inverse problem `log(price) → σ`. That
   single-mechanism story is wrong. There are two separate effects:
   - **(a) Exact `T=0` is a decoder degeneracy (dominant), not ill-conditioning.**
     Preprocessing sets `T_years = calendar_days_to_expiry / 365`, so every
     contract quoted on its own expiry date gets **exactly** `T = 0`. The BS
     decoder clamps `sqrt_T = sqrt(clamp(T, 1e-10))`, so at `T = 0` it returns
     intrinsic value **for any σ̂ whatsoever** — vega underflows and the `1e-8`
     price clamp zeroes the gradient a second time. The decoder output is
     mathematically independent of the quantity being learned on these rows —
     they are structurally unfittable, not merely hard. Verified irreducible SSE
     floor on the test split: call 9.86% of SST (38,765 rows, 4.28%), put 15.02%
     of SST (39,260 rows, 2.68%); `R²(log, full)` ceilings 0.9014 (call) / 0.8498
     (put) — the trained models already sit within ~0.007–0.017 of a ceiling **no
     model can cross** under this preprocessing. Over 99.9% of that SSE is the OTM
     subset, pinned at `log(1e-8) = −18.42` against a market target near −12.
     Remediation (queued, not done): exclude unresolved expiry-day rows, or derive
     a defensible intraday maturity from documented quote/settlement timestamps.
     **Do not** "fix" this with an epsilon added to `T` — that manufactures
     optionality the contract does not have.
   - **(b) Genuine small-but-positive-T ill-conditioning (secondary, still real).**
     For `0 < T ≤ 1 day`, `∂log(price)/∂σ` really does blow up near ATM and the
     inverse problem really is ill-conditioned — this part of the original story
     stands, it just isn't the dominant term.
   - **The headline metric is unaffected either way.** `R²(log, T > 1/365)`
     already excludes every `T=0` row by construction, so the reported **0.9925
     (call) / 0.9817 (put)** stand unchanged. `R²(price) ≈ 0.9998` while
     full-domain `R²(log) ≈ 0.89` remains true as a description, just not for the
     reason originally given. Report all of R²(price), R²(log, full), and
     R²(log, T>1day) together; treat the last as headline. See
     `PROGRESS_phase2_v3.md` ("2026-08-11: near-expiry tail, corrected
     mechanism") and `report.md` §6 for the full account, kept alongside the
     retired explanation for history rather than overwriting it.
2. **The pricing path has two modes.** Default training/eval uses
   `log(clamp(price, 1e-8))`. `--stable-log` swaps in `bs_log_normalized_price`
   (`log_ndtr` + `log(-expm1(...))`), which is exact but whose gradient explodes
   near ATM. **`--stable-log` is eval-only**; dropping it into training NaNs out
   because the global `clip_grad_norm_` then crushes the rest of the batch. The
   stable-log eval *confirmed* the clamp masks a real model error on the tail, it
   does not create one.

## Two claims that were overstated in early docs/writeup (corrected 2026-08-11)

`report.md` originally claimed the BS-decoder design gives "exact greeks" and
"arbitrage-free by construction" prices. Both are overstated; if you see either
phrase reintroduced, it is regressing a known correction:

- **Not "arbitrage-free by construction" — pointwise Black–Scholes-consistent.**
  Feeding a predicted σ̂(K,T) through BS gives a valid price at that one query
  (it satisfies the BS PDE and static no-arbitrage bounds there), but nothing
  guarantees calendar-spread or butterfly consistency **across** the predicted
  surface, since σ̂ is fit independently per query. A violation-rate diagnostic
  on the predicted surface is queued work, not yet built.
- **Not "exact greeks" — analytic BS partial greeks at the predicted σ, holding
  σ fixed.** `analysis/greeks_check.py` differentiates the BS decoder at a
  frozen σ̂; it checks that autograd through the decoder agrees with the
  closed-form BS formula, not the total sensitivity of the learned system (which
  would include a `dσ̂/dS · vega` term). The script's own header comments
  (lines 83–87) already state this correctly — it was only the `report.md`
  prose that overclaimed.

`compute_loss` ships with **MSE active and `F.huber_loss(delta=1.0)` commented
out** — a deliberate one-line toggle. Huber was tried and reverted (it stopped
earlier and hurt R²(price) without moving R²(log)); leave MSE as the default
unless a future dataset has heavier tails.

## Status, layout, and history

- **`PROGRESS_phase2_v3.md` is the source of truth for current training state** —
  read it before resuming. All 12 core runs (`{call,put} × {vol_surface,
  spot_history, vix_history} × {MLP-head, DeepONet}`) are trained **on v3**, but
  ⚠️ the four `vix_history` runs are **VVIX artifacts, quarantined 2026-08-11**
  (input mislabel — see `train_model_v3/_vix_is_vvix_LEGACY/MIGRATION_NOTE.md`);
  8 runs are valid v3 results. **As of 2026-08-12/13 the canonical dataset is
  v5** (quote-quality-filtered v4, built + validated + fully analyzed — see
  above; v4 is retained only as the unfiltered robustness sample), and **all 12
  runs — not only the 4 VIX ones — are pending retraining on v5** (deferred
  commands in `DEFERRED_GPU_COMMANDS.md`); no v4 or v5 model has been trained
  yet — that is the one remaining item before the dataset/analysis side of the
  paper is complete.
- **`PROGRESS_paper_artifacts.md` is the source of truth for the `analysis/`
  package and publication-artifact status.** The package has two halves. The
  **model-evaluation half** (`_common.py`, `eval_to_json.py`,
  `aggregate_results.py`, `greeks_check.py`, written 2026-06-12) has **never
  been executed** — zero `metrics.json` anywhere in the repo, no canonical
  aggregated results — but only because no v4/v5-trained model exists yet to
  evaluate; this is GPU-blocked, not stalled. The **data-side half**
  (`analysis/baselines.py`, `analysis/data_quality.py`, `analysis/diagnostics.py`)
  is fully written and has been **run on the v3, v4, and canonical v5 test
  splits** (2026-08-12/13) — see `results/{baselines,data_quality,
  diagnostics_tail}_v5.*` and `PROGRESS_paper_artifacts.md`'s 2026-08-13
  addendum. Every neural R²/RMSE/MAE number currently in `report.md` is still
  hand-copied from a `loss_history.txt` tail (v3-trained), not generated by
  `eval_to_json.py`/`aggregate_results.py` — that remains true until a v4/v5
  model is trained.
- **`report.md` is the paper skeleton** — its top status table tracks which
  sections are backed by real numbers vs `> **TODO**`. A 2026-08-11
  documentation-only pass corrected several overstated/incorrect claims there
  (arbitrage-free, exact greeks, the near-expiry tail mechanism, the call-vs-put
  R² framing) — see that file's own top-of-document correction note for the
  full list before citing any of the corrected sections.
- `plan_phase1.md`, `plan_phase2{,_v2,_v3}.md` are design docs; `plan_phase2_v3.md`
  predates the v3.1/v3.2 changes (per-query σ̂, filtered eval, `--eval-only`/
  `--stable-log`) — trust `PROGRESS_phase2_v3.md` over it where they disagree.
- `train_model/` (v1) and `train_model_v2/` are superseded; v2 still had the
  learned two-network design with `pretrain_net2.py`. `OLD_*/` and `archive/` and
  `wrds_data/` (the older 2020-2025-named-but-different dataset) are legacy —
  `wrds_data_2020-2025/` is the current data.
- `final_report.tex` / `DLforPhysicalSystems_project.pdf` are the writeup.
