# Manuscript Readiness Inventory — v5 Experimental Campaign

**Purpose.** A single reference for whoever drafts the results section next: which file backs
which claim, exactly. Every number below cites the file it was pulled from. Do not re-derive or
re-round anything here without also updating the citation.

**Status as of this writing.** The full v5 experimental campaign is complete: canonical dataset
built + validated, all 12 canonical seed-42 cells trained/evaluated, 8 seed-43/44 replications of
the 4 headline `vol_surface` configs trained/evaluated/aggregated, CPU baselines/data-quality/tail
diagnostics run on v5. Nothing here is invented — see §7 for what genuinely remains open.

---

## 1. Dataset & preprocessing

### 1.1 Dataset versions

| Version | Status | Call rows (train/val/test) | Put rows (train/val/test) | Validation | Source |
|---|---|---:|---:|---|---|
| v3 | historical, not comparable to v4/v5 (see §1.3) | — (see `PROGRESS_phase2_v3.md`) | — | n/a | `report.md`, `PROGRESS_phase2_v3.md` |
| v4 | unfiltered robustness sample; noncanonical for training | 4,955,592 total (3,360,823 / 754,276 / 840,493) | 7,991,054 total (5,370,629 / 1,229,955 / 1,390,470) | 31/31 checks passed | `wrds_data_2020-2025/VALIDATION_v4.json`, `wrds_data_2020-2025/DATASET_MANIFEST.json` |
| **v5** | **canonical for training** | 4,846,268 total (3,297,722 / 729,204 / 819,342) | 7,925,349 total (5,323,327 / 1,221,643 / 1,380,379) | **45/45 checks passed**, zero remaining static-bound violations | `wrds_data_2020-2025/VALIDATION_v5.json` (`n_checks: 45`), `wrds_data_2020-2025/DATASET_MANIFEST.json` |

Split date boundaries are bit-identical between v4 and v5 (2128 train / 266 val / 267 test dates,
2015-02-02 .. 2025-08-29) — the quote filter removed rows, not trade dates
(`wrds_data_2020-2025/DATASET_MANIFEST.json`).

**HDF5 SHA-256 (v5, canonical):**
- call: `ecf94de2996e3212ac791f944f847142e7a809c640b24cbd5989e8bdba801153`
- put: `46f3ee9d03d3423c6330c2b35a4d09893b0f538bdc046bc1018ebab79861c756`

(`wrds_data_2020-2025/DATASET_MANIFEST.json`; also reproduced in `wrds_data_2020-2025/VALIDATION_v5.json`.)

Rows removed by the quote filter going v4→v5: call 2.2061% (109,324 rows), put 0.8222% (65,705
rows) (`wrds_data_2020-2025/DATASET_MANIFEST.json`).

### 1.2 The two headline preprocessing decisions

1. **Maturity rule — settlement-aware, strictly >24h to settlement.** `min_maturity_days=1.0` on
   `t_basis=settlement` ("contracts with strictly more than 24 hours remaining to settlement").
   Rationale: avoids the v3 accident where `T = calendar_days/365` silently zeroed exact-expiry-day
   rows via float32 rounding, and correctly separates AM-settled SPX (loses 6.5h to the 9:30 ET
   opening print) from PM-settled SPXW contracts. Full writeup: `CLAUDE.md` ("Maturity rule (v4,
   decided)"), `PROGRESS_phase2_v3.md`'s 2026-08-12 section. Recorded in
   `wrds_data_2020-2025/DATASET_MANIFEST.json` (`maturity_rule` block, identical for v4 and v5).
2. **Quote filter — zero-tolerance static no-arbitrage midpoint filter (v4→v5).** Drop any row
   whose midpoint (`V/K` units, `fwd = M·e^{-qT}`, `disc = e^{-rT}`) falls outside
   `[max(fwd-disc,0), fwd]` (calls) / `[max(disc-fwd,0), disc]` (puts), computed float64 pre-cast.
   Rationale: these midpoints are unreachable by the BS decoder at any σ̂ given the supplied
   `r`/`q`/settlement inputs (not "provable market arbitrage" — a weaker, defensible claim; see
   `results/QUOTE_QUALITY_REPORT.md` §2.3 for the exact wording constraint). Full writeup and the
   concentration caveat (removal skews toward deep-ITM calls / deep-OTM puts / 2-7-day calls):
   `results/QUOTE_QUALITY_REPORT.md`, `PROGRESS_phase2_v3.md`, `report.md` §4.1a.

### 1.3 v3-vs-v4/v5 comparability

v3 numbers (in `report.md`'s existing T1 table and `PROGRESS_phase2_v3.md`'s 12-run table) are
**historical and not comparable** to v4/v5 — three things changed at once (maturity basis, quote
filtering, AM/PM contract-identity collapse). No model has ever been trained on v4 as the canonical
target; v4 is retained only as the unfiltered robustness sample. **v5 is the only canonical target.**
(`CLAUDE.md` "v3-vs-v4/v5 comparability rule"; `wrds_data_2020-2025/DATASET_MANIFEST.json`
`"canonical": "v5"`.)

---

## 2. Main results (T1) — canonical v5, seed 42

**Source:** `results/table_T1_v5.md` / `results/table_T1_v5.tex` (also machine-readable:
`results/all_metrics_v5.csv`). Exactly the 12 canonical seed-42 cells: `{call, put} ×
{vol_surface, spot_history, vix_history} × {A (MLP-head), B (DeepONet)}`.

| option | branch | arch | R2(price) | R2(log,T>1d) | RMSE(log) |
|---|---|---|---:|---:|---:|
| call | vol_surface | A | 0.999977 | 0.991044 | 0.221894 |
| call | vol_surface | B | 0.999914 | 0.990290 | 0.231041 |
| call | spot_history | A | 0.999812 | 0.922020 | 0.654761 |
| call | spot_history | B | 0.999181 | 0.944390 | 0.552923 |
| call | vix_history | A | 0.999848 | 0.921333 | 0.657639 |
| call | vix_history | B | 0.998125 | 0.939874 | 0.574940 |
| put | vol_surface | A | 0.992553 | 0.977898 | 0.294136 |
| put | vol_surface | B | 0.993780 | 0.981298 | 0.270566 |
| put | spot_history | A | 0.952078 | 0.909410 | 0.595489 |
| put | spot_history | B | 0.942330 | 0.901553 | 0.620777 |
| put | vix_history | A | 0.962518 | 0.920722 | 0.557070 |
| put | vix_history | B | 0.961502 | 0.917777 | 0.567323 |

(full precision and additional columns — R2(log,full), RMSE(price), MAE(price) — in
`results/table_T1_v5.md`; note on v5, T>1day and full-domain are the same number because v5 has no
`T=0` rows, see §5)

**Headline framing:** vol_surface reaches ~0.99 (call) / ~0.98 (put) log-R²; spot_history and
vix_history sit in the 0.90–0.94 band on both option types. This is the core gap the paper's
architecture claims rest on.

**R2(price) near-saturation caveat (calls).** Every call cell reaches R2(price) ≥ 0.998, including
the weakest branch (vix_history B, 0.998125). This is expected and already documented as
uninformative on its own — the zero-parameter intrinsic baseline alone reaches R2(price) ≈ 0.9986–
0.9995 on this scale (§4 below; `PROGRESS_paper_artifacts.md` addendum "0. 2026-08-11", item 1;
`CLAUDE.md` "Two claims that were overstated" section). Use R2(log) as the discriminating metric for
calls; R2(price) is still meaningful for puts (spot_history/vix_history puts drop to 0.94–0.96,
comparable-magnitude but not vacuous).

---

## 3. Replication / robustness — seeds 42/43/44

**Source:** `results/replication_seeds_v5.md` (also `results/replication_seeds_v5.csv`). Covers
only the 4 headline `vol_surface` configs (call/put × A/B); `spot_history`/`vix_history` cells are
**single-seed** (see caveat below).

| option | branch | arch | seeds | R2(price) mean±SD | R2(log,filtered) mean±SD | RMSE(log) mean±SD |
|---|---|---|---|---:|---:|---:|
| call | vol_surface | A | 42,43,44 | 0.999972 ± 0.000005 | 0.990996 ± 0.000073 | 0.222493 ± 0.000896 |
| call | vol_surface | B | 42,43,44 | 0.999865 ± 0.000131 | 0.991075 ± 0.000716 | 0.221388 ± 0.008838 |
| put | vol_surface | A | 42,43,44 | 0.994660 ± 0.001898 | 0.979099 ± 0.001424 | 0.285924 ± 0.009813 |
| put | vol_surface | B | 42,43,44 | 0.995392 ± 0.001403 | 0.981738 ± 0.000611 | 0.267345 ± 0.004491 |

**What this confirms.** Across-seed dispersion on the 4 vol_surface cells is small relative to the
vol_surface-vs-spot/vix-history gap in §2 (e.g. call vol_surface A: SD 0.000073 on R2(log) vs a gap
of ~0.07 to spot_history A's 0.922020) — the vol_surface architecture's advantage over
spot_history/vix_history is robust to seed variation, by roughly **40x to just under three orders
of magnitude** depending on the specific cell/metric compared (not "two orders of magnitude" flatly
— that phrasing overstated it for some cells and was corrected).

**Architecture-ranking caution (state verbatim, do not soften).** The replications strongly confirm
vol_surface's advantage over spot/vix history, but **architecture ranking (A vs B) is not
established**: call A vs B are effectively tied (both metrics overlap within ~1 SD:
R2(log) 0.990996±0.000073 vs 0.991075±0.000716), and put B looking somewhat better than put A
(0.981738±0.000611 vs 0.979099±0.001424) is based on only 3 seeds — not enough for a strong
architecture-superiority claim. This is an open item for the results section, not a settled
finding — see §7.

**Single-seed caveat (must not be implied away).** `spot_history` and `vix_history` cells (all 8 of
them, both option types × both architectures) are seed-42 only. Their replication dispersion is
**genuinely unmeasured, not zero** — `results/replication_seeds_v5.md` correctly excludes these
rows rather than showing a fake `±0.000000` (a bug fixed in commit `c6801b5`, "Fix fake std=0 for
single-seed cells in replication table"). Do not present or imply any variance number for these 8
cells.

---

## 4. Baselines

**Source:** `results/baselines_v5.json` / `results/baselines_v5.csv`. Same test split, same
`compute_metrics` schema as the neural models. Two families per option type: BS constant-σ /
IV-surface interpolation (branch-comparable to `vol_surface`), realized-vol/EWMA (branch-comparable
to `spot_history`), scalar-VIX (context-free, NOT branch-comparable to `vix_history` — see below).

**Best-baseline ranking (must always report both R2(log) and R2(price), never just one):**
- **Call:** best R2(log) baseline is `vix_level_scaled_pxfit` at **+0.7904** (R2(price)
  0.999809) — also the best price-accurate one on this option type
  (`results/baselines_v5.json` record `baseline="vix_level_scaled_pxfit"`,
  `metrics.all.r2_log_full = 0.7904245616752839`, `r2_price = 0.9998091473999837`; corroborated in
  `PROGRESS_paper_artifacts.md` addendum -1.6).
- **Put:** two different baselines win on the two metrics, and both must be reported — never just
  one:
  - The **least-negative R2(log)** baseline is `vix_level_scaled` at **-0.3563** (`results/
    baselines_v5.json`, `option_type="put"`, `baseline="vix_level_scaled"`, `metrics.all.
    r2_log_full = -0.3563177573450871`) — but it is **price-inaccurate**: R2(price) =
    **-3.8454** (`metrics.all.r2_price = -3.8453744869713047`), i.e. catastrophic in price space.
  - The **best price-accurate** baseline is `surface_spline` at R2(price) **+0.9966**
    (`results/baselines_v5.json`, `baseline="surface_spline"`, `metrics.all.r2_price =
    0.9965944957550128`), but its R2(log) is **-2.5186** (`metrics.all.r2_log_full =
    -2.518558932468415`) — far worse in log space than `vix_level_scaled`.

  Other put baselines for context (`results/baselines_v5.json`, `metrics.all`): `const_sigma`
  R2(log) -0.3881 / R2(price) -2.3582; `ewma_vol_scaled` R2(log) -0.7157 / R2(price) -14.4237;
  `realized_vol_21d_scaled` R2(log) -0.7259 / R2(price) -14.6129; `prev_day_surface_bilinear`
  R2(log) -2.5318 / R2(price) 0.9911. (`PROGRESS_paper_artifacts.md` addendum -1.6 also discusses
  this family, but the numbers above are read directly from `results/baselines_v5.json`.)

**Interpretation caveats (reuse this exact wording, do not re-derive):**
1. **Calls are near-vacuous on R2(price).** The zero-parameter `intrinsic_zero_vol` baseline
   alone reaches R2(price) = 0.9985863019037233 on v5
   (`results/baselines_v5.json`, `baseline="intrinsic_zero_vol"`, `metrics.all.r2_price`) — see
   also §2's near-saturation caveat above.
2. **Puts vs IV-surface interpolation: a metric tradeoff, not a tie.** (Calls: the
   `surface_spline`/`surface_bilinear`/`surface_nearest` family, branch-comparable to
   `vol_surface`, reaches R2(price) ≈ 0.9999–0.99998, e.g. `surface_spline` `r2_price =
   0.9999824083511915` — `results/baselines_v5.json`.) For puts, comparing the neural
   `vol_surface` cells (`results/table_T1_v5.md`) against the `surface_spline` baseline
   (`results/baselines_v5.json`, `option_type="put"`, `baseline="surface_spline"`,
   `metrics.all`: `r2_log_full = -2.518558932468415`, `r2_price = 0.9965944957550128`) shows the
   two metrics disagree on which one wins:
   - **Log-space (the harder, more informative metric — see caveat 1 above and §2's
     near-saturation discussion of R2(price)):** the neural models win decisively.
     `results/table_T1_v5.md`: put vol_surface arch A R2(log,T>1d) = 0.977898, arch B =
     0.981298 — both far above `surface_spline`'s -2.5186. Not a small gap.
   - **Price-space:** `surface_spline` edges out the neural models slightly — R2(price)
     0.996594 vs 0.992553 (arch A) / 0.993780 (arch B), from `results/table_T1_v5.md` — but this
     is the metric caveat 1 already documents as near-saturated/uninformative on calls, and the
     put price gap here is small in absolute terms.

   Both numbers exist already in `results/table_T1_v5.md` and `results/baselines_v5.json` — no
   regeneration needed. Do not describe this as the models being "tied" to surface interpolation;
   state both directions of comparison.
3. **Scalar-VIX anti-inference caveat (verbatim from the data itself).** Every `vix_level_*`
   baseline record in `results/baselines_v5.json` carries this note field:
   > "ANTI-INFERENCE: this is a contemporaneous scalar VIX benchmark (one index close per date) and
   > NOT the model's `vix_history` branch (a 21x4 OHLC tensor, close-normalised so the absolute VIX
   > level is destroyed), so it cannot be used to rank the neural input branches against each
   > other."

   Do not read `vix_level_scaled_pxfit`'s strong call ranking (+0.7904) as evidence that the
   `vix_history` neural branch would outperform `vol_surface` — it is a different input with
   different information content. Same wording appears in `analysis/baselines.py`'s output schema
   and `report.md` (search "ANTI-INFERENCE" / "contemporaneous scalar VIX").
4. **v4-vs-v5 baseline comparison is not apples-to-apples.** v5's baseline numbers are slightly
   worse than v4's on both option types (call +0.7904 vs v4's +0.8011; put -2.5186 vs v4's -2.4688)
   because the quote filter removed the most extreme mispriced rows, which carried roughly twice
   their row-share of total variance — this shrinks the R² denominator faster than the numerator.
   Not a data-quality regression. (`PROGRESS_paper_artifacts.md` addendum -1.6.)

---

## 5. Data quality / diagnostics

**Sources:** `results/data_quality_v5.json`, `results/diagnostics_tail_v5.json`,
`results/QUOTE_QUALITY_REPORT.md`.

- **Zero remaining static-bound violations.** `results/data_quality_v5.json`:
  `bounds.n_violations = 0`, `bounds.pct_violations = 0.0` for both call (819,342 rows) and put
  (1,380,379 rows), across every maturity bucket (`d=2-7` through `d>365`) and every moneyness
  bucket (`M<0.8` through `M>1.2`).
- **Duplicate-input rate 0.00%.** `results/data_quality_v5.json`: `duplicates.max_group_size = 1`
  for both option types — every row is unique on `(date, log_m, T, r, q)`, even without the
  `am_settlement` key. This confirms the v4→v5 fix of the v3 issue where AM-settled SPX and
  PM-settled SPXW collapsed to identical model inputs (17.82% call / 19.81% put in v3).
- **T=0 decoder-degeneracy ceiling not binding.** `results/diagnostics_tail_v5.json`:
  `ceiling.n_T0 = 0`, `ceiling.ceiling_binding = false`, `ceiling.r2_log_full_ceiling = 1.0` for
  both option types — no row in v5 sits at exact T=0 (the maturity rule in §1.2 already excludes
  it), so the mechanism that capped v3's R²(log,full) at 0.9014/0.8498 does not apply here. Full
  mechanism writeup (kept for context, not re-litigated): `CLAUDE.md` "Two recurring numerical
  facts", item 1.
- **Quote filter derivation and concentration caveat.** `results/QUOTE_QUALITY_REPORT.md` — full
  trace of the two deepest violating rows found pre-filter (a train-split deep-ITM LEAPS call with
  a 0.00 bid, depth 9.0337 in V/K units; a test-split deep-OTM LEAPS put, depth 0.1693), and the
  wording constraints on how to describe the filter (§2.3: not "provable market arbitrage",
  the IV-NULL agreement is corroborating not independent evidence).

---

## 6. Provenance / reproducibility

- **Single source of truth for dataset versions/paths/hashes:**
  `wrds_data_2020-2025/DATASET_MANIFEST.json` (`"canonical": "v5"`, generated
  2026-08-13T12:40:12+00:00).
- **Dataset validation:** `wrds_data_2020-2025/VALIDATION_v5.json` (45/45 checks, `n_checks: 45`),
  `wrds_data_2020-2025/VALIDATION_v4.json` (31/31 checks).
- **Commits (tracked worktree, all on branch `snapshot/pre-publication-2026-08-11`):**
  - `00897f6` — "Canonical v5 seed-42 matrix: 12/12 runs, evaluated, aggregated"
  - `c6801b5` — "Fix fake std=0 for single-seed cells in replication table"
  - `d5410e5` — "Seed-43/44 vol_surface replications: 8/8 complete, validated, aggregated"

  (confirmed via `git log --oneline`; the tracked worktree is clean as of this inventory.)
- **Checkpoint policy.** `.gitignore` excludes `*.pth` (trained weights) and `*.h5`/`*.csv`/
  `*.parquet` globally, with explicit exceptions carved out for `results/*.csv`/`*.json`/`*.md`
  (those are force-tracked — confirmed via `git ls-files results/`). What IS committed per run:
  `config.json`, `metrics.json`, `loss_history.txt`, `loss_plot.png`. What is NOT committed:
  `best_model.pth`, `best_model_phase1.pth`, `final_model.pth`, and per-batch logs.
  **Weights for reproduction live locally, uncommitted, under
  `train_model_v3/{call,put}/results_{vol_surface,spot_history,vix_history}[_don]_v5[_seed43|_seed44]/`**
  (e.g. `train_model_v3/call/results_vol_surface_v5/best_model.pth`) — anyone reproducing figures
  or re-running eval needs access to that local directory tree, not just the git history.

---

## 7. Open items before submission

**Blocking (preprint-critical — must be resolved before submission):**
- **Per-query-vs-scalar-σ ablation not yet run.** This is the headline ablation motivating the
  conditional-vol-head design (per-query σ̂ is the paper's headline contribution 1, and it is what
  the proposed core figure F2 depends on). Without it, that headline architectural claim is
  currently unsupported. Speccable but not started on v5 (WS4 in `PROGRESS_paper_artifacts.md`).
- **Greeks check (`analysis/greeks_check.py`) — conditionally blocking.** Needed only if the paper
  retains its Greeks claim (the analytic-partial-greeks-at-frozen-σ claim, already corrected once
  from an "exact greeks" overclaim — see `CLAUDE.md`'s "Two claims that were overstated" section
  and prior addenda). If the paper drops the Greeks claim entirely, this becomes optional and can
  move to the nice-to-have list. The script exists but has not been executed on a v5-trained
  model.
- **The "matches or beats a direct-price regressor" claim is currently unsupported.** No
  direct-price-regression baseline has been run to back it. For a first preprint, the recommended
  resolution is to **remove this claim from the manuscript** rather than add another experimental
  workstream.
- **Figures F1–F6 not yet built.** No figure files exist yet under `results/` or elsewhere in this
  repo as of this inventory (checked: no `.png`/`.svg` matching F1–F6 naming). Needed for any
  results/methods section that references a figure.
- **Related-work citations, environment/dependency pinning, and replacing stale v3 material in
  `report.md` with v5** — still necessary, unchanged from prior status.

**Not blocking (reporting constraint, not a missing-work item):**
- **Architecture ranking (A vs B) is underdetermined — not a blocker, provided the results section
  makes no A-vs-B superiority claim.** See §3's caution — call A/B are tied, put B-over-A rests on
  only 3 seeds. If the manuscript states or implies architecture A or B is superior, that claim is
  currently unsupported and must be removed or explicitly qualified as provisional (3 seeds); if
  the manuscript makes no such claim, this item does not block submission.

**Nice to have (do not block a first draft):**
- **Arbitrage-violation-rate diagnostic on the predicted surface not yet run.** A surface-wide
  calendar-spread/butterfly violation-rate diagnostic (to substantiate the "pointwise
  BS-consistent, not arbitrage-free by construction" framing) is speculative/queued, not built.
  See `CLAUDE.md`'s "Two claims that were overstated" section for the framing this diagnostic
  would support. (Distinct from the Greeks check above, which is conditionally blocking.)
- **Throughput numbers are not benchmark-quality — do not cite them for any performance claim.**
  `throughput_contracts_per_s` appears in every `metrics.json` under each `results_*_v5*/`
  directory, but it is a noisy, uncontrolled measurement that varies run-to-run with system load.
  It is explicitly **NOT suitable** for any performance/latency claim in the paper until measured
  under a controlled, repeated-timing benchmark. Flag this so nobody accidentally cites it.

---

## Appendix: file index

| File | What it backs |
|---|---|
| `wrds_data_2020-2025/DATASET_MANIFEST.json` | dataset versions, paths, hashes, canonical flag |
| `wrds_data_2020-2025/VALIDATION_v5.json` | 45/45 v5 validation checks |
| `wrds_data_2020-2025/VALIDATION_v4.json` | 31/31 v4 validation checks |
| `results/table_T1_v5.md` / `.tex` | main results table (T1), 12 canonical seed-42 cells |
| `results/all_metrics_v5.csv` | machine-readable version of T1 |
| `results/replication_seeds_v5.md` / `.csv` | seed-43/44 replication of the 4 vol_surface cells |
| `results/baselines_v5.json` / `.csv` | all baseline records, both option types, all strata |
| `results/data_quality_v5.json` | duplicate-input and static-bound-violation checks |
| `results/diagnostics_tail_v5.json` | T=0 ceiling / conditioning diagnostics |
| `results/QUOTE_QUALITY_REPORT.md` | quote-filter derivation, worst-violation traces, wording rules |
| `PROGRESS_paper_artifacts.md` | addendum log — full narrative history, dated, append-only |
| `PROGRESS_phase2_v3.md` | training-state source of truth |
| `report.md` | paper skeleton / draft prose (next phase, out of scope here) |
