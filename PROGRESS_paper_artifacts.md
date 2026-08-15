## -1.8. 2026-08-15 addendum — manuscript readiness inventory corrected

`results/MANUSCRIPT_READINESS_INVENTORY.md` (added in addendum -1.7 below) contained a
put-baseline mischaracterization and a stale-v3-sourced "tied" claim; both are now fixed, plus
§7's open items are reclassified by actual blocking status. No numbers in any `results/*` file
changed — this was a prose/citation fix only. See the inventory file itself for the corrected
text.

---

## -1.7. 2026-08-15 addendum — manuscript readiness inventory added

The full v5 experimental campaign is now complete: canonical dataset (build + 45/45 validation),
all 12 canonical seed-42 cells trained and evaluated, and 8 seed-43/44 replications of the 4
headline `vol_surface` configs trained/evaluated/aggregated (commits `00897f6`, `c6801b5`,
`d5410e5`). A single reference document collecting every file that backs every claim needed to
start drafting the results section now exists: **`results/MANUSCRIPT_READINESS_INVENTORY.md`**.
It covers dataset/preprocessing, T1, replication/robustness (including the corrected "~40x to
just under three orders of magnitude" robustness claim and the architecture-ranking caution),
baselines, data quality/diagnostics, provenance, and a scoped open-items list (figures, greeks/
arbitrage diagnostics, the scalar-σ ablation, and the throughput-noise caveat). Documentation
only — no `.py` files, training runs, or GPU use.

---

# Progress Report — Paper Artifacts (analysis/ build)

**Session date:** 2026-06-12. **Author:** Claude (auto mode). **Status of repo
work (as of the original session):** code written, **nothing executed** (GPU was
occupied; user instructed not to run). This report is for a reviewing agent +
the next working session. **This status line describes only the original
2026-06-12 session** — see the addenda below (newest first) for what has
happened since; as of 2026-08-13 the data-side of the `analysis/` package
(baselines, data quality, diagnostics) has been fully executed on v3, v4, and
the canonical v5 test splits. Only the model-evaluation half
(`eval_to_json.py`, `aggregate_results.py`, `greeks_check.py`) remains
unexecuted, and only because no v4/v5 model exists yet to evaluate.

---

## -1.6. 2026-08-13 addendum — v5 build, validation, and all three CPU analyses COMPLETE

This supersedes every "build in progress" / "not yet built" / "provisional" / "approximate"
qualifier attached to v5 in addendum -1.5 below and elsewhere in this file. The v5 HDF5 build
finished (`wrds_data_2020-2025/deeponet_tensors_{call,put}_v5.h5`) and passed **45/45 validation
checks** (`VALIDATION_v5.json`: the 31 original v4-style checks plus 7 new v5-specific checks, ×2
option types), with **zero remaining static-bound violations, max gap 0.000e+00**. Final row counts
(confirming the approximate float32-based percentages in addendum -1.5 to within ~0.001 pp):

| | Call | Put |
|---|---:|---:|
| Total rows | 4,846,268 | 7,925,349 |
| Train | 3,297,722 | 5,323,327 |
| Validation | 729,204 | 1,221,643 |
| Test | 819,342 | 1,380,379 |
| Removed by filter | 2.2061% (109,324) | 0.8222% (65,705) |

HDF5 SHA-256: call `ecf94de2996e3212ac791f944f847142e7a809c640b24cbd5989e8bdba801153`, put
`46f3ee9d03d3423c6330c2b35a4d09893b0f538bdc046bc1018ebab79861c756`. Split date boundaries are
**bit-identical to v4** (2128/266/267 dates, 2015-02-02 .. 2025-08-29) — the filter removed no trade
date, only individual rows. v4 files re-hashed byte-identical to their original values (call
`5f1c42aa2cc125fa0344b9e3d19fb374278aa629560c420c7ea4c76a6f910546`, put
`2ba5b13b0beb374d86790b6fc94799ea92dbd057454fca4af86f357a1bafbd29`), confirming v4 is unchanged and
retained as the unfiltered robustness sample. `wrds_data_2020-2025/DATASET_MANIFEST.json` now exists
and is populated as the single source of truth for dataset versions, paths, and hashes across the
repo (a lineage bug in it — call/put source-hash mixup — is being fixed concurrently by another
agent; this note only asserts the file exists and is populated, not its internal correctness).
`wrds_data_2020-2025/smoke_v5/` exists (`deeponet_tensors_{call,put}_v5_smoke.h5`) and is
gate-verified — a real training script's `add_provenance()` was called against it directly and
passed.

**All three v5 CPU analyses are complete** (`analysis/{data_quality,diagnostics,baselines}.py
--dataset-version v5`), full test split, no subsampling. *Data quality*
(`results/data_quality_v5.json`): **zero static-bound violations across every maturity bucket (d=2–7
through d>365) and every moneyness bucket (M<0.8 through M>1.2), for both option types**;
duplicate-input rate remains 0.00% (unchanged from v4). *Diagnostics*
(`results/diagnostics_tail_v5.json`): T=0 ceiling still not binding — `n_T0=0`,
`ceiling_binding=False`, `r2_log_full_ceiling=1.0` for both option types (unchanged from v4, since
the maturity floor is unaffected by the quote filter). *Baselines*
(`results/baselines_v5.{json,csv}`): best price-accurate baseline (R²(price)>0.99) by R²(log) —
**call** `vix_level_scaled_pxfit` **+0.7904** (price 0.999809, vs v4's +0.8011/0.999802); **put**
`surface_spline` **−2.5186** (price 0.996594, vs v4's −2.4688/0.997004). **v5's numbers are slightly
worse than v4's on both sides — this is understood and expected, not a data-quality regression:** the
rows the filter removed carried roughly twice their row-share of total variance (call −2.52% of
rows / −5.16% of SST; put −0.73% of rows / −1.41% of SST) — removing the most extreme mispriced rows
shrinks the R² denominator faster than the numerator, so v5 is a genuinely harder benchmark on the
same metric than v4, not an apples-to-apples "did it get better" comparison. State this explicitly
wherever v4-vs-v5 baseline numbers are compared. The same scalar-VIX anti-inference caveat as v4
applies.

**Test suite:** 292 tests pass as of this writing (may have grown further from the concurrently-
running provenance-fix and manifest-fix agents — say "292+ as of the last full run" rather than a
stale exact number if citing later). `git diff --check` clean.

**What remains genuinely pending — unchanged by this note.** No neural model has been trained on v4
or v5; every R²/RMSE/MAE number attached to a trained model anywhere in these docs remains
v3-trained, historical, and not comparable to v4 or v5. The 12 core training runs (canonical target
now v5), the seed-43/44 replication for the four headline `vol_surface` configs, the
`put/spot_history` DeepONet rerun, the per-query-vs-scalar ablation, the greeks-check execution,
figures F1–F6, and the surface-wide arbitrage violation-rate diagnostic are all still pending GPU
work. **Only canonical v5 neural training remains before the dataset/analysis side of the paper is
complete** — not any remaining data-quality question.

Files touched by this addendum: `report.md`, `PROGRESS_phase2_v3.md`, `PROGRESS_paper_artifacts.md`
(this file), `CLAUDE.md`. Documentation only — no `.py` files, training runs, or GPU use;
`analysis/eval_to_json.py`/the 12 training scripts and `wrds_data_2020-2025/make_dataset_manifest.py`
were being edited concurrently by other agents (provenance-integration and manifest-lineage fixes
respectively) and were left untouched by this pass.

---

## -1.5. 2026-08-12 (later) addendum — quote-quality filter approved; v5 supersedes v4 as training target

`analysis/quote_quality.py` (`results/QUOTE_QUALITY_REPORT.md`) evaluated a deterministic,
zero-tolerance static no-arbitrage midpoint filter against the full v4 dataset and it has now been
**approved**: drop any row whose midpoint, in normalized `V/K` units, falls outside
`[max(fwd−disc,0), fwd]` (calls) / `[max(disc−fwd,0), disc]` (puts), computed in float64 before the
float32 HDF5 cast. Measured on the float32 v4 export (approximate — the v5 float64 recomputation may
shift these): call ~2.206% removed, put ~0.822% removed; train/test differential negligible
(+0.64 pp call, −0.16 pp put), which is why this rule and not a spread-based one was chosen (spread
rules shift 1.9–4.4× more of train than test — secular quote-tightening 2015→2025 — which would
distort the evaluation distribution).

**Two wording constraints that apply to any prose drawing on this filter** (both were corrected in
`results/QUOTE_QUALITY_REPORT.md` this pass, see its §2.3 and its "Why this one" paragraph):
1. The `impl_volatility` NULL agreement (~98% of violations) is **corroborating, not independent**,
   evidence — OptionMetrics' own inversion fails for the same underlying reason (inconsistent
   midpoint/carry inputs), so it is the same evidence seen twice, not a second witness.
2. Do not say "provable market arbitrage" or "provably unfittable" unqualified — the supplied `r`,
   `q`, and settlement inputs are estimates, not observed riskless trades. The defensible claim is
   "midpoints inconsistent with the static bounds under the supplied spot, rate, dividend-yield, and
   settlement inputs" — i.e. the BS decoder cannot reach these prices at any σ̂ **given those
   inputs**.

Removal is concentrated (deep-ITM calls `M>1.2` ~34% of that bucket, deep-OTM puts `M<0.8` ~39%,
2–7-day calls ~11.6%), though those buckets are small in absolute share (2.06%/0.30% of their option
type) and the largest absolute contributors are moderately-ITM and near-ATM calls — see the quote-
quality report §3 for the full breakdown; any manuscript prose citing the headline 2.2%/0.8% must
also disclose this concentration.

**Dataset versioning:** the filtered sample is **v5** (`schema_version` stays `"v4"`, new attr
`dataset_version = "v5"`), files `wrds_data_2020-2025/deeponet_tensors_{call,put}_v5.h5`. **v4 is
retained**, relabeled "unfiltered robustness sample; noncanonical for training" (not invalid). All 12
training runs will target v5 once built; `DATASET_MANIFEST.json` (queued) becomes the single source
of truth for versions/paths/hashes. **The v5 build, `VALIDATION_v5.json`, and the v5 re-runs of
baselines/diagnostics/data-quality are in progress as of this note** — this changes WS3's eventual
target (baselines will need a v5 re-run once v5 exists) but **does not change WS3's completion status
on v4**, which stands as recorded in addendum #2 below and is retained as the robustness-sample
record. Do not cite v5 row counts or hashes beyond the approximate removal percentages above until
the build completes and reports final numbers.

A quote-liquidity stratification (`best_bid > 0`, relative spread ≤ 1) is being added as *reporting
strata only*, not a second filter, so a v5-trained model's results can be shown not to depend on the
retained zero-bid/wide-spread quotes.

Files touched by this addendum: `report.md`, `PROGRESS_phase2_v3.md`, `PROGRESS_paper_artifacts.md`
(this file), `CLAUDE.md`, `results/QUOTE_QUALITY_REPORT.md`. Documentation only — no `.py` files,
training runs, or GPU use; `analysis/*.py`, `train_model_v3/**`, `wrds_data_2020-2025/**`, and
`tests/**` were being edited concurrently by other agents and were left untouched by this pass.

---

## -1. 2026-08-12 addendum — v4 dataset build completed (canonical dataset switch)

The blocker referenced throughout this file's §0 item 1 and item 7 ("pending on the corrected v4
HDF5 the preprocessing agent is building") is resolved: the v4 build finished 2026-08-12 09:14 and
passed validation. **v4 is now the canonical dataset.** This does not change anything about the
`analysis/` package's execution status (still not run, see §0 item 5/§2) — it changes what the
package must eventually be run *against*.

**Build facts** (full detail in `PROGRESS_phase2_v3.md`'s "2026-08-12: v4 dataset build completed"
section; do not re-derive):

| | Call | Put |
|---|---:|---:|
| Total rows | 4,955,592 | 7,991,054 |
| Train | 3,360,823 | 5,370,629 |
| Validation | 754,276 | 1,229,955 |
| Test | 840,493 | 1,390,470 |
| Unique dates | 2,661 | 2,661 |

Coverage 2015-02-02 .. 2025-08-29; files `wrds_data_2020-2025/deeponet_tensors_{call,put}_v4.h5`
(8.06 GB / 13.00 GB); HDF5 SHA-256 abbreviated call `5f1c42aa…10546` / put `2ba5b13b…bd29`, full
hashes in `wrds_data_2020-2025/VALIDATION_v4.json` (cite, do not invent). Source VIX CSV SHA-256
(full): `e2f18c516c9d48fb1730c2c0203fce36900c70aa0635b99e64afdde157346301`. **Validation: 31/31
checks passed** (`validate_v4.py` → `VALIDATION_v4.json`).

**Maturity decision.** Canonical cutoff `--min-maturity-days 1.0` on settlement-aware maturity:
"contracts with strictly more than 24 hours remaining to settlement." Observed minimum 1.729 days
(AM-settlement 6.5h offset, expected not anomalous). Do not re-cut v4 to imitate v3's `float32(1/365)`
rounding accident (which only ever dropped exact `T=0`, not genuine 1-day contracts) — on v4,
`T > 1/365` is a no-op, so headline and full-domain `R²(log)` are the same number on v4.

**Why this reshapes the `analysis/` workstreams in this file:**
- **WS3 (baselines, §4 below)** was explicitly "blocked on v4" in this file's history and marked
  "pending re-run on the v4 sample" throughout §0 item 1. That blocker is **resolved and the work is
  done**: `analysis/baselines.py` has been run on the full v4 test split → `results/baselines_v4.*`
  (2026-08-12). Results and the three interpretation rules are in addendum #2 below; the tenor/delta
  axis recovery this file documents (§3.1) was completed as part of that run. ⚠️ Note the scalar
  `vix_level_*` baselines are **not** a stand-in for the `vix_history` neural branch — see
  addendum #2 before citing them.
- **The verification gates in §2** still apply to the v3-trained `results_*` dirs as written — they
  are a code-correctness check, not a dataset claim, so they should still be run first regardless of
  which dataset produced the weights being checked.
- **All 12 core model runs must be retrained on v4** before `aggregate_results.py`'s output is
  canonical (§4/§5.1 of `PROGRESS_phase2_v3.md`; the reasoning — three simultaneous dataset changes
  that would confound a partial v3/v4 table — is spelled out there and in `report.md`'s top
  correction note). This is not a change to the `analysis/` code itself, only to what input it will
  eventually run against.
- **No v4 model metrics exist.** Nothing in this addendum states or implies a v4 R²/RMSE/MAE number;
  none have been produced. **Update (2026-08-12, later same day): baselines/diagnostics/data-quality
  analyses on v4 have now completed** — see the new "2026-08-12 addendum #2" section below for the
  full results. This does not change the "no v4 model metrics" statement above: WS4 (per-query vs
  scalar-σ̂ ablation) and the 12 core retraining runs are still pending GPU work.

Files touched by this addendum: `report.md`, `PROGRESS_phase2_v3.md`, `PROGRESS_paper_artifacts.md`
(this file), `CLAUDE.md`, `wrds_data_2020-2025/README.txt`,
`train_model_v3/_vix_is_vvix_LEGACY/DEFERRED_GPU_COMMANDS.md`. Documentation only — no `.py` files,
training runs, or GPU use.

---

## -0.5. 2026-08-12 addendum #2 — v4 CPU analyses (baselines, data quality, diagnostics) completed

All three ran on the **full v4 test split** (call 840,493 rows / put 1,390,470 rows, 267 test dates,
2024-08-07 .. 2025-08-29). Outputs: `results/{data_quality_v4.json, data_quality_{duplicates,
bounds}_v4.csv, diagnostics_{tail,strata,conditioning}_v4.{json,csv}, baselines_v4.{json,csv}}`. The
v3 outputs are preserved untouched. **This is baselines and data-diagnostics only — no v4 model has
been trained; WS4 (ablation) and the 12-run v4 retraining matrix are unaffected and still pending.**

**Data quality — v3 → v4:**

| | v3 | v4 |
|---|---:|---:|
| Duplicate-input rows, call | 17.82% | **0.00%** |
| Duplicate-input rows, put | 19.81% | **0.00%** |
| Max duplicate group size | 2 | **1** (every row unique) |
| Static-bound violations, call | 4.36% | **2.5165%** |
| Static-bound violations, put | 1.45% | **0.7257%** |
| Violations at `T ≤ 1 day` | 31–33% | **0.00%** |

All remaining v4 violations are lower-bound (100%), none in the near-expiry bucket. Severe outliers
persist on v4 (call bound-violation depths 6.49 and 6.33 in `V/K` units, traced to one-sided stale
quotes such as 2025-05-15 SPX Dec-2030 C400 with bid 13.0 / ask 5433.7 and NULL vendor IV) — a
deterministic quote-quality filter is being evaluated separately and **must be decided before v4
training**; this is an open item, not resolved.

**Diagnostics — the T=0 decoder degeneracy is structurally eliminated on v4.** `n_T0 = 0` and
`pct_T0 = 0.0` for both option types; `ceiling_binding = False`; **`r2_log_full_ceiling = 1.0`** for
both. On v3 that ceiling capped `R²(log, full)` at 0.9014 (call) / 0.8498 (put) — a bound no model
could cross, regardless of architecture. On v4 there is no such bound. Also note: on v4 the
`T > 1/365` filter is a **no-op** (minimum observed maturity is 1.729 days), so "headline" and
"full-domain" `R²(log)` are the same number on v4 (this was already documented in
`PROGRESS_phase2_v3.md`'s 2026-08-12 section; confirmed here by the diagnostics run).

**Baselines on v4 (full test split), interpretation rules that must be followed when citing these:**
1. `vix_level_*` uses the **contemporaneous scalar VIX close** — describe it as a *contemporaneous
   scalar volatility benchmark*, and do not read its ranking as saying anything about the
   `vix_history` **neural branch** (a 21-day×4 OHLC tensor, close-normalized, a different input with
   different information content). An earlier informal summary made exactly this conflation error;
   do not reproduce it.
2. On puts, never state a bare "best baseline" — always carry both metrics. Least-negative log-R² is
   `vix_level_scaled` (−0.3336) but its price-R² is a catastrophic −3.245063; the best
   *price-accurate* baseline is `surface_spline` (log-R² −2.4688, price-R² +0.997004).
3. On calls, the highest log-R² baseline is `vix_level_scaled_pxfit` at **+0.8011** (not the plain
   `vix_level_scaled` +0.7974) — subject to caveat 1 above. `surface_spline` has the best call
   **price**-R² (0.999852) while ranking third on log-R² (+0.7764).

Full call ranking by log-R²: `vix_level_scaled_pxfit` +0.8011 (price 0.999802) > `vix_level_scaled`
+0.7974 (0.999796) > `surface_spline` +0.7764 (**0.999852**) > `surface_bilinear` +0.7763 >
`surface_nearest` +0.7751 > `prev_day_surface_bilinear` +0.7395 > `intrinsic_zero_vol` −15.8866
(0.999449). Full put ranking by log-R²: `vix_level_scaled` −0.3336 (price **−3.245063**) >
`const_sigma` −0.3665 (−1.920967) > `surface_spline` −2.4688 (**+0.997004**) > `intrinsic_zero_vol`
−33.7053 (0.602701).

Files touched: `report.md` (§ top correction note, §4.2, §5.2, §6, §7), `PROGRESS_phase2_v3.md`,
`PROGRESS_paper_artifacts.md` (this file). Documentation only — no `.py` files, training runs, or
GPU use; `analysis/*.py` and `tests/**` were being edited concurrently by other agents and were left
untouched by this pass.

---

## 0. 2026-08-11 addendum — verified findings + documentation corrections

A separate documentation-only pass (no `.py` files touched, nothing executed, no GPU use) corrected
several claims that had drifted ahead of what was actually verified, and recorded new findings that
were verified directly against the HDF5 test splits by an independent reviewer/agent. Summary (full
detail lives in `report.md` §6/§7 and `PROGRESS_phase2_v3.md`'s "2026-08-11" sections):

1. **Call vs put R² — corrected framing.** Zero-param intrinsic/zero-vol baseline: call R²(price) =
   0.999466, put = 0.632539. BS constant-σ baseline: call = 0.999811 (σ=0.15), put = 0.946693
   (σ=0.20). Best neural: call = 0.999854, put = 0.997241. **Call R²(price) is near-vacuous** — the
   zero-parameter baseline alone reaches 0.9995 and ties/beats several of the 12 runs. **Put
   R²(price) is informative against naive baselines but not against IV-surface interpolation**,
   which an independent reviewer measured at put R²(price) ≈ 0.997208 — essentially tied with the
   neural 0.997241. This inverts what `report.md`'s original "Calls vs puts" paragraph said. The
   BS-constant-σ / IV-interp numbers here were reviewer-supplied and have since been
   **reproduced in-repo (2026-08-12)** by `analysis/baselines.py` on the full v3 test split →
   `results/baselines_vix.{json,csv}`. Caveat: they are **v3-sample** numbers (calendar-day
   maturity, T=0 rows present, AM/PM contract inputs collapsed) and must be regenerated on the v4
   sample before use as canonical manuscript values.
2. **"Arbitrage-free by construction" is false.** Corrected everywhere in `report.md` to "pointwise
   Black–Scholes-consistent": feeding a predicted σ̂(K,T) through BS gives valid prices at each
   queried point (static bounds respected) but does **not** guarantee calendar-spread or butterfly
   consistency across the surface. A violation-rate diagnostic on the predicted surface is queued,
   not built.
3. **"Exact greeks" is overstated.** Corrected everywhere in `report.md` to "analytic BS partial
   greeks at the predicted σ, holding σ fixed". `analysis/greeks_check.py` already documents this
   caveat correctly in its own source comments (lines 83–87) — only the `report.md` prose overclaimed
   it; the script itself needs no code change.
4. **Near-expiry tail — mechanism corrected, not just re-described.** The original narrative ("~1.5%
   of contracts carry ~93% of log-space SSE because T→0 is ill-conditioned") conflated two distinct
   effects. The **dominant** one is exact-`T=0` **decoder degeneracy**: `T_years =
   calendar_days_to_expiry / 365` gives contracts quoted on their own expiry date exactly `T=0`, and
   the BS decoder's `sqrt(clamp(T, 1e-10))` then returns intrinsic value independent of σ̂ — a
   structural, not numerical, unfittability. Verified irreducible SSE floor: call 9.86% of SST
   (38,765 rows, 4.28%), put 15.02% of SST (39,260 rows, 2.68%); R²(log, full) ceilings 0.9014
   (call) / 0.8498 (put) vs currently reported 0.894105 / 0.832595 — the trained models are already
   within ~0.007–0.017 of the ceiling. Genuine small-but-positive-T ill-conditioning is real and
   secondary, not retracted. **The headline `R²(log, T>1/365)` is unaffected** (0.9924 call /
   0.9817 put stand). Remediation is queued: exclude unresolved expiry-day rows, or derive a
   defensible intraday maturity from quote/settlement timestamps — explicitly not an epsilon added
   to `T`.
5. **Analysis pipeline status clarified.** `analysis/_common.py`, `eval_to_json.py`,
   `aggregate_results.py`, `greeks_check.py` exist (written 2026-06-12, this file's own session) but
   have never been executed — zero `metrics.json` anywhere in the repo, no canonical aggregated
   results. `report.md` previously said the greeks script was "not built"; that was stale and is
   fixed. Every number currently in `report.md` is hand-copied from a `loss_history.txt` tail, not
   generated by these scripts — flagged explicitly wherever a number appears.
6. **Additional verified facts recorded (not previously in any doc):**
   - All 12 runs use **seed 42** (verified in all 12 `config.json`). The "A/B tie within seed noise"
     phrasing in `report.md`/`PROGRESS_phase2_v3.md` is **not supported by a single seed** — downgraded
     to "close on this seed, untested hypothesis pending 3-seed re-runs."
   - **Duplicate/unidentifiable inputs:** preprocessing drops `symbol`, `optionid`, `am_settlement`,
     so AM-settled SPX and PM-settled SPXW contracts can collapse to identical model inputs. 17.82%
     of call test rows / 19.81% of put test rows sit in duplicate-input groups; 97.9%/96.3% of those
     pairs carry different market prices — a real identifiability problem. But the induced
     irreducible error floor is only 0.007%/0.006% of test variance — not a significant source of
     reported model error. Recorded as a disclosure issue, not overstated as an error source.
   - **BS static-bound violations:** 4.36% call / 1.45% put test rows, median depth ~3.6e-4 in `V/K`
     units, 31–33% at `T ≤ 1 day` — consistent with a close-time vs settlement-time mismatch, not
     wholesale data corruption. One outlier (call violation depth 6.49) under separate investigation.
   - **Branch-contribution reframing is a hypothesis, not adopted.** Put/vol_surface being close to
     IV-surface interpolation raises the possibility that the paper's real contribution is
     `spot_history`/`vix_history` pricing *without* a contemporaneous IV surface (making
     `vol_surface` a sanity check, not the headline). Recorded in `report.md` §7 explicitly as an
     **untested hypothesis** conditional on per-branch baselines — the paper has not been reframed
     around it. (The `vix_history` half of this hypothesis is now additionally suspended — see item
     7.)
7. **VVIX quarantine (2026-08-11) — `vix_history` trained runs are not VIX results.** The
   `vix_history` branch was trained on **VVIX** (CBOE volatility-of-VIX, secid `152892`, from the
   mislabeled `vvix_2015_2025.csv`), not VIX. The four trained run dirs (`call|put ×
   results_vix_history[,_don]`) are quarantined unchanged under
   `train_model_v3/_vix_is_vvix_LEGACY/` (with `MIGRATION_NOTE.md` + `DEFERRED_GPU_COMMANDS.md`);
   `analysis/aggregate_results.py` now filters them out and warns if they are present. Their T1 rows
   in `report.md`/`PROGRESS_phase2_v3.md` are marked **pending retraining** and any ranking claims
   citing `vix_history` numbers are suspended. **`vix_history` baselines are being recomputed from
   real VIX by the baselines agent** (`analysis/baselines.py`), on top of the corrected v4 HDF5 the
   preprocessing agent is building. The four corrected VIX model runs are queued for GPU retraining
   (commands in the quarantine dir).

Files touched in this addendum: `report.md`, `PROGRESS_phase2_v3.md`, `PROGRESS_paper_artifacts.md`
(this file), `CLAUDE.md`. No `.py` files, `pub_plan.md`, training runs, or GPU use were touched —
this was a documentation-only correction pass ahead of a preprint.

---

Goal: finish the publication critical path in `report.md` (see `pub_plan.md` §6).
The 12 core runs are trained; what remained is all the `analysis/` code, the
baselines/ablation runs, figures, and prose. Plan of record:
`C:\Users\iamlz\.claude\plans\i-want-to-continue-mighty-bunny.md`.

---

## 1. What was built this session (NEW, unvalidated — needs a run to confirm)

All under a new self-contained `analysis/` package (respects the CLAUDE.md
"each script re-implements the stack" constraint — it imports only from a shared
`analysis/_common.py`, never from the train scripts).

| File | Purpose | Workstream |
|---|---|---|
| `analysis/_common.py` | Shared: FNO encoder + both readouts (A `VolEstimatorModelA`, B `VolDeepONetModelB`), BS pricer (stable + clamped), HDF5 test-split loader (reads the extra `moneyness`/`normalized_price`/`date`/`branch_u` columns the train loader skips), `compute_metrics` canonical schema, `load_run` (config→model→weights, `strict=True`). | WS1 base |
| `analysis/eval_to_json.py` | Re-runs test eval per `results_*/` dir, writes `metrics.json` in the canonical schema. `--all` sweeps all 12, `--max-contracts N` smoke. | WS1 |
| `analysis/aggregate_results.py` | Sweeps all `metrics.json`, emits `results/all_metrics.csv`, `results/table_T1.md`, `results/table_T1.tex`. Makes the T1 table reproducible. | WS1 |
| `analysis/greeks_check.py` | Autograd ∂price/∂S and vega through BS vs closed-form at the model's σ̂; writes `results/greeks_check.json` + `greeks_check_arrays.npz`. Cheap/CPU. | WS5 |

**Design choices baked in (so the reviewer can sanity-check them):**
- Model classes mirror the train scripts' module structure so `state_dict` keys
  match exactly: A → `encoder.*`, `head.*`; B → `branch.*`, `trunk.*`,
  `sigma_bias`. One `FNO_MarketEncoder` serves both, but the **final activation
  differs** and is set per readout via `final_silu`: model A applies
  `F.silu(fc2(x))` (`train_*.py:133`), model B returns the raw `fc2(x)`
  (`train_*_don.py:133`) because the DeepONet dot product needs a signed branch
  basis. (Projection width also differs: `latent_dim` vs `p_dim`.) **This was a
  bug in the first cut of `_common.py` — it applied SiLU unconditionally, which
  silently corrupts all six model-B evals while still passing `strict=True` load
  and the model-A gate. Fixed; see the model-B gate below.**
- Arch (A vs B) is inferred from the results-dir suffix `_don` (and a `p_dim` key
  fallback) in `_common.infer_arch`.
- `compute_metrics` is a line-for-line port of the train-script eval block
  (`train_vol_surface.py:613-691`): same R² definitions, same `1e-10`
  denominators, same `T>1/365` filter, same `sq.err>10` tail accounting. This is
  what makes the **verification gate** below meaningful.
- `config.json` stores an **absolute** `h5_path` from the training machine;
  `_common.resolve_h5_path` falls back to the repo-relative
  `wrds_data_2020-2025/deeponet_tensors_{type}.h5` if that path is absent.

---

## 2. Verification gates (RUN THESE FIRST when the GPU frees)

Nothing here has been executed. Before trusting any number:

```bash
conda activate dl_new
# 1. Consistency gate — must reproduce the known T1 row:
python analysis/eval_to_json.py train_model_v3/call/results_vol_surface
#    EXPECT  R2(price) ~= 0.999851   R2(log,T>1day) ~= 0.992481  (model A)
#    (these are in train_model_v3/call/results_vol_surface/loss_history.txt)
# 1b. Model-B gate — the A gate CANNOT catch the model-B encoder (final_silu)
#     bug, so run a _don dir too:
python analysis/eval_to_json.py train_model_v3/call/results_vol_surface_don
#    EXPECT  R2(log,T>1day) ~= 0.992413  (model B, from report.md T1).
#    If this is far off, the FNO branch's final activation is wrong (must be
#    final_silu=False for model B).
# 2. If both match, sweep everything:
python analysis/eval_to_json.py --all
python analysis/aggregate_results.py        # -> results/table_T1.md
# 3. Greeks gate — max abs diff should be ~1e-6 or smaller:
python analysis/greeks_check.py
```

If gate 1 does **not** reproduce the row, the most likely culprits are: (a) a
`state_dict` key mismatch (compare `torch.load(best_model.pth).keys()` to the
model's `state_dict().keys()`), or (b) metric drift (the train block casts to
numpy float32 mid-stream; `compute_metrics` uses float64 — differences should be
< 1e-5, not material, but confirm).

---

## 3. Open questions RESOLVED this session (important for WS3/WS6)

Read from `wrds_data_2020-2025/pre_process_{1,2}_hdf5.py`:

1. **`branch_u` is a tenor × delta IV surface, NOT moneyness × maturity.** It is
   OptionMetrics' standardized vol surface: 11 tenors (`days`) × 17 deltas, built
   by sorting `["date","cp_flag","days","delta"]` and aggregating
   `impl_volatility` (`pre_process_1_data.py:86-102`). **Consequence:** the
   IV-interpolation baseline (WS3) cannot bilinearly interpolate in
   `(moneyness, T)` directly — it must either (i) map each contract's
   `(moneyness, T)` to `(delta, days)` first (delta needs a σ, i.e. circular /
   needs a BS delta calc), or (ii) interpolate on the `(days, delta)` grid after
   computing each contract's BS delta. The grid's exact tenor/delta axis values
   are **not stored in the HDF5**; recover them from the raw OptionMetrics surface
   table (unique `days`, unique `delta`) when building the baseline.
2. **There is NO per-contract observed-IV column in the HDF5.** Columns are
   `branch_u, spot_history, vix_history, trunk_y[log_m,T,r,q], target_v_log,
   moneyness, normalized_price, date, split_id` (`pre_process_2_hdf5.py:128-173`).
   **Consequence for figure F2 (smile recovery — the core-claim figure):**
   observed per-contract IV must be **recovered by BS-inverting
   `normalized_price`** (Brent/Newton on `bs_normalized_price - mkt = 0`). Plan
   for this; it's the one figure with a real data dependency.
3. **Scripts are import-safe** (`if __name__ == "__main__"` guards), but each
   redefines identically-named classes — hence the single clean copy in
   `_common.py` rather than importlib gymnastics.

---

## 4. Remaining workstreams — specs for the next session

### WS2 — Re-run put/spot_history DeepONet outlier (GPU; user runs)
`put·spot_history·B` has R²(price)=0.837 (outlier vs A's 0.963). Re-run:
```bash
python train_model_v3/put/train_spot_history_don.py --p-dim 128 --trunk-hidden 128
```
Then `eval_to_json.py` on its dir; if still low after a clean run, keep it and
report honestly as a B-instability on the weak branch. (Consider more patience /
longer fine-tune if it again early-stops.)

### WS3 — External baselines `analysis/baselines.py` (✅ **complete** — written + run on the v3 test split 2026-08-12 → `results/baselines_vix.*`; on the v4 sample 2026-08-12 → `results/baselines_v4.*`; and on the **canonical v5** sample 2026-08-12/13 → `results/baselines_v5.*`; see addendum -1.6 and the "2026-08-12 addendum #2" section near the top of this file)
Same test split, **same `compute_metrics` schema**, writes
`results/baselines/<name>_metrics.json`. Two families:
- **BS constant-σ:** (i) ATM IV of the day, (ii) VIX. ATM IV: needs the δ≈0.50
  column at the nearest tenor from `branch_u` → **requires the delta axis** (see
  §3.1). VIX: from `vix_history` last close — **confirm its scale** (vol points
  vs decimal; divide by 100 if in points) before pricing.
- **IV interpolation:** σ per query from the observed `branch_u` surface via
  bilinear / nearest-neighbor on the `(days, delta)` grid (see §3.1 mapping
  caveat). SVI per-date = `(stretch)`, skip for v1.
**Blocked on:** recovering the tenor/delta axes + confirming VIX scale. I did not
write this blind because the interpolation geometry is wrong if the axes are
assumed. Inspect `wrds_data_2020-2025/` raw surface table first.

### WS4 — Per-query vs scalar-σ̂ ablation (GPU; the headline ablation)
For a **fair** comparison it must reuse the exact two-phase schedule. Cleanest
path: copy `train_model_v3/call/train_vol_surface.py` →
`analysis/ablation_scalar_sigma.py` (or `train_model_v3/call/train_vol_surface_scalar.py`)
and make ONE surgical change — replace `ConditionalVolHead` with a scalar head
that ignores `(log_m, T)`:
```python
class ScalarVolHead(nn.Module):           # one σ̂ per market state (the old v3)
    def __init__(self, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, latent, log_m, T):    # log_m, T accepted but UNUSED
        return F.softplus(self.net(latent))
```
Train on `call·vol_surface` and `put·vol_surface` (strongest branch). Expect a
large R²(log,T>1day) gap. Dump per-contract `(moneyness, σ̂_scalar)` for F2.
I specced rather than cloned 800 lines blind — a clone can't be smoke-tested now
and the schedule must match exactly to be a fair ablation.

### WS6 — Figures `analysis/make_figures.py` (NOT yet written)
- **F3 error heatmap (moneyness×maturity)**, **F4 SSE-vs-T tail**, **F5 pred-vs-
  market scatter** — all need only per-contract `(log_pred, target, T, log_m)`.
  Suggest adding a `--dump-arrays` flag to `eval_to_json.py` that saves these to
  `results_*/preds.npz`, then F3/F4/F5 read that. Low risk; write next.
- **F6 greeks agreement** — reads `results/greeks_check_arrays.npz` (already
  produced by WS5). Ready to plot.
- **F2 smile recovery** — needs WS4 output + observed IV via BS inversion (§3.2).
  The core figure; do it after WS4.
- **F1 architecture diagram** — hand-drawn (tikz/draw.io), not code.

### WS7 — Fill `report.md`
Replace each `> **TODO**` block with backed numbers as workstreams land: T1 from
`aggregate_results.py`; T2 from baselines; T3 from the scalar ablation; T4 from
F3's grid; greeks line from WS5. Write §2 Related Work prose + citations. Update
the abstract's baseline-gap TODO once T2 exists. Keep `report.md` as the working
draft (user's choice); fork a fresh `paper/` LaTeX only at the end — do **not**
edit the stale `final_report.tex` (PINN story). Honesty constraint: no number in
`report.md` without a backing file under `results/`.

---

## 5. Suggested order next session
1. GPU free → run §2 verification gates; fix any state_dict/metric mismatch.
2. `eval_to_json.py --all` + `aggregate_results.py` → refresh T1.
3. `greeks_check.py` → greeks line + F6.
4. Inspect raw surface for delta/tenor axes + VIX scale → write/run `baselines.py` (T2).
5. Write/run the scalar-σ̂ ablation (T3) + WS2 outlier re-run.
6. `make_figures.py` (F3/F4/F5, then F2) + dump-arrays flag.
7. Fold into `report.md`; Related Work; finalize.

## 6. For the reviewing agent — what to scrutinize
- `_common.py` model classes vs the real `state_dict` keys (the consistency gate
  catches this, but a static review helps).
- `compute_metrics` parity with `train_vol_surface.py:613-691` (esp. the price-R²
  denominator and the `T>1/365` mask).
- Whether `greeks_check.py`'s `delta := dc/dM` convention is the one the paper
  wants (vs ∂C/∂S in dollar terms) — stated at the top of that file.
- The §3 data findings (tenor×delta surface; no observed-IV column) — these
  reshape WS3/WS6 and should be confirmed against the raw data before coding.
