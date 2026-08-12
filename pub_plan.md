# Publication Plan — From Codebase to arXiv / ML-Conference Paper

**Working title:** *Learning Implied Volatility, Not Prices: A Neural-Operator
Encoder with a Differentiable Black–Scholes Decoder for Option Pricing*

**Target:** **arXiv preprint + a finance/ML workshop** — primary path. Good
venues: **ICAIF** (ACM Int'l Conf. on AI in Finance), NeurIPS / ICML workshops
on ML-in-finance or scientific ML, and quant-finance-adjacent workshops. The
arXiv preprint is the durable artifact (CV + interview talking piece); the
workshop submission gets it reviewed fast. **Stretch extensions** (only if the
results are strong): a quant-finance journal (*Quantitative Finance*, *JFDS*),
which weights financial rigor over ML novelty, or a full ML-conference paper.

**Why this target (goal = preparing as a quant researcher).** Quant teams reward
rigorous empirical work on financial data, demonstrated finance understanding
(no-arbitrage, greeks, IV-surface structure, honest error analysis), and a
concrete defensible artifact — *not* a main-conference acceptance line. This
project's strengths (the analytical-BS structural argument, the honest
near-expiry tail analysis, the no-leakage time split) are exactly the
quant-relevant parts and are largely done. So the critical path is **trimmed**:
no second underlying and no exhaustive ablation matrix required to ship. Format
≈ 4–8 pp, lighter on exhaustive baselines, heavy on rigor and honesty.

> **Scope & contribution level (read this and stay honest while drafting).**
> This is an **empirical / application paper**, not a novel-method paper. The
> core idea — encode market state, emit a per-contract σ̂, and price it through a
> closed-form Black–Scholes layer instead of learning the price — is *sound and
> elegant but not novel to the ML community*: differentiable BS layers, learning
> IV instead of price, and analytical-pricer decoders are established in the
> quant-ML literature, and the encoders (FNO, DeepONet) are off-the-shelf. That
> is **fine** — the value here is rigor and honesty (no-leakage time split,
> exact-greeks check, candid analysis of the ill-conditioned near-expiry tail,
> economic vs. statistical significance), which is exactly what a quant audience
> rewards. **Implication:** target a workshop / arXiv preprint, *not* a main ML
> conference (NeurIPS/ICML main track select for transferable methodological
> novelty this paper does not claim). While drafting, **do not over-claim
> novelty** — frame contributions as "a clean, carefully-evaluated study of X,"
> not "we propose a new method." Reviewers penalize manufactured novelty far more
> than honest scoping. (If you ever wanted main-conference caliber, the realistic
> path is a *new* contribution on the open T→0 tail problem — a tail-aware loss,
> reparametrization, or σ̂ uncertainty quantification with theory — which is a
> different, months-longer paper and out of scope here.)

> **Framing decision (read first).** `final_report.tex` documents the
> **superseded v1/v2 physics-informed story** — PDE / arbitrage / boundary
> losses and a DeepONet-vs-FNO loss-weight ablation. The current v3 design
> **deleted all of that**: there is no learned pricer and no PINN loss terms.
> This paper is a **clean reframe**, not an edit of that report. Do **not**
> carry over PINN claims, the loss-weight sensitivity study, or the "physics-
> informed" title. Fork a fresh `paper/` directory.

---

## 1. Thesis & Positioning

**Contribution in one paragraph.** Most neural option pricers learn a map
`(market state, contract) → price`. We instead learn a map to a **per-contract
implied volatility** `σ̂` and feed it through the **closed-form Black–Scholes
formula** to obtain the price. Because BS is built from differentiable
primitives (`exp`, `erf`, `sqrt`), gradients flow from a plain data loss on
`log(V/K)` back through BS into the network — no neural surrogate pricer is
needed. This single architectural choice buys three properties for free:
(i) **no surrogate approximation error** — all error is attributable to `σ̂`;
(ii) **exact greeks** — every greek is the closed-form BS greek at the
estimated `σ̂`; (iii) **arbitrage-free by construction** — a BS price for any
`σ̂ ≥ 0` satisfies the BS PDE and the no-arbitrage bounds, so the auxiliary
PDE/arbitrage penalties that PINN approaches need become unnecessary. The
network's only job is the genuinely hard part: estimating a volatility
**surface** (smile/skew) from market state, which we do per contract query.

**Positioning against the literature.**
- vs **PINN / learned-pricer** approaches (incl. our own v1/v2): we remove the
  surrogate entirely; arbitrage-freeness is structural, not a soft penalty.
- vs **direct price/IV regression**: the BS inductive bias is what makes greeks
  exact and the price arbitrage-free; we ablate this (see §3).
- vs **classic IV interpolation (SVI / splines)**: those interpolate the
  *observed* surface; our encoder learns to map raw market state (incl.
  non-surface inputs like SPX/VIX history) to `σ̂`, and is differentiable
  end-to-end.

**Headline claims to support with evidence.**
1. Per-query `σ̂` recovers the volatility smile/skew that a single-scalar `σ̂`
   cannot (ablation + smile figure).
2. The analytical-BS decoder matches or beats a direct-price regressor of equal
   capacity, while giving exact greeks (baseline + greeks check).
3. Among market-state inputs (IV surface / SPX history / VIX history), which
   prices best — and by how much (branch comparison).

---

## 2. Proposed Paper Structure (~8–9 pp body + appendix)

1. **Introduction.** The pricing problem; the *learn-IV-vs-learn-price*
   dichotomy; why structural arbitrage-freeness + exact greeks matter;
   contributions list (the three headline claims).
2. **Related work.** Neural option pricing; PINNs in finance; neural operators
   (FNO, DeepONet); IV-surface modeling (SVI, SSVI, splines).
3. **Method.**
   - 3.1 The inverse IV problem statement (from `plan_phase2_v3.md` §"Key Idea").
   - 3.2 FNO market-state encoder (4 `SpectralConv2d` blocks + 1×1-conv skips,
     SiLU; 1-D HDF5 vector reshaped to the 2-D grid).
   - 3.3 Per-query `σ̂` readouts — **(A) MLP-head** on `[latent, log_m, T]`;
     **(B) DeepONet** branch·trunk dot product, `σ̂ = softplus(⟨b,t⟩+bias)+floor`.
   - 3.4 The differentiable Black–Scholes decoder; differentiability argument;
     exact-greeks corollary; no-arbitrage-by-construction argument.
   - 3.5 Loss: data MSE on `log(V/K)`; the `1e-8` clamp; *why* no PDE/arb/anchor
     terms (they are redundant given the decoder).
4. **Data.** WRDS / OptionMetrics SPX 2015–2025; `volume>0` filter; three branch
   inputs (IV surface 11×17, SPX OHLCV 21×5, VIX OHLC 21×4); trunk
   `[log_m, T, r, q]`; target `log(mid/K)`; the **date-based 80/10/10
   no-leakage split** (cite `plan_phase1.md` §2.3). Retention/row-count table.
5. **Experiments.** Full matrix (§3 of this plan); baselines; metric definitions.
6. **Results & analysis.** Branch comparison; A-vs-B; the near-expiry log tail
   (the project's defining numerical fact); greeks agreement; efficiency.
7. **Limitations & discussion.** Ill-conditioned `T→0` tail; single underlying
   (SPX); European-style assumption; no transaction costs / bid-ask.
8. **Conclusion.**
- **Appendix.** Numerical stability (`--stable-log`, `log_ndtr`, `log(-expm1)`);
  full hyperparameters; extended tables; reproducibility statement.

---

## 3. Experimental Matrix (what to run)

**Critical path for the workshop/preprint** = §3a (12 runs) + §3b baselines +
§3d greeks + the **per-query-vs-scalar** ablation in §3c. Everything marked
*(stretch)* is for the journal/conference extension and can be cut from v1.

### 3a. Core model runs — 12 total
> **STATUS UPDATE (superseded):** the status table below is stale. All 12 runs are
> now trained — see `report.md` Table T1 for the authoritative results. The
> "10 remaining runs" note and the row-by-row ☐ status no longer apply.

`{call, put} × {vol_surface, spot_history, vix_history} × {MLP-head (A), DeepONet (B)}`.

For every run record on the **test split**: `R²(price)`, **`R²(log, T>1/365)`
(headline)**, `R²(log, full)`, `RMSE(price)`, `RMSE(log)`, mean/median absolute
price error in $, and throughput (contracts/s).

| # | option | branch | A: `train_*.py` | B: `train_*_don.py` | status |
|---|--------|--------|-----------------|----------------------|--------|
| 1 | call | vol_surface  | ☑ done | ☑ done | **trained** |
| 2 | call | spot_history | ☐ | ☐ | to run |
| 3 | call | vix_history  | ☐ | ☐ | to run |
| 4 | put  | vol_surface  | ☐ | ☐ | to run |
| 5 | put  | spot_history | ☐ | ☐ | to run |
| 6 | put  | vix_history  | ☐ | ☐ | to run |

→ **10 remaining runs** (rows 2–6 × {A,B}). Suggested order from PROGRESS:
`call/{spot_history, vix_history}`, then the three `put/*`. Use matched-capacity
overrides (`--latent-dim 128 --head-hidden 128` for A; `--p-dim 128
--trunk-hidden 128` for B). Watch for patience-limited runs; train longer if so.

### 3b. External baselines (new code — see §5)
- **BS constant-σ.** Price each test contract with a single daily σ — (i) ATM IV
  of the day, (ii) VIX — then BS. Establishes the value added by a per-query
  surface. Report same metrics.
- **Classic IV interpolation.** Predict σ for each query from the *observed*
  11×17 surface via (i) 2-D spline / bilinear interpolation, (ii) nearest
  neighbor, and (optionally) (iii) an SVI slice fit per date; then BS price. The
  standard practitioner baseline the model must beat to justify learning.

### 3c. Ablations
- **Per-query σ̂ vs single-scalar σ̂** (the v3 → v3.1 change). The headline
  ablation: a single σ̂ per market state cannot fit the smile. Quantify the
  R²(log, T>1/365) gap and show it visually (F2).
- **Analytical-BS decoder vs direct-price regressor.** *(stretch — strong if
  time allows.)* Same encoder capacity, swap the BS decoder for an MLP that
  predicts `log(V/K)` directly. Isolates the value of the BS inductive bias (and
  shows the regressor cannot give exact greeks / guarantee no-arbitrage).
- **Loss: MSE vs Huber(δ=1.0).** *(stretch.)* One-line toggle already in
  `compute_loss`. Report why MSE is kept (Huber stopped earlier, hurt R²(price),
  didn't move R²(log)).
- **Near-expiry handling.** *(stretch.)* Exclude / down-weight `T < 1/365`
  during training vs not — does removing the ill-conditioned tail from the
  objective help the rest of the domain?

### 3d. Greeks validation
Autograd `∂price/∂S` through the BS decoder vs closed-form `e^{-qT}N(d1)` (put
analogue) on sampled test contracts; expect agreement to ~1e-7. Repeat for vega.
This substantiates the "exact greeks" claim.

---

## 4. Metrics, Tables & Figures

**Standardized metrics** (always on the **test split**): `R²(price)`,
`R²(log, full)`, **`R²(log, T>1/365)` (headline)**, `RMSE(price)`, `RMSE(log)`,
and an **economic-significance** column — mean/median absolute price error in
dollars — because the log metric is dominated by the near-expiry tail whose
*dollar* error is ~1e-6. Always report price-space, log-full, and log-filtered
together (per the project's two numerical facts).

**Tables.**
- **T1 — Main results:** branch × architecture (the 12 runs), all metrics.
- **T2 — Baselines vs model:** BS-constant-σ and IV-interpolation vs best model.
- **T3 — Ablations:** per-query vs scalar; BS-decoder vs direct regressor;
  MSE vs Huber; (optional) near-expiry handling.
- **T4 — Error stratification:** RMSE / R² over a moneyness × maturity grid.

**Figures.**
- **F1 — Architecture diagram:** market-state grid → FNO encoder → per-query σ̂
  (A and B readouts) → analytical BS → `log(V/K)`. *(new, drawn)*
- **F2 — Smile recovery:** predicted vs observed IV across moneyness for sample
  dates/maturities, per-query σ̂ vs scalar σ̂. **The core-claim figure.** *(new)*
- **F3 — Error heatmap:** moneyness × maturity. *(new; eval already computes the
  stratification block)*
- **F4 — Near-expiry log tail:** SSE concentration vs T (the ~1.5% contracts /
  ~93% of SSE story). *(new; tail-diagnostic block has the numbers)*
- **F5 — Predicted vs market price** scatter / reliability. *(new)*
- **F6 — Greeks agreement:** autograd vs closed-form delta/vega. *(new)*
- **F7 — Training curves.** *(`loss_plot.png` already produced per run)*

Note for execution: F7 exists; F3/F4 are computable from the existing
`--eval-only` diagnostic block; F1/F2/F5/F6 need new plotting code.

---

## 5. New Code / Analysis Artifacts

None of these exist yet. Respect the CLAUDE.md constraint that each `train_*.py`
is **self-contained** — put new analysis in a standalone `analysis/` dir that
**loads checkpoints + the HDF5 directly** and re-implements the loader/BS pricer
rather than importing train scripts.

- `analysis/baselines.py` — BS-constant-σ (ATM IV, VIX) and IV-interpolation
  (spline / NN / SVI) baselines over the same HDF5 test split; writes a metrics
  JSON in the same schema as model eval.
- `analysis/eval_to_json.py` (or extend each script's `--eval-only`) — dump the
  per-run metrics to `results_*/metrics.json` so aggregation is mechanical.
- `analysis/aggregate_results.py` — sweep all `results_*/` dirs, read
  `config.json` + `metrics.json`, emit LaTeX for T1–T3.
- `analysis/make_figures.py` — F2–F6 from `best_model.pth` + test split.
- `analysis/greeks_check.py` — autograd-vs-closed-form delta/vega (§3d).
- `analysis/ablation_direct_pricer.py` — the direct-price-regressor variant
  (encoder + MLP head predicting `log(V/K)`, no BS decoder).

**Decision flagged for execution:** start a fresh `paper/` (recommended) rather
than editing `final_report.tex`, given the framing shift.

---

## 6. Reproducibility & Submission Checklist

- **Determinism:** seed 42 fixed; one `config.json` per run already saved.
- **Environment:** conda `dl_new` (torch, polars, numpy, h5py, matplotlib) —
  **pin exact versions** in the appendix / `environment.yml`.
- **Data availability:** WRDS / OptionMetrics is **licensed** — describe the
  access path and the exact filters/secid (SPX `108105`); state that raw data
  cannot be redistributed; release preprocessing code so a WRDS subscriber can
  reproduce the HDF5.
- **Code release:** the repo is currently not under meaningful version control
  (per CLAUDE.md). Prepare a **clean public mirror** (remove legacy
  `OLD_*/`, `archive/`, `train_model/`, `train_model_v2/`, large data) before
  release.
- **arXiv:** category `q-fin.CP` (primary — signals the quant audience),
  `cs.LG` (cross-list); generic preprint template, or the workshop's template if
  it has one.
- **Venue specifics:** check ICAIF / target-workshop deadlines and page limits
  early and work backward from them; preprint to arXiv first (or per workshop
  dual-submission policy).
- **Suggested timeline / ordering (critical path first):**
  1. Finish the 10 remaining model runs (3a).
  2. Implement + run baselines (3b) and the per-query-vs-scalar ablation (3c).
  3. Greeks check (3d).
  4. Aggregate metrics → T1, T2, T4; generate F1–F7.
  5. Draft `paper/` (fresh, not `final_report.tex`); arXiv preprint.
  6. Workshop submission.
  7. *(stretch)* Add the §3c stretch ablations + a second underlying → extend to
     a quant-finance journal or full ML conference.

---

## 7. Open Risks / Decisions for the User

- **Single underlying (SPX).** For the workshop/preprint, **scope explicitly to
  SPX as a stated limitation** — adding a second underlying is the natural
  journal/conference *extension*, not a v1 blocker.
- **Weak branches.** If `spot_history` / `vix_history` underperform
  `vol_surface`, keep them — the negative result *is* the branch-comparison
  contribution (how much market-state information beyond the surface helps).
- **Exercise style.** SPX options are European-style (good — BS applies
  directly); state this explicitly. If any American-style contracts slipped in,
  address the BS-mispricing assumption.
- **Smile-figure data.** Confirm we can reconstruct observed per-contract IV for
  F2 from the HDF5 (the surface `branch_u` is gridded; per-contract observed IV
  may need to be recovered by inverting the market price, or read from a
  retained column).

---

## Reference Files (no changes made by this plan)
- `plan_phase1.md` — data pipeline & split design.
- `plan_phase2_v3.md` — the analytical-BS method (incl. v3.1 per-query σ̂).
- `PROGRESS_phase2_v3.md` — live status: only `call/vol_surface` trained; the two
  numerical facts; CLI flags; how to run.
- `final_report.tex` — **stale** v1/v2 PINN writeup, to be superseded.
- `CLAUDE.md` — self-contained-script constraint, numerical-tail facts, commands.
