# Progress Report — Paper Artifacts (analysis/ build)

**Session date:** 2026-06-12. **Author:** Claude (auto mode). **Status of repo
work:** code written, **nothing executed** (GPU was occupied; user instructed not
to run). This report is for a reviewing agent + the next working session.

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

### WS3 — External baselines `analysis/baselines.py` (NOT yet written)
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
