# Deferred GPU commands — the full v5 retraining matrix (all 12 core runs)

> **Scope changed 2026-08-12 (twice).** This file used to cover only the 4 `vix_history` reruns, then
> all 12 runs on **v4**. It now targets **v5**, the quote-filtered canonical sample, for all 12 core
> runs (`{call,put} × {vol_surface, spot_history, vix_history} × {MLP-head A, DeepONet B}`) plus the
> seed-replication extension. It still lives inside the VVIX quarantine directory for continuity of
> history — the rebuild started here — but it is no longer a VIX-only document.

**NOTHING IN THIS FILE MAY BE LAUNCHED YET. Two preconditions, both currently unmet:**

1. **`wrds_data_2020-2025/deeponet_tensors_{call,put}_v5.h5` must exist and be validated.** They are
   being built now (zero-tolerance static no-arbitrage midpoint filter applied to v4). Every command
   below points at them; until they land, all 12 scripts abort at startup by design.
2. **The GPU must be confirmed free.** It is **currently busy** (~1986 MiB in use, verified
   2026-08-12). Do not launch on a shared GPU — check `nvidia-smi` first.

- **Agreed run order:** (1) the 12 one-batch smoke jobs in §1 below, (2) confirm provenance on the
  smoke runs and delete the `*_v5_smoke` dirs, (3) the matched 12-model seed-42 v5 matrix (§2), (4)
  eval + aggregate (§5), (5) decide on seed-43/44 repetitions (§3) from how the seed-42 matrix comes
  out — do not pre-commit to §3/§4 before §2's results are in.

Run everything **from the repo root** (`DL_optionPricing/`).

## Why v5 is the canonical sample (do not lose this)

v4 fixed the *inputs* (real CBOE VIX, settlement-aware maturities, `T > 1 day` export filter, AM/PM
contract identity). v5 fixes the *targets*: a zero-tolerance static no-arbitrage midpoint filter
drops rows whose `mid/K` falls outside the static bounds implied by the supplied
spot/rate/dividend/settlement inputs (call ≈2.2% of rows, put ≈0.8%). Those quotes are unreachable
by the Black–Scholes decoder at **any** σ̂ — they are an irreducible error floor, not signal.

The tensor **schema is unchanged**, so the file attrs separate the two axes:

| attr | value | meaning |
|---|---|---|
| `schema_version` | `"v4"` | tensor structure — unchanged by v5 |
| `dataset_version` | `"v5"` | **sample definition** — the filtered sample |
| `quote_filter` | `"static_bounds_midpoint"` | which filter |
| `quote_filter_tolerance` | `0.0` | zero tolerance, as approved |

The unfiltered **v4 files remain valid and are kept as a robustness sample**, but they are
**noncanonical for training**. A headline table must not mix v4-trained and v5-trained rows — that
confounds the branch comparison (`vol_surface ≫ spot_history > vix_history`) with a sample change.
All 12 core runs are trained on v5. No exceptions, no partial tables.

## Provenance is enforced, not asserted

Every one of the 12 scripts validates the input HDF5 **before any data loads** and aborts with a loud
`SystemExit` naming what was wrong. `config.json` records provenance **read from the file's own
attrs**, plus `h5_sha256` of the HDF5 itself and the sample keys `dataset_version`, `quote_filter`,
`quote_filter_tolerance`.

Gates common to all 12:

- `dataset_version == "v5"` — **the new headline gate.** `schema_version` alone is no longer
  sufficient: a v4 file also carries `schema_version == "v4"`, so handing a v4 file to a training
  script used to succeed silently. It now fails loudly.
- `quote_filter == "static_bounds_midpoint"` and `quote_filter_tolerance == 0.0`
- `schema_version == "v4"` (rejects v3 outright — no such attr there)
- `cp_flag` matches the script's `option_type` (the call and put files are structurally identical)
- `export_t_basis == "settlement"`, `export_min_maturity_days == 1.0`
- `split_id` and the script's own `branch_key` exist as datasets

Additionally, and only on the 4 **`vix_history`** scripts: `phase1_vix_source_series` must name
`CBOE VIX` and `phase1_vix_sha256` must equal the reviewed CSV's SHA-256
`e2f18c516c9d48fb1730c2c0203fce36900c70aa0635b99e64afdde157346301`. That gate exists specifically to
make the VVIX mislabel unrepeatable. The 8 **non-VIX** scripts deliberately do **not** carry it — a
`vol_surface` or `spot_history` model never reads `vix_history`, so it would be a false constraint.
They record the general dataset provenance (`export_t_basis`, `export_min_maturity_days`,
`trunk_y_columns`, `am_settlement_coding`, all `phase1_*` attrs) under `config.json → data_provenance`.

Regression tests: `tests/test_v5_gate_all_scripts.py` (all 12, incl. "a v4 file is refused") and
`tests/test_vix_provenance_gate.py` (the VIX-only gate).

All defaults now point at `deeponet_tensors_{call,put}_v5.h5` and fresh `results_*_v5` dirs, so a
bare invocation can reach neither older data nor older weights. The `--h5-path` / `--results-dir`
flags below are redundant with those defaults but are kept explicit — a command should state which
data produced which result.

**Never point `--results-dir` at an existing v3 or v4 `results_*` dir.** Those weights back earlier
tables; overwriting them destroys the only reference.

## 1. Smoke — plumbing check (needs a v5 smoke HDF5)

All 12, 1 batch each, into throwaway `*_v5_smoke` dirs. Requires
`wrds_data_2020-2025/smoke_v5/deeponet_tensors_{call,put}_v5_smoke.h5` (a v5-attr smoke file; the
existing `smoke_v4/` files are v4 and will be **refused**, which is itself a useful gate check).

```bash
# --- vol_surface (headline) ---
conda run -n dl_new python train_model_v3/call/train_vol_surface.py      --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_call_v5_smoke.h5 --results-dir train_model_v3/call/results_vol_surface_v5_smoke      --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/call/train_vol_surface_don.py  --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_call_v5_smoke.h5 --results-dir train_model_v3/call/results_vol_surface_don_v5_smoke  --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/put/train_vol_surface.py       --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_put_v5_smoke.h5  --results-dir train_model_v3/put/results_vol_surface_v5_smoke       --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/put/train_vol_surface_don.py   --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_put_v5_smoke.h5  --results-dir train_model_v3/put/results_vol_surface_don_v5_smoke   --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0

# --- spot_history ---
conda run -n dl_new python train_model_v3/call/train_spot_history.py     --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_call_v5_smoke.h5 --results-dir train_model_v3/call/results_spot_history_v5_smoke     --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/call/train_spot_history_don.py --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_call_v5_smoke.h5 --results-dir train_model_v3/call/results_spot_history_don_v5_smoke --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/put/train_spot_history.py      --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_put_v5_smoke.h5  --results-dir train_model_v3/put/results_spot_history_v5_smoke      --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/put/train_spot_history_don.py  --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_put_v5_smoke.h5  --results-dir train_model_v3/put/results_spot_history_don_v5_smoke  --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0

# --- vix_history (corrected CBOE VIX) ---
conda run -n dl_new python train_model_v3/call/train_vix_history.py      --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_call_v5_smoke.h5 --results-dir train_model_v3/call/results_vix_history_v5_smoke      --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/call/train_vix_history_don.py  --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_call_v5_smoke.h5 --results-dir train_model_v3/call/results_vix_history_don_v5_smoke  --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/put/train_vix_history.py       --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_put_v5_smoke.h5  --results-dir train_model_v3/put/results_vix_history_v5_smoke       --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
conda run -n dl_new python train_model_v3/put/train_vix_history_don.py   --h5-path wrds_data_2020-2025/smoke_v5/deeponet_tensors_put_v5_smoke.h5  --results-dir train_model_v3/put/results_vix_history_don_v5_smoke   --max-train-batches 1 --epochs-adam 1 --epochs-finetune 0
```

Smoke results are **disposable**: 1-batch weights and meaningless metrics. Delete every
`*_v5_smoke` dir once the plumbing is confirmed, before anything runs `aggregate_results.py` —
they must never reach a manuscript table.

## 2. Full runs (need `deeponet_tensors_{call,put}_v5.h5`)

Headline `vol_surface` first — those four carry the main claim.

```bash
# --- vol_surface (headline, run these first) ---
conda run -n dl_new python train_model_v3/call/train_vol_surface.py      --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vol_surface_v5
conda run -n dl_new python train_model_v3/call/train_vol_surface_don.py  --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vol_surface_don_v5
conda run -n dl_new python train_model_v3/put/train_vol_surface.py       --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vol_surface_v5
conda run -n dl_new python train_model_v3/put/train_vol_surface_don.py   --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vol_surface_don_v5

# --- spot_history ---
conda run -n dl_new python train_model_v3/call/train_spot_history.py     --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_spot_history_v5
conda run -n dl_new python train_model_v3/call/train_spot_history_don.py --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_spot_history_don_v5
conda run -n dl_new python train_model_v3/put/train_spot_history.py      --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_spot_history_v5
conda run -n dl_new python train_model_v3/put/train_spot_history_don.py  --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_spot_history_don_v5

# --- vix_history (corrected CBOE VIX; replaces the quarantined VVIX runs) ---
conda run -n dl_new python train_model_v3/call/train_vix_history.py      --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vix_history_v5
conda run -n dl_new python train_model_v3/call/train_vix_history_don.py  --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vix_history_don_v5
conda run -n dl_new python train_model_v3/put/train_vix_history.py       --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vix_history_v5
conda run -n dl_new python train_model_v3/put/train_vix_history_don.py   --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vix_history_don_v5
```

## 3. Seed replication — the 4 headline `vol_surface` configs (seeds 43, 44)

Previously agreed queued work: the headline numbers should come with a spread, not a single seed.
Seed 42 is the default and is already covered by §2, so only 43 and 44 are extra. Distinct result
dirs per seed — never overwrite the seed-42 run.

```bash
# seed 43
conda run -n dl_new python train_model_v3/call/train_vol_surface.py     --seed 43 --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vol_surface_v5_seed43
conda run -n dl_new python train_model_v3/call/train_vol_surface_don.py --seed 43 --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vol_surface_don_v5_seed43
conda run -n dl_new python train_model_v3/put/train_vol_surface.py      --seed 43 --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vol_surface_v5_seed43
conda run -n dl_new python train_model_v3/put/train_vol_surface_don.py  --seed 43 --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vol_surface_don_v5_seed43

# seed 44
conda run -n dl_new python train_model_v3/call/train_vol_surface.py     --seed 44 --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vol_surface_v5_seed44
conda run -n dl_new python train_model_v3/call/train_vol_surface_don.py --seed 44 --h5-path wrds_data_2020-2025/deeponet_tensors_call_v5.h5 --results-dir train_model_v3/call/results_vol_surface_don_v5_seed44
conda run -n dl_new python train_model_v3/put/train_vol_surface.py      --seed 44 --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vol_surface_v5_seed44
conda run -n dl_new python train_model_v3/put/train_vol_surface_don.py  --seed 44 --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5  --results-dir train_model_v3/put/results_vol_surface_don_v5_seed44
```

Report the headline as mean ± spread over seeds {42, 43, 44}.

## 4. Watch item — `put/spot_history` DeepONet (B)

On v3 this run was an unstable outlier: **R²(price) 0.837 vs 0.963** for the matched MLP-head (A)
run. Its §2 command above is the rerun. When it lands, check whether v5 fixes it:

- If R²(price) comes back in line with A, the v3 result was a bad-seed/optimization artifact — say so
  and move on.
- If it is still far below A, do **not** quietly report the low number as a DeepONet property.
  Re-run it under seeds 43/44 (same pattern as §3, `results_spot_history_don_v5_seed43/44`) and
  report the spread, so instability is visible rather than mistaken for a systematic architecture gap.

```bash
conda run -n dl_new python train_model_v3/put/train_spot_history_don.py --seed 43 --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5 --results-dir train_model_v3/put/results_spot_history_don_v5_seed43
conda run -n dl_new python train_model_v3/put/train_spot_history_don.py --seed 44 --h5-path wrds_data_2020-2025/deeponet_tensors_put_v5.h5 --results-dir train_model_v3/put/results_spot_history_don_v5_seed44
```

## 5. Post-run checklist

1. **Provenance.** For every `results_*_v5*/config.json`: confirm `dataset_version` is `v5`,
   `quote_filter` is `static_bounds_midpoint`, `quote_filter_tolerance` is `0.0`, and that
   `h5_sha256` is present and identical across all runs that used the same HDF5. A missing or
   mismatched hash means the run is not attributable to a dataset — discard it.
2. **VIX sanity.** On the 4 VIX runs, `vix_provenance.source_series` must read
   `CBOE VIX (S&P 500 volatility index)`. Spot-check `vix_history` values against Cboe VIX closes:
   VIX ran ~9–83 over 2015–2025, VVIX ran ~60–200 — a max above ~90 means the wrong series is back.
3. **Delete every `*_v5_smoke` directory** before aggregation.
4. `conda run -n dl_new python analysis/eval_to_json.py --all --dataset-version v5`
5. `conda run -n dl_new python analysis/aggregate_results.py --dataset-version v5`
   The bare, no-flag forms of both commands now **hard-error by design** — the dataset version must
   be stated explicitly so a v4 robustness run can never be silently aggregated into the v5 table.
6. Update **`report.md` T1** from the aggregated output (not hand-copied from `loss_history.txt`),
   and update `PROGRESS_phase2_v3.md`: all 12 rows move to v5, the `vix_history` rows lose "pending
   retraining", and the branch-ranking claim is restated only if the v5 numbers still support it.
7. State plainly in `report.md` that **all 12 rows are v5 (quote-filtered)**, and that the v4
   unfiltered runs, if any are reported, appear only as a labelled robustness check.
