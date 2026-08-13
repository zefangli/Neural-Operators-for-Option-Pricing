# v4 preprocessing — smoke-test report (rebuilt 2026-08-11, corrected VIX)

**Status: smoke build valid.** Both `--self-test` passes and the end-to-end
`test_vix_loader.py` tensor-match check pass on the artifacts below.

**What changed vs the previous (invalid) smoke build.** The earlier smoke
artifacts in this directory were built from a VIX CSV that actually carried
**VVIX** (the vol-of-VIX index; `vix_level` ranged 86.87–125.73). The loader
was rewritten for the corrected **Cboe wide-format SPX VIX file**
(`vix_2015_2025.csv`, columns `date,vixo,vixh,vixl,vix,vxno,...,vxdo,...`),
reads only the VIX columns, filters the 3 no-print holiday rows, validates the
schema fail-fast, and records full provenance (see §5). The rebuilt smoke
`vix_level` ranges **12.10–40.11** (Dec 2019–Feb 2020 SPX VIX, including the
2020-02-28 pandemic spike close of 40.11) — the old 125.73 max is gone.

## 1. Commands

```
conda run -n dl_new python pre_process_1_data_v4.py --self-test
conda run -n dl_new python pre_process_2_hdf5_v4.py --self-test
conda run -n dl_new python pre_process_1_data_v4.py --date-start 2019-11-01 \
    --date-end 2020-02-29 --parts-dir smoke_v4/parts_diag \
    --min-maturity-days -1 --chunk-months 3 --force
conda run -n dl_new python pre_process_2_hdf5_v4.py --type call --input smoke_v4/parts_diag \
    --output smoke_v4/deeponet_tensors_call_v4_smoke.h5
conda run -n dl_new python pre_process_2_hdf5_v4.py --type put  --input smoke_v4/parts_diag \
    --output smoke_v4/deeponet_tensors_put_v4_smoke.h5
conda run -n dl_new python test_vix_loader.py
```

(Phase 1 run with `--min-maturity-days -1` keeps `T<=0` rows in the parquet;
phase 2 applies the paper maturity cut, `min_maturity_days=1.0` default.)

## 2. Row counts at each pipeline stage

| stage | 2019-11-01..2020-01-31 | 2020-02-01..2020-02-29 |
|---|---|---|
| after joins (options × spot × vix_level × div × rate × vol surface) | 249,985 | 101,160 |
| after vol-surface validation (len 187) | 249,985 | 101,160 |
| after min-maturity (disabled, `-1`; `T<=0` counted) | 249,985 (`T<=0`: 4,473) | 101,160 (`T<=0`: 1,866) |
| after lookback warmup + history attach | 176,456 | 101,160 |
| **parquet part written** | **176,456** | **101,160** |

Total phase-1 output: **277,616 rows** in
`smoke_v4/parts_diag/deeponet_training_data_{2019-11-01,2020-02-01}.parquet`.

Phase-2 export (cp-flag + null filter + `T*365 > 1.0` cut):

| HDF5 | rows | train | val | test |
|---|---|---|---|---|
| `smoke_v4/deeponet_tensors_call_v4_smoke.h5` | **98,672** | 71,753 | 9,699 | 17,220 |
| `smoke_v4/deeponet_tensors_put_v4_smoke.h5` | **169,465** | 124,469 | 16,920 | 28,076 |

## 3. Date coverage

Phase-1 window `2019-11-01 .. 2020-02-29`; the 21-trading-day lookback warmup
consumes Nov, so the option sample is **2019-12-02 .. 2020-02-28 = 61 unique
trade dates** (identical in both HDF5 files), covering the Dec/Jan/Feb monthly
SPX expiries plus SPXW weeklies. Split boundaries (by unique date, 80/10/10):
last train date 2020-02-10, last val date 2020-02-19.

## 4. VIX-level range (genuine SPX VIX)

| dataset | min | median | max |
|---|---|---|---|
| call HDF5 `vix_level` | 12.10 | 14.52 | 40.11 |
| put HDF5 `vix_level` | 12.10 | 14.56 | 40.11 |
| full source file (2015-01-02 .. 2025-08-29, 2,706 rows) | 9.14 | 16.52 | 82.69 |

Reference check: 2015-01-02 VIX OHLC = [17.76, 20.14, 17.05, 17.79] — asserted
exactly in `pre_process_1_data_v4.py --self-test`.

## 5. VIX provenance (recorded in `pipeline_config_v4.json`, mirrored into the
HDF5 attrs as `phase1_vix_*`)

* **source file:** `vix_2015_2025.csv` (Cboe wide format)
* **SHA-256:** `e2f18c516c9d48fb1730c2c0203fce36900c70aa0635b99e64afdde157346301`
* **source series:** CBOE VIX (S&P 500 volatility index)
* **source columns:** `date,vixo,vixh,vixl,vix`; mapping
  `vixo->open, vixh->high, vixl->low, vix->close`
* **row counts:** 2,709 raw; **2,706** after dropping the 3 no-print holiday
  rows (2021-04-02, 2021-07-25, 2021-12-24; empty VIX OHLC); VXN/VXD columns
  are never selected
* **date range:** 2015-01-02 .. 2025-08-29
* **legacy guard:** the loader refuses any other layout (e.g. the old
  VVIX/WRDS `secid,date,open,high,low,close,ticker` format) with a clear error
  — verified in `--self-test`.

## 6. Validation results

* `pre_process_1_data_v4.py --self-test`: OK (CBOE VIX loader: unique sorted
  dates, finite positive OHLC, `low<=min(open,close)`/`high>=max(open,close)`,
  2015-01-02 reference row, VXN/VXD excluded by construction, 21-day window vs
  hand-computed, legacy-VVIX rejection; plus settlement-maturity and
  duplicate-key checks).
* `pre_process_2_hdf5_v4.py --self-test`: OK.
* `test_vix_loader.py`: OK — HDF5 attrs identify CBOE VIX with the exact
  SHA-256; every `vix_history` row for the first 3 distinct dates matches a
  directly-calculated raw 21-day normalized window (call: 4,366 rows, put:
  8,118 rows, float32 tolerance); every `vix_level` matches the raw VIX close
  for that date; `vix_level` max < 90 confirms no VVIX contamination.

The v3 artifacts (`deeponet_tensors_{call,put}.h5`,
`deeponet_training_data_parts/`) are untouched; every v4 output path contains
`v4` (or `smoke_v4`).
