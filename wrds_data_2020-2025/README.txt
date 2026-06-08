WRDS OptionMetrics IvyDB US — SPX 2015-2025 preprocessing
=========================================================

Security: secid 108105 (SPX)
Date range: 2015-01-02 to 2025-08-29 (see *_2015_2025.csv files)

Environment
-----------
Use the conda environment `dl_new`:

    conda activate dl_new
    cd wrds_data_2020-2025

Dependencies: polars, numpy, h5py

Filters and features
--------------------
- Options: calls and puts (cp_flag C/P), volume > 0
- Targets: log_moneyness = log(S/K), log_normalized_price = log(mid/K)
- branch_u: implied vol surface (187 = 11 tenors × 17 deltas) per (date, cp_flag)
- spot_history: last SPOT_LOOKBACK_DAYS trading days of normalized SPX OHLCV
- vix_history: last SPOT_LOOKBACK_DAYS trading days of normalized VIX OHLC
- Rows before full lookback window are dropped (no left-padding)
- Default SPOT_LOOKBACK_DAYS = 21 (set in pre_process_1_data.py or --lookback-days)

Run order
---------
1. Phase 1 — parquet:
       python pre_process_1_data.py

   Smoke test (optional date slice):
       python pre_process_1_data.py --date-start 2020-01-01 --date-end 2021-12-31

   The full run writes yearly files under deeponet_training_data_parts/. This
   directory is the parquet dataset used by Phase 2 and inspection; keeping it
   partitioned avoids building one large in-memory table.
   Re-running Phase 1 skips existing yearly part files by default. Use --force
   only when you intentionally want to rebuild all parts.

2. Phase 2 — HDF5 (separate call / put files):
       python pre_process_2_hdf5.py --type call
       python pre_process_2_hdf5.py --type put

   If deeponet_training_data.parquet is not present, Phase 2 falls back to
   deeponet_training_data_parts/. It appends each parquet part into HDF5 instead
   of loading the full data into RAM.

3. Phase 3 — inspect:
       python pre_process_3_print_dataset.py

Outputs
-------
- deeponet_training_data.parquet
- deeponet_tensors_call.h5
- deeponet_tensors_put.h5

HDF5 split_id: 0=train, 1=val, 2=test (80/10/10 by unique trade dates)

Training (future — train_code/)
-------------------------------
Load one branch input: branch_u, spot_history, or vix_history.
See ../plan.md for training scope.

Citation
--------
Wharton Research Data Services. "WRDS" wrds.wharton.upenn.edu, accessed 2026-02-15.
