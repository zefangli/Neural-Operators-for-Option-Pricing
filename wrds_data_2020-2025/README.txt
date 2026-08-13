WRDS OptionMetrics IvyDB US — SPX 2015-2025 preprocessing
=========================================================

Security: secid 108105 (SPX)
Date range: 2015-01-02 to 2025-08-29 (see *_2015_2025.csv files)

Use the v4 pipeline. The v3 scripts are LEGACY and no longer run.
--------------------------------------------------------------------
`pre_process_1_data.py` / `pre_process_2_hdf5.py` (v3) are kept for provenance
only. v3 phase 1 CANNOT read the current `vix_2015_2025.csv`: it expects
`open/high/low/close`, and the corrected Cboe file has `vixo/vixh/vixl/vix`, so
it dies with `ColumnNotFoundError: unable to find column "open"`. Do not "fix"
that by renaming columns — v3 also lacks settlement-aware maturity and drops the
contract identifiers the paper needs. Use the _v4 scripts below.

Two corrections are baked into v4 (2026-08-11/12):

1. VIX, not VVIX. Until 2026-08-11 the file named `vix_2015_2025.csv` actually
   held CBOE **VVIX** (vol-of-VIX, secid 152892). Every v3 `vix_history` tensor
   and all four trained `vix_history` runs are therefore VVIX artifacts; they are
   quarantined under `train_model_v3/_vix_is_vvix_LEGACY/` and the VVIX CSV now
   sits in `_legacy_vvix/`. v4 reads genuine Cboe SPX VIX.
2. Settlement-aware maturity. v3 used calendar days/365, which maps every
   expiry-day contract to exactly T=0 (where Black-Scholes returns intrinsic for
   any sigma, making those rows unfittable) and collapses AM-settled SPX and
   PM-settled SPXW into identical model inputs. v4 computes T from documented
   quote/settlement times, which drops the duplicate-input rate to 0.00%.

Environment
-----------
    conda activate dl_new
    cd wrds_data_2020-2025

Dependencies: polars, numpy, h5py

Input CSVs (active)
-------------------
- optionPrice_2015_2025.csv        SPX option quotes
- volatilitySurface_2015_2025.csv  OptionMetrics IV surface (11 tenors x 17 deltas)
- interestRate_2015_2025.csv       zero curve
- spx_dividendYield_2015_2025.csv  dividend yield
- spx_price_2015_2025.csv          SPX OHLCV
- vix_2015_2025.csv                Cboe wide format. SHA-256 (full, not
                                   abbreviated -- the training scripts compare
                                   against this exact value):
    E2F18C516C9D48FB1730C2C0203FCE36900C70AA0635B99E64AFDDE157346301
                                   2,709 raw rows / 2,706 usable (3 holidays
                                   carry an empty VIX print and are dropped).
                                   date,vixo,vixh,vixl,vix,vxno..vxn,vxdo..vxd
                                   ONLY vixo/vixh/vixl/vix (SPX VIX) is used.
                                   VXN (Nasdaq-100) and VXD (DJIA) are present in
                                   the file and deliberately ignored.
- _legacy_vvix/                    quarantined VVIX file — never an input.

Filters and features
--------------------
- Options: calls and puts (cp_flag C/P), volume > 0
- Targets: log_moneyness = log(S/K), log_normalized_price = log(mid/K)
- branch_u:      IV surface (187 = 11 tenors x 17 deltas) per (date, cp_flag)
- spot_history:  21 trading days of SPX OHLCV, normalized by the last close
- vix_history:   21 trading days of CBOE SPX VIX OHLC, normalized by the last
                 VIX close (21 x 4 = 84)
- vix_level:     contemporaneous UNNORMALIZED CBOE VIX close (the level that
                 vix_history's normalization destroys)
- v4 also retains: symbol, optionid, root, am_settlement, exercise_style,
  impl_volatility (OptionMetrics observed IV), market delta, best_bid,
  best_offer, spread_norm, half_spread_norm, strike, T_calendar, T_settlement
- Rows before the full lookback window are dropped (no left-padding)
- Default lookback = 21 trading days (--lookback-days)

Run order (v4)
--------------
1. Self-tests first (touch no data, seconds):

       python pre_process_1_data_v4.py --self-test
       python pre_process_2_hdf5_v4.py --self-test
       python test_vix_loader.py

2. Phase 1 — parquet parts:

       python pre_process_1_data_v4.py --chunk-months 3

   Use --chunk-months 3 for the full rebuild: phase 2 loads one part at a time,
   and a 12-month part (~1.2M rows) needs ~20 GB and will thrash.
   Writes to deeponet_training_data_parts_v4/ (--parts-dir to override).
   Re-running skips existing parts; --force rebuilds them.

   Optional smoke slice:
       python pre_process_1_data_v4.py --date-start 2019-11-01 --date-end 2020-02-29 \
           --parts-dir smoke_v4/parts_diag --min-maturity-days -1 --chunk-months 3

3. Phase 2 — HDF5 (separate call / put files):

       python pre_process_2_hdf5_v4.py --type call
       python pre_process_2_hdf5_v4.py --type put

   Output paths must contain 'v4' (guard: the v3 .h5 files are irreplaceable).
   Default output: deeponet_tensors_{call,put}_v4.h5

Maturity rule
-------------
--min-maturity-days defaults to 1.0, the paper sample rule (keep T > 1 day).
  0.0  keeps everything with strictly positive maturity
  -1   DIAGNOSTIC ONLY — keeps T <= 0 rows, which are not priceable
--t-basis {settlement,calendar}; 'settlement' is the v4 default, 'calendar'
reproduces the v3 behaviour for comparison.

T=0 rows are excluded, never epsilon-patched: adding a small positive T would
manufacture optionality the contract does not have.

Outputs
-------
- deeponet_training_data_parts_v4/   parquet parts
- deeponet_tensors_call_v4.h5
- deeponet_tensors_put_v4.h5
- pipeline_config_v4.json            provenance (source files, SHA-256s, the
                                     settlement convention, row counts)

Canonical build (completed 2026-08-12 09:14)
---------------------------------------------
                        Call            Put
Total rows              4,955,592       7,991,054
Train                   3,360,823       5,370,629
Validation                754,276       1,229,955
Test                       840,493      1,390,470
Unique dates                2,661          2,661

Coverage 2015-02-02 .. 2025-08-29; last train date 2023-07-17; last validation
date 2024-08-06. Built from 43 quarterly parquet parts (8.06 GB call /
13.00 GB put). Observed minimum maturity: 1.729 days (AM-settlement 6.5h
offset from an end-of-day quote to the 09:30 ET opening print -- expected, not
an anomaly; PM-settled SPXW bottoms out at exactly 2.000 days).

HDF5 SHA-256 (abbreviated): call 5f1c42aa...10546, put 2ba5b13b...bd29 -- full
hashes are in VALIDATION_v4.json, do not invent or re-quote them from memory.

Validation: all 31 checks passed (validate_v4.py -> VALIDATION_v4.json),
covering finite values, time-ordered non-overlapping splits by unique date,
zero duplicate model inputs (down from v3's 17.8%/19.8%), genuine VIX levels
(9.14-82.69, not VVIX's 60-200 range), and direct raw-VIX-to-HDF5 matches.

v3-vs-v4 comparability: v4 changes three things at once relative to v3 --
maturity basis, filtering, and contract identity (see "Two corrections" above
plus the settlement-aware maturity rule). All existing v3-trained model
metrics are therefore historical and not comparable to v4 results; all 12
core training runs are being retrained on v4, not only the four vix_history
ones (train_model_v3/_vix_is_vvix_LEGACY/DEFERRED_GPU_COMMANDS.md). No v4
model has been trained yet.

HDF5 split_id: 0=train, 1=val, 2=test (80/10/10 by unique trade date, so no
same-day leakage across splits).

Backward compatibility: v4 keeps every v3 dataset name, shape, dtype and row
alignment, so existing v3 consumers load a v4 file unchanged. New fields are
added alongside; trunk_y stays 4 columns.

Training
--------
The four vix_history training scripts validate provenance before training and
REFUSE any HDF5 that is not schema_version v4 with a CBOE VIX source hash — a v3
file cannot be silently retrained and labeled VIX. See
train_model_v3/_vix_is_vvix_LEGACY/DEFERRED_GPU_COMMANDS.md.

Citation
--------
Wharton Research Data Services. "WRDS" wrds.wharton.upenn.edu, accessed 2026-02-15.
Cboe Global Markets, VIX historical data (vix_2015_2025.csv).
