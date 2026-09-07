# Duplicate-input rates over all splits

Generated 2026-09-07 18:47:47. CPU only, read-only, no model. Dataset v5, script `analysis/duplicate_inputs_all_splits.py`, git HEAD `394bec038d66`.

The published duplicate-input check (`validate_v4.py` section 8, `analysis/data_quality.py`) is computed on the **test split only**. This table extends it to train, validation and the full file, and adds cross-split collisions.

## Keys

| key | columns | meaning |
|---|---|---|
| `input_key` | `date`, `log_moneyness`, `T_years`, `r`, `q` | the full model input for a row. Two rows sharing it are the same question asked twice; a target disagreement inside such a group is unfittable by construction. This is the published statistic's key. |
| `query_key` | `log_moneyness`, `T_years`, `r`, `q` | `input_key` with the date dropped. **Diagnostic only.** The model is conditioned on the market state, which is a function of the date, so the same query on two dates is not a duplicate input -- the branch tensor differs and different prices are correct. A high collision rate here reflects the listed strike/maturity schedule repeating, not leakage. |

Grouping is on the **exact float32 bits** stored in the HDF5 -- no rounding and no tolerance. `validate_v4.py` upcasts to float64 first, which is exact for float32 input, so the groups are identical.

## `input_key`

| option | split | rows | groups | dup groups | rows in dup groups | % rows in dup groups | max group |
|---|---|---|---|---|---|---|---|
| call | train | 3,297,722 | 3,297,722 | 0 | 0 | 0.000000 | 1 |
| call | val | 729,204 | 729,204 | 0 | 0 | 0.000000 | 1 |
| call | test | 819,342 | 819,342 | 0 | 0 | 0.000000 | 1 |
| call | FULL | 4,846,268 | 4,846,268 | 0 | 0 | 0.000000 | 1 |
| put | train | 5,323,327 | 5,323,327 | 0 | 0 | 0.000000 | 1 |
| put | val | 1,221,643 | 1,221,643 | 0 | 0 | 0.000000 | 1 |
| put | test | 1,380,379 | 1,380,379 | 0 | 0 | 0.000000 | 1 |
| put | FULL | 7,925,349 | 7,925,349 | 0 | 0 | 0.000000 | 1 |

Cross-split (groups formed on the full file whose rows do not all lie in one split):

| option | groups spanning >1 split | rows in them | % of rows | train&val | train&test | val&test |
|---|---|---|---|---|---|---|
| call | 0 | 0 | 0.000000 | 0 | 0 | 0 |
| put | 0 | 0 | 0.000000 | 0 | 0 | 0 |

`input_key` cross-split collisions are zero **by construction**: the splits are cut by date and the date is part of the key. The row is reported so the construction is checked rather than assumed.

## `query_key`

| option | split | rows | groups | dup groups | rows in dup groups | % rows in dup groups | max group |
|---|---|---|---|---|---|---|---|
| call | train | 3,297,722 | 3,297,722 | 0 | 0 | 0.000000 | 1 |
| call | val | 729,204 | 729,204 | 0 | 0 | 0.000000 | 1 |
| call | test | 819,342 | 819,342 | 0 | 0 | 0.000000 | 1 |
| call | FULL | 4,846,268 | 4,846,268 | 0 | 0 | 0.000000 | 1 |
| put | train | 5,323,327 | 5,323,327 | 0 | 0 | 0.000000 | 1 |
| put | val | 1,221,643 | 1,221,643 | 0 | 0 | 0.000000 | 1 |
| put | test | 1,380,379 | 1,380,379 | 0 | 0 | 0.000000 | 1 |
| put | FULL | 7,925,349 | 7,925,349 | 0 | 0 | 0.000000 | 1 |

Cross-split (groups formed on the full file whose rows do not all lie in one split):

| option | groups spanning >1 split | rows in them | % of rows | train&val | train&test | val&test |
|---|---|---|---|---|---|---|
| call | 0 | 0 | 0.000000 | 0 | 0 | 0 |
| put | 0 | 0 | 0.000000 | 0 | 0 | 0 |

A `query_key` collision carries no information about leakage even when it happens: the branch tensor differs, so the model sees different inputs and different prices are correct. In this sample there are none at all -- 0 groups span more than one split. A listed strike does recur day after day, but the query does not: log-moneyness is log(S/K) and so carries the day's spot level, and the settlement-aware maturity is continuous in years, so the same contract on two dates lands on a different `(log_m, T)`.

## Reproduction check

- **call**: test-split `input_key` rate 0.00000000% vs wrds_data_2020-2025/VALIDATION_v5.json duplicate_input_pct_test = 0.00000000% (tolerance 1e-09).
- **put**: test-split `input_key` rate 0.00000000% vs wrds_data_2020-2025/VALIDATION_v5.json duplicate_input_pct_test = 0.00000000% (tolerance 1e-09).
