# Cubic-spline IV-surface baseline: grid-boundary error decomposition

Generated 2026-09-07 15:22:52 - CPU only, no checkpoint loaded.
Dataset v5, script `analysis/spline_boundary_decomposition.py`, git HEAD `6a28e8b4a057`.

Rows are classified by the delta of the **converged** sigma from the damped fixed point (the same array `baselines.py` decides `n_out_of_grid_delta` on), not by the |delta|=50 initial guess. Boundaries are exclusive: exactly 10 or 90 delta, and exactly 10 or 730 days, count as in-grid. The two delta masks are mutually exclusive but either may co-occur with `tenor_out`, so the three out-of-grid masks do not partition the split -- only `{in_grid, not_in_grid}` does, and only that pair's SSE shares sum to 100%. `delta_in_grid` is the looser reading of "in-grid" -- in-grid on the delta axis alone, tenor clamping ignored; it is listed because the two readings give different R2(log) and a citation must say which it means.

## call

819,342 test rows. Aggregate R2(price) = 0.999982, R2(log) = 0.764181 (reproduces `baselines_v5.json` to 1e-06).

| mask | n_rows | %_rows | %_log_SSE | %_price_SSE | R2(log) | R2(price) | n_sq_log>10 | n_sq_log>1 | n_rel_px_err>100% |
|---|---|---|---|---|---|---|---|---|---|
| `below_10_delta` | 185,863 | 22.68 | 99.40 | 0.35 | -1.0096 | 0.897119 | 25,324 | 49,652 | 9,082 |
| `above_90_delta` | 41,744 | 5.09 | 0.00 | 61.36 | 1.0000 | 0.999989 | 0 | 0 | 0 |
| `tenor_out` | 170,616 | 20.82 | 60.65 | 52.26 | 0.5740 | 0.999926 | 15,483 | 30,227 | 7,504 |
| `in_grid` | 500,722 | 61.11 | 0.12 | 13.62 | 0.9979 | 0.999321 | 0 | 0 | 1 |
| `not_in_grid` | 318,620 | 38.89 | 99.88 | 86.38 | 0.5960 | 0.999985 | 25,339 | 50,597 | 10,517 |
| `delta_in_grid` | 591,735 | 72.22 | 0.60 | 38.30 | 0.9920 | 0.998842 | 15 | 945 | 1,436 |
| `not_delta_in_grid` | 227,607 | 27.78 | 99.40 | 61.70 | 0.4673 | 0.999989 | 25,324 | 49,652 | 9,082 |

Overlaps: 60,499 rows are both below-10-delta and tenor-clamped, 19,104 both above-90-delta and tenor-clamped, 0 in both delta masks (0 by construction). 318,620 rows are clamped on at least one axis: 148,004 delta-only, 91,013 tenor-only, 79,603 both.

Baseline diagnostics recomputed here (must match `baselines_v5.csv`): n_out_of_grid_delta = 227607, n_out_of_grid_tenor = 170616.

## put

1,380,379 test rows. Aggregate R2(price) = 0.996594, R2(log) = -2.518559 (reproduces `baselines_v5.json` to 1e-06).

| mask | n_rows | %_rows | %_log_SSE | %_price_SSE | R2(log) | R2(price) | n_sq_log>10 | n_sq_log>1 | n_rel_px_err>100% |
|---|---|---|---|---|---|---|---|---|---|
| `below_10_delta` | 578,610 | 41.92 | 99.94 | 72.94 | -10.5081 | 0.718584 | 272,433 | 386,343 | 4,512 |
| `above_90_delta` | 24,114 | 1.75 | 0.00 | 0.30 | 0.9998 | 0.999957 | 0 | 0 | 0 |
| `tenor_out` | 274,497 | 19.89 | 26.29 | 17.17 | -2.0357 | 0.996465 | 84,613 | 112,851 | 8,639 |
| `in_grid` | 671,230 | 48.63 | 0.02 | 17.88 | 0.9939 | 0.998186 | 0 | 0 | 11 |
| `not_in_grid` | 709,149 | 51.37 | 99.98 | 82.12 | -6.1571 | 0.995106 | 272,433 | 388,039 | 8,646 |
| `delta_in_grid` | 777,655 | 56.34 | 0.06 | 26.76 | 0.9843 | 0.997782 | 0 | 1,696 | 4,145 |
| `not_delta_in_grid` | 602,724 | 43.66 | 99.94 | 73.24 | -7.8499 | 0.994982 | 272,433 | 386,343 | 4,512 |

Overlaps: 155,139 rows are both below-10-delta and tenor-clamped, 12,933 both above-90-delta and tenor-clamped, 0 in both delta masks (0 by construction). 709,149 rows are clamped on at least one axis: 434,652 delta-only, 106,425 tenor-only, 168,072 both.

Baseline diagnostics recomputed here (must match `baselines_v5.csv`): n_out_of_grid_delta = 602724, n_out_of_grid_tenor = 274497.

## Mechanism

Mechanism. The OptionMetrics surface is standardized on |delta| in [10, 90] and 10 to 730 calendar days, and baselines.py clamps any query outside that box to the nearest boundary rather than extrapolating. For puts, 41.92% of the test rows converge to |delta| below 10 and are therefore all priced off the single 10-delta column, an implied volatility that is far too low for the wing they actually sit in. Those rows carry 99.94% of the total log-space SSE and 272,433/272,433 of the squared-log-error observations above 10, which is what drives the aggregate R2(log) to -2.5186; on the rows clamped on neither axis the same spline reaches R2(log) = 0.9939 (0.9843 if only the delta clamp is excluded and tenor-clamped rows are kept). The error there is multiplicative, and the price scale it acts on is small: the mean true normalized price on those rows is 0.00203 against 0.0202 on the in-grid rows and 0.0886 on the deep-ITM rows, so a mispricing worth a squared log error above 10 -- a factor of 24 or more on the price -- still only costs 0.00172 in RMSE(V/K). Those rows do carry 72.94% of the price-space SSE, a large share and not a negligible one, but R2(price) is scored against the total variance of V/K, which the in-the-money rows set; that denominator is large enough that 72.94% of the residual sum still leaves R2(price) = 0.996594. The two metrics are not in conflict -- they weight the same errors on different scales -- and a single clamped boundary column is what separates them.
