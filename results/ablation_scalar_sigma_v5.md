**Headline: the per-query architecture decisively outperforms the scalar-sigma ablation, and the gap is catastrophic on puts.** Removing the per-contract conditioning (sigma_hat from the market-state latent alone, i.e. flat-vol Black-Scholes with no smile/skew) drops call R2(log,T>1d) from 0.991044 to 0.800103 and call R2(price) from 0.999977 to 0.999825; on puts it drops R2(log,T>1d) from 0.977899 to -0.316871 and R2(price) from 0.992553 to -3.090190 -- i.e. **negative**, worse than predicting the mean price. Seeds: 42 only; the gap needs no replication to be decisive.

This table is NOT part of T1 (results/table_T1_v5.md) and must never be merged into it: it reports a deliberately crippled architecture.

| option | variant | sigma conditioning | R2(price) | R2(log,T>1d) | RMSE(log) | RMSE(price) |
|---|---|---|---|---|---|---|
| call | per_query | latent_plus_query | 0.999977 | 0.991044 | 0.221896 | 0.002168 |
| call | scalar_sigma | market_state_only | 0.999825 | 0.800103 | 1.048319 | 0.005962 |
| put | per_query | latent_plus_query | 0.992553 | 0.977899 | 0.294127 | 0.001924 |
| put | scalar_sigma | market_state_only | -3.090190 | -0.316871 | 2.270415 | 0.045080 |
