QUARANTINED — CBOE VVIX (not VIX). Do not feed this into any pipeline.
======================================================================

vvix_2015_2025.csv
  Index:   CBOE VVIX — the volatility OF VIX ("vol of vol")
  secid:   152892      ticker: VVIX
  Columns: secid,date,cusip,ticker,...,low,high,open,close,...
  SHA-256: B366B326814737B4A7C48657B9171CD3E953B30D7B1BE3C4FD5C6474D0514571

Why it is here
--------------
Until 2026-08-11 this file was named `vix_2015_2025.csv` and sat in the active
data directory, so every `vix_history` branch build and all four `vix_history`
trained runs consumed VVIX while being labeled VIX. It was moved here on
2026-08-12 so no preprocessing run can pick it up by filename or by globbing the
parent directory.

It is retained, not deleted, purely as an audit anchor: this checksum identifies
the exact bytes behind the quarantined weights in
`train_model_v3/_vix_is_vvix_LEGACY/`. See that directory's MIGRATION_NOTE.md.

The real VIX
------------
The corrected SPX VIX now lives in the parent directory as `vix_2015_2025.csv`
(wide-format Cboe: date,vixo,vixh,vixl,vix,vxno,...,vxd; SPX VIX is the
vixo/vixh/vixl/vix quartet), SHA-256
E2F18C516C9D48FB1730C2C0203FCE36900C70AA0635B99E64AFDDE157346301.

VVIX is a legitimate series, just not the one this project models. Nothing here
is evidence about VIX.
