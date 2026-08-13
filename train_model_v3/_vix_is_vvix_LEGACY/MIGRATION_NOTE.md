# Migration note — `vix_history` branch trained on VVIX (not VIX)

**Date of quarantine:** 2026-08-11

## What happened

The `vix_history` branch input was built from a **mislabeled CSV**: the file named for VIX was in
fact the CBOE **VVIX** index — the *volatility of VIX* (secid `152892`, ticker `VVIX`), not VIX
itself. All four trained runs that consumed it are therefore **VVIX artifacts**, not VIX results.
They are quarantined here, unchanged and undeleted, so they remain available for audit but are
removed from active discovery (`analysis/aggregate_results.py` globbing, doc tables, etc.).

**The results in these directories are not evidence about VIX.** Do not relabel them as VIX.

## Legacy source file

- File: `vvix_2015_2025.csv` (the CSV previously routed into `vix_history`)
- Index: CBOE VVIX — "volatility of volatility"
- secid: `152892`, ticker: `VVIX`
- Columns: `open`, `high`, `low`, `close`
- **SHA-256:** `B366B326814737B4A7C48657B9171CD3E953B30D7B1BE3C4FD5C6474D0514571`
- Quarantined location: `wrds_data_2020-2025/_legacy_vvix/vvix_2015_2025.csv`
  (moved out of the active data directory 2026-08-12 so no preprocessing run can
  pick it up; kept, not deleted, so the four runs above remain reproducible)

This checksum is the audit anchor: it identifies exactly which bytes produced the
quarantined weights. The corrected CBOE VIX file is a different series entirely —
see its own SHA-256 below.

## Affected directories (original paths → current location)

| Original path | Quarantined at |
|---|---|
| `train_model_v3/call/results_vix_history/` | `train_model_v3/_vix_is_vvix_LEGACY/call_results_vix_history/` |
| `train_model_v3/call/results_vix_history_don/` | `train_model_v3/_vix_is_vvix_LEGACY/call_results_vix_history_don/` |
| `train_model_v3/put/results_vix_history/` | `train_model_v3/_vix_is_vvix_LEGACY/put_results_vix_history/` |
| `train_model_v3/put/results_vix_history_don/` | `train_model_v3/_vix_is_vvix_LEGACY/put_results_vix_history_don/` |

Each directory contains `config.json`, `best_model.pth`, `best_model_phase1.pth`,
`final_model.pth`, `loss_history.txt`, `loss_plot.png`. Nothing was deleted or modified.

## Corrected VIX source (replaces VVIX)

- File: wide-format Cboe SPX VIX (columns `vixo`, `vixh`, `vixl`, `vix` — i.e. open/high/low/close)
- SHA-256: `E2F18C516C9D48FB1730C2C0203FCE36900C70AA0635B99E64AFDDE157346301`
- Preprocessing of this file into the v4 HDF5 (`deeponet_tensors_{call,put}_v4.h5`) is owned by the
  preprocessing agent (`wrds_data_2020-2025/`), not this directory.

## Status

Corrected VIX models are **pending GPU retraining**. Deferred, ready-to-run commands live in
`DEFERRED_GPU_COMMANDS.md` in this directory. Until those runs complete, `vix_history` rows in
report tables are marked "pending retraining" and the branch-ranking claims citing VIX-history
numbers are suspended.
