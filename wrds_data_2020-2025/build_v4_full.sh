#!/usr/bin/env bash
# Full canonical v4 build. CPU only, strictly sequential (phase 2 peaks ~4 GB per
# part, so overlapping the stages would thrash). Phase 1 keeps ALL maturities
# (--min-maturity-days -1) for auditability; phase 2 applies the paper rule
# T > 1 day. Never touches any v3 artifact.
#
# Run detached; progress lands in build_v4_full.log.
set -euo pipefail
cd "$(dirname "$0")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== PHASE 1: parquet parts (all maturities retained) ==="
conda run --no-capture-output -n dl_new python pre_process_1_data_v4.py \
    --chunk-months 3 --min-maturity-days -1
log "phase 1 done"

for t in call put; do
    log "=== PHASE 2: $t HDF5 (paper rule T > 1 day) ==="
    conda run --no-capture-output -n dl_new python pre_process_2_hdf5_v4.py \
        --type "$t" --min-maturity-days 1.0
    log "phase 2 $t done"
done

log "=== BUILD COMPLETE ==="
ls -la deeponet_tensors_call_v4.h5 deeponet_tensors_put_v4.h5
