#!/usr/bin/env bash
# Full CPU analyses on one dataset version. Strictly sequential -- each pass
# streams multi-GB HDF5s, so overlapping them would thrash. Outputs are tagged
# with the version, so earlier versions' results are never overwritten.
#
#   ./run_v4_analyses.sh v5     # canonical (quote-filtered) sample
#   ./run_v4_analyses.sh v4     # unfiltered robustness sample
set -uo pipefail
cd "$(dirname "$0")/.."
ver="${1:-v5}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
rc=0

for step in data_quality diagnostics baselines; do
    extra=""
    # skip the 8.6 GB raw-CSV rescan; the deepest bound violations are already
    # traced in results/QUOTE_QUALITY_REPORT.md
    [ "$step" = "data_quality" ] && extra="--no-trace-raw"
    log "=== $step ($ver) ==="
    conda run --no-capture-output -n dl_new python "analysis/${step}.py" \
        --dataset-version "$ver" $extra
    s=$?
    log "$step exit=$s"
    [ $s -ne 0 ] && rc=$s
done

log "=== ALL $ver ANALYSES DONE (rc=$rc) ==="
ls -la results/*_${ver}*.json results/*_${ver}*.csv 2>/dev/null
exit $rc
