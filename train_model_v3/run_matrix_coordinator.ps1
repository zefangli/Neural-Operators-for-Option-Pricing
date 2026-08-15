# Sequential coordinator for the remaining 11 seed-42 v5 canonical runs.
# Must itself run as a Windows-owned detached process (launched via
# Start-Process -WindowStyle Hidden), matching run_detached.ps1's own pattern --
# a 20+ hour job cannot be owned by any single agent tool call.
#
# One job at a time, in order. Before each: refuse if that job's results-dir,
# log, sentinel, or tmp-sentinel already exists (halt for review, never
# delete/resume). Check RAM/pagefile headroom, retrying with a wait rather than
# failing outright, since headroom fluctuates with the user's own apps. After
# each job: exit code 0, checkpoints, config provenance, finite loss history --
# stop the whole sequence immediately on any failure.

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$CALL_H5   = "..\wrds_data_2020-2025\deeponet_tensors_call_v5.h5"
$PUT_H5    = "..\wrds_data_2020-2025\deeponet_tensors_put_v5.h5"
$CALL_SHA  = "ecf94de2996e3212ac791f944f847142e7a809c640b24cbd5989e8bdba801153"
$PUT_SHA   = "46f3ee9d03d3423c6330c2b35a4d09893b0f538bdc046bc1018ebab79861c756"
$VIX_SHA   = "e2f18c516c9d48fb1730c2c0203fce36900c70aa0635b99e64afdde157346301"
$VIX_SERIES = "CBOE VIX (S&P 500 volatility index)"

$MIN_RAM_GB      = 2.0
$MIN_PAGEFILE_GB = 8.0
$RESOURCE_RETRY_DELAY_SEC = 300
$RESOURCE_RETRY_MAX       = 24   # 24 * 5min = 2h before giving up

# side, script, results-dir, is_vix
$JOBS = @(
    @{side='call'; script='train_vol_surface_don.py';  rdir='results_vol_surface_don_v5';  vix=$false},
    @{side='put';  script='train_vol_surface.py';       rdir='results_vol_surface_v5';      vix=$false},
    @{side='put';  script='train_vol_surface_don.py';   rdir='results_vol_surface_don_v5';  vix=$false},
    @{side='call'; script='train_spot_history.py';      rdir='results_spot_history_v5';     vix=$false},
    @{side='call'; script='train_spot_history_don.py';  rdir='results_spot_history_don_v5'; vix=$false},
    @{side='put';  script='train_spot_history.py';      rdir='results_spot_history_v5';     vix=$false},
    @{side='put';  script='train_spot_history_don.py';  rdir='results_spot_history_don_v5'; vix=$false},
    @{side='call'; script='train_vix_history.py';       rdir='results_vix_history_v5';      vix=$true},
    @{side='call'; script='train_vix_history_don.py';   rdir='results_vix_history_don_v5';  vix=$true},
    @{side='put';  script='train_vix_history.py';       rdir='results_vix_history_v5';      vix=$true},
    @{side='put';  script='train_vix_history_don.py';   rdir='results_vix_history_don_v5';  vix=$true}
)

$CoordLog = "runs\matrix_coordinator.log"
$CoordStatus = "runs\matrix_coordinator.status.json"
New-Item -ItemType Directory -Force -Path "runs","jobs" | Out-Null

function Log($msg) {
    $line = "[$((Get-Date).ToString('o'))] $msg"
    $line | Out-File -FilePath $CoordLog -Append -Encoding Unicode
}

function Write-Status($obj) {
    $tmp = "$CoordStatus.tmp"
    ($obj | ConvertTo-Json -Depth 5) | Set-Content -Path $tmp -Encoding Unicode
    Move-Item -Path $tmp -Destination $CoordStatus -Force
}

function Get-ResourceHeadroom {
    $os = Get-CimInstance Win32_OperatingSystem
    $ramGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)   # FreePhysicalMemory is in KB
    $pfTotal = 0.0; $pfUsed = 0.0
    foreach ($pf in (Get-CimInstance Win32_PageFileUsage)) {
        $pfTotal += $pf.AllocatedBaseSize
        $pfUsed  += $pf.CurrentUsage
    }
    $pageGB = [math]::Round(($pfTotal - $pfUsed) / 1024, 2)   # MB -> GB
    [PSCustomObject]@{ RamFreeGB = $ramGB; PageFreeGB = $pageGB }
}

function Wait-ForResources($jobName) {
    for ($i = 0; $i -lt $RESOURCE_RETRY_MAX; $i++) {
        $r = Get-ResourceHeadroom
        Log ("resource check before $jobName -- RAM free {0} GiB, pagefile free {1} GiB" -f $r.RamFreeGB, $r.PageFreeGB)
        if ($r.RamFreeGB -ge $MIN_RAM_GB -and $r.PageFreeGB -ge $MIN_PAGEFILE_GB) {
            return $r
        }
        Log "  below threshold (need >= ${MIN_RAM_GB} GiB RAM, >= ${MIN_PAGEFILE_GB} GiB pagefile) -- waiting ${RESOURCE_RETRY_DELAY_SEC}s"
        Start-Sleep -Seconds $RESOURCE_RETRY_DELAY_SEC
    }
    return $null
}

function Test-NoActiveCudaTraining {
    # Defensive check only -- the coordinator itself runs strictly sequentially and
    # waits for each sentinel, so this should never actually fire in normal operation.
    try {
        $procs = & nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>$null
        $training = $procs | Where-Object { $_ -match 'python' }
        return -not $training
    } catch {
        return $true   # nvidia-smi unavailable -- don't block on a check we can't run
    }
}

function Validate-Job($job, $sentinelPath, $rdirPath) {
    if (-not (Test-Path $sentinelPath)) { return "no sentinel at $sentinelPath" }
    $sentinel = Get-Content $sentinelPath -Raw | ConvertFrom-Json
    if ($sentinel.exit_code -ne 0) { return "exit_code=$($sentinel.exit_code)" }

    $cfgPath = Join-Path $rdirPath "config.json"
    if (-not (Test-Path $cfgPath)) { return "missing config.json" }
    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json

    if ($cfg.dataset_version -ne 'v5') { return "config dataset_version=$($cfg.dataset_version), expected v5" }
    if ($cfg.quote_filter -ne 'static_bounds_midpoint') { return "config quote_filter=$($cfg.quote_filter)" }
    $expectSha = if ($job.side -eq 'call') { $CALL_SHA } else { $PUT_SHA }
    if ($cfg.h5_sha256 -ne $expectSha) { return "h5_sha256 mismatch: got $($cfg.h5_sha256), expected $expectSha" }
    if ($cfg.seed -ne 42) { return "seed=$($cfg.seed), expected 42" }

    if ($job.vix) {
        $prov = $cfg.vix_provenance
        if (-not $prov) { return "missing vix_provenance block" }
        if ($prov.source_series -ne $VIX_SERIES) { return "vix source_series=$($prov.source_series)" }
        if ($prov.vix_csv_sha256 -ne $VIX_SHA) { return "vix_csv_sha256 mismatch" }
    }

    foreach ($ckpt in @('best_model_phase1.pth','best_model.pth','final_model.pth')) {
        if (-not (Test-Path (Join-Path $rdirPath $ckpt))) { return "missing checkpoint $ckpt" }
    }

    $lossPath = Join-Path $rdirPath "loss_history.txt"
    if (-not (Test-Path $lossPath)) { return "missing loss_history.txt" }
    $lossText = Get-Content $lossPath -Raw
    if ($lossText -match '(?i)\bnan\b|\binf\b') { return "non-finite value in loss_history.txt" }
    if ($lossText -notmatch 'Test R\^2') { return "loss_history.txt has no test metrics block" }

    return $null   # null = valid
}

Log "=== MATRIX COORDINATOR START: $($JOBS.Count) jobs ==="
Write-Status @{ status = 'running'; started = (Get-Date).ToString('o'); completed_jobs = @(); current_job = $null }

$completed = @()
foreach ($job in $JOBS) {
    $name = "$($job.side)_$([IO.Path]::GetFileNameWithoutExtension($job.script))_v5_seed42"
    $rdirPath = Join-Path $job.side $job.rdir
    $logPath = "runs\$name.log"
    $sentinelPath = "runs\$name.sentinel"
    $jobScriptPath = "jobs\$name.cmd"

    Log "--- [$($completed.Count + 1)/$($JOBS.Count)] $name ---"
    Write-Status @{ status = 'running'; started = (Get-Date).ToString('o'); completed_jobs = $completed; current_job = $name }

    # Refuse to touch a pre-existing, non-empty run without a valid completed
    # sentinel -- halt for review rather than delete or resume it.
    $preExisting = @()
    foreach ($p in @($rdirPath, $logPath, $sentinelPath, "$sentinelPath.tmp")) {
        if (Test-Path $p) { $preExisting += $p }
    }
    if ($preExisting.Count -gt 0) {
        $msg = "HALT: pre-existing artifact(s) for $name : $($preExisting -join ', ') -- refusing to overwrite or resume"
        Log $msg
        Write-Status @{ status = 'halted'; reason = $msg; completed_jobs = $completed; current_job = $name }
        exit 1
    }

    if (-not (Test-NoActiveCudaTraining)) {
        $msg = "HALT: another CUDA training process detected before $name"
        Log $msg
        Write-Status @{ status = 'halted'; reason = $msg; completed_jobs = $completed; current_job = $name }
        exit 1
    }

    $res = Wait-ForResources $name
    if ($null -eq $res) {
        $msg = "HALT: resource headroom never recovered above threshold before $name"
        Log $msg
        Write-Status @{ status = 'halted'; reason = $msg; completed_jobs = $completed; current_job = $name }
        exit 1
    }

    $h5 = if ($job.side -eq 'call') { $CALL_H5 } else { $PUT_H5 }
    @"
@echo off
cd /d "%~dp0\.."
conda run --no-capture-output -n dl_new python $($job.side)\$($job.script) --seed 42 --h5-path $h5 --results-dir $($job.side)\$($job.rdir)
exit /b %ERRORLEVEL%
"@ | Set-Content -Path $jobScriptPath -Encoding ASCII

    $launch = Start-Process -WindowStyle Hidden -PassThru -FilePath "powershell.exe" `
        -RedirectStandardOutput "runs\$name.launch.out" -RedirectStandardError "runs\$name.launch.err" `
        -ArgumentList @(
            "-NoProfile","-ExecutionPolicy","Bypass","-File","run_detached.ps1",
            "-SentinelPath",$sentinelPath, "-LogPath",$logPath, "-JobScript",$jobScriptPath
        )
    Log "launched pid=$($launch.Id)"

    while (-not (Test-Path $sentinelPath)) { Start-Sleep -Seconds 30 }

    $problem = Validate-Job $job $sentinelPath $rdirPath
    if ($problem) {
        $msg = "HALT: $name failed validation -- $problem"
        Log $msg
        Write-Status @{ status = 'halted'; reason = $msg; completed_jobs = $completed; current_job = $name }
        exit 1
    }

    Log "$name OK"
    $completed += $name
}

Log "=== MATRIX COORDINATOR DONE: all $($JOBS.Count) jobs completed and validated ==="
Write-Status @{ status = 'completed'; finished = (Get-Date).ToString('o'); completed_jobs = $completed; current_job = $null }
