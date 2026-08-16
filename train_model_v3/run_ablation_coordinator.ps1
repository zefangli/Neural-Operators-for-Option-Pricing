# Two-job coordinator for the scalar-sigma ablation (call + put, vol_surface,
# seed 42, canonical v5). Same detached-process pattern as
# run_matrix_coordinator.ps1 / run_replication_coordinator.ps1 -- itself must be
# launched via Start-Process -WindowStyle Hidden so it survives its own launcher
# being stopped. Reuses run_detached.ps1 unmodified.

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$CALL_H5  = "..\wrds_data_2020-2025\deeponet_tensors_call_v5.h5"
$PUT_H5   = "..\wrds_data_2020-2025\deeponet_tensors_put_v5.h5"
$CALL_SHA = "ecf94de2996e3212ac791f944f847142e7a809c640b24cbd5989e8bdba801153"
$PUT_SHA  = "46f3ee9d03d3423c6330c2b35a4d09893b0f538bdc046bc1018ebab79861c756"

$MIN_RAM_GB = 2.0
$MIN_PAGEFILE_GB = 8.0
$RESOURCE_RETRY_DELAY_SEC = 300
$RESOURCE_RETRY_MAX = 24

$JOBS = @(
    @{side='call'; script='train_vol_surface_scalarsigma.py'; rdir='results_vol_surface_scalarsigma_v5'},
    @{side='put';  script='train_vol_surface_scalarsigma.py'; rdir='results_vol_surface_scalarsigma_v5'}
)

$CoordLog = "runs\ablation_coordinator.log"
$CoordStatus = "runs\ablation_coordinator.status.json"
New-Item -ItemType Directory -Force -Path "runs","jobs" | Out-Null

function Log($msg) {
    "[$((Get-Date).ToString('o'))] $msg" | Out-File -FilePath $CoordLog -Append -Encoding Unicode
}
function Write-Status($obj) {
    $tmp = "$CoordStatus.tmp"
    ($obj | ConvertTo-Json -Depth 5) | Set-Content -Path $tmp -Encoding Unicode
    Move-Item -Path $tmp -Destination $CoordStatus -Force
}
function Get-ResourceHeadroom {
    $os = Get-CimInstance Win32_OperatingSystem
    $ramGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $pfTotal = 0.0; $pfUsed = 0.0
    foreach ($pf in (Get-CimInstance Win32_PageFileUsage)) {
        $pfTotal += $pf.AllocatedBaseSize; $pfUsed += $pf.CurrentUsage
    }
    $pageGB = [math]::Round(($pfTotal - $pfUsed) / 1024, 2)
    [PSCustomObject]@{ RamFreeGB = $ramGB; PageFreeGB = $pageGB }
}
function Wait-ForResources($jobName) {
    for ($i = 0; $i -lt $RESOURCE_RETRY_MAX; $i++) {
        $r = Get-ResourceHeadroom
        Log ("resource check before $jobName -- RAM free {0} GiB, pagefile free {1} GiB" -f $r.RamFreeGB, $r.PageFreeGB)
        if ($r.RamFreeGB -ge $MIN_RAM_GB -and $r.PageFreeGB -ge $MIN_PAGEFILE_GB) { return $r }
        Log "  below threshold -- waiting ${RESOURCE_RETRY_DELAY_SEC}s"
        Start-Sleep -Seconds $RESOURCE_RETRY_DELAY_SEC
    }
    return $null
}
function Validate-Job($job, $sentinelPath, $rdirPath) {
    if (-not (Test-Path $sentinelPath)) { return "no sentinel at $sentinelPath" }
    $sentinel = Get-Content $sentinelPath -Raw -Encoding Unicode | ConvertFrom-Json
    if ($sentinel.exit_code -ne 0) { return "exit_code=$($sentinel.exit_code)" }
    $cfgPath = Join-Path $rdirPath "config.json"
    if (-not (Test-Path $cfgPath)) { return "missing config.json" }
    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
    if ($cfg.dataset_version -ne 'v5') { return "dataset_version=$($cfg.dataset_version)" }
    if ($cfg.quote_filter -ne 'static_bounds_midpoint') { return "quote_filter=$($cfg.quote_filter)" }
    $expectSha = if ($job.side -eq 'call') { $CALL_SHA } else { $PUT_SHA }
    if ($cfg.h5_sha256 -ne $expectSha) { return "h5_sha256 mismatch" }
    if ($cfg.seed -ne 42) { return "seed=$($cfg.seed)" }
    foreach ($ckpt in @('best_model_phase1.pth','best_model.pth','final_model.pth')) {
        if (-not (Test-Path (Join-Path $rdirPath $ckpt))) { return "missing checkpoint $ckpt" }
    }
    $lossPath = Join-Path $rdirPath "loss_history.txt"
    if (-not (Test-Path $lossPath)) { return "missing loss_history.txt" }
    $lossText = Get-Content $lossPath -Raw
    if ($lossText -match '(?i)\bnan\b|\binf\b') { return "non-finite value in loss_history.txt" }
    if ($lossText -notmatch 'Test R\^2') { return "no test metrics block" }
    return $null
}

Log "=== ABLATION COORDINATOR START: $($JOBS.Count) jobs ==="
Write-Status @{ status='running'; started=(Get-Date).ToString('o'); completed_jobs=@(); current_job=$null }

$completed = @()
foreach ($job in $JOBS) {
    $name = "$($job.side)_train_vol_surface_scalarsigma_v5_seed42"
    $rdirPath = Join-Path $job.side $job.rdir
    $logPath = "runs\$name.log"
    $sentinelPath = "runs\$name.sentinel"
    $jobScriptPath = "jobs\$name.cmd"

    Log "--- $name ---"
    Write-Status @{ status='running'; started=(Get-Date).ToString('o'); completed_jobs=$completed; current_job=$name }

    $preExisting = @()
    foreach ($p in @($rdirPath, $logPath, $sentinelPath, "$sentinelPath.tmp")) {
        if (Test-Path $p) { $preExisting += $p }
    }
    if ($preExisting.Count -gt 0) {
        $msg = "HALT: pre-existing artifact(s) for $name : $($preExisting -join ', ')"
        Log $msg
        Write-Status @{ status='halted'; reason=$msg; completed_jobs=$completed; current_job=$name }
        exit 1
    }

    $res = Wait-ForResources $name
    if ($null -eq $res) {
        $msg = "HALT: resource headroom never recovered before $name"
        Log $msg
        Write-Status @{ status='halted'; reason=$msg; completed_jobs=$completed; current_job=$name }
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
    Write-Status @{ status='running'; started=(Get-Date).ToString('o'); completed_jobs=$completed; current_job=$name; launched_pid=$launch.Id }

    while (-not (Test-Path $sentinelPath)) { Start-Sleep -Seconds 30 }

    $problem = Validate-Job $job $sentinelPath $rdirPath
    if ($problem) {
        $msg = "HALT: $name failed validation -- $problem"
        Log $msg
        Write-Status @{ status='halted'; reason=$msg; completed_jobs=$completed; current_job=$name }
        exit 1
    }
    Log "$name OK"
    $completed += $name
}

Log "=== ABLATION COORDINATOR DONE: $($completed.Count)/$($JOBS.Count) jobs ==="
Write-Status @{ status='completed'; finished=(Get-Date).ToString('o'); completed_jobs=$completed; current_job=$null }
