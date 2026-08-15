# Runs a .cmd job file to completion in the FOREGROUND of this process, then writes
# an atomic completion/failure sentinel. Meant to be launched via
# `Start-Process -WindowStyle Hidden -PassThru powershell.exe -ArgumentList
#   '-File', 'run_detached.ps1', '-SentinelPath', ..., '-LogPath', ..., '-JobScript', ...`
# so the resulting process tree is owned by Windows, not by whatever launched it --
# stopping the launcher (an agent tool call, a terminal, a monitoring loop) must NOT
# kill this process or its child.
#
# Takes a JOB SCRIPT PATH, not an inline command string: Start-Process's
# -ArgumentList mangles embedded quotes with no error surfaced under a hidden
# window. A bare file path has no quoting to get wrong.
#
# Encoding: Unicode (UTF-16LE) throughout -- header, redirected job output, and
# footer all use the same encoding PowerShell's `*>>` redirection defaults to, so
# the log is never a mixed-encoding file.
#
# Sentinel is written atomically: build the JSON, write to a .tmp file, then rename --
# a reader never observes a half-written sentinel.

param(
    [Parameter(Mandatory=$true)][string]$SentinelPath,
    [Parameter(Mandatory=$true)][string]$LogPath,
    [Parameter(Mandatory=$true)][string]$JobScript
)

# Refuse to clobber a prior run's artifacts -- a stale log/sentinel here means
# something already used this path, so silently overwriting it would hide that.
foreach ($p in @($LogPath, $SentinelPath, "$SentinelPath.tmp")) {
    if (Test-Path $p) {
        Write-Error "REFUSING TO START: $p already exists. Remove or rename it first."
        exit 1
    }
}

$ErrorActionPreference = 'Continue'
$start = Get-Date

"[$($start.ToString('o'))] START pid=$PID job=$JobScript" | Out-File -FilePath $LogPath -Encoding Unicode

# cmd.exe /c with a single quoted path argument -- no nested-quoting hazard.
& cmd.exe /c "`"$JobScript`"" *>> $LogPath
$exitCode = $LASTEXITCODE

$end = Get-Date
$status = if ($exitCode -eq 0) { 'completed' } else { 'failed' }

$sentinel = [ordered]@{
    status       = $status
    exit_code    = $exitCode
    job_script   = $JobScript
    pid          = $PID
    start_time   = $start.ToString('o')
    end_time     = $end.ToString('o')
    duration_sec = [math]::Round(($end - $start).TotalSeconds, 1)
} | ConvertTo-Json

$tmp = "$SentinelPath.tmp"
Set-Content -Path $tmp -Value $sentinel -Encoding Unicode
Move-Item -Path $tmp -Destination $SentinelPath -Force

"[$($end.ToString('o'))] END status=$status exit_code=$exitCode" | Out-File -FilePath $LogPath -Append -Encoding Unicode
