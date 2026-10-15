param(
    [ValidateSet(2,3)]
    [int]$ActivePerCampaign = 3
)
$ErrorActionPreference = 'Stop'
$Project = 'E:\Google Drive\My Drive\01_ACTIVE_RESEARCH\FANET_UAV_RESEARCH\02_FANET_ADAPTIVE_CRITICAL_LINK_DETECTION'
$Python = (Get-Command python -ErrorAction Stop).Source
$NewQueue = Join-Path $Project 'run_acld_p0_6_throttled_queue.py'
$Logs = Join-Path $Project 'acld_p0_6_logs_PARALLEL'
$Lock = Join-Path $Logs 'THROTTLED_QUEUE.lock'
if (-not (Test-Path -LiteralPath $NewQueue)) { throw "Missing $NewQueue; finish Google Drive sync first" }
Set-Location -LiteralPath $Project

# Never change the six-way shard count. Lower only active process concurrency.
& $Python -u $NewQueue --project-root $Project --active-per-campaign $ActivePerCampaign --check-only
if ($LASTEXITCODE -ne 0) { throw '6-shard preflight failed; nothing was stopped' }

$Pattern = 'run_acld_p0_6_(parallel_queue|independent_queue|throttled_queue|unit_worker)\.py|run_acld_4day_queue\.py'
$Processes = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and [string]$_.CommandLine -match $Pattern
})
$Elsewhere = @($Processes | Where-Object { [string]$_.CommandLine -notlike "*$Project*" })
if ($Elsewhere.Count) {
    $Elsewhere | Select-Object ProcessId,CommandLine | Format-List
    throw 'P0.6 processes in another project location. Refusing to start another queue.'
}
$C2 = @($Processes | Where-Object { $_.CommandLine -match '--campaign\s+C2\b' })
if ($C2.Count) { throw 'C2 already running: cannot change concurrency during serial benchmark.' }

$SuspendManifest = Join-Path $Logs 'SUSPENDED_PROCESSES.json'
if ((Test-Path -LiteralPath $SuspendManifest) -and $Processes.Count) {
    $s = Get-Content -LiteralPath $SuspendManifest -Raw | ConvertFrom-Json
    if ($s.state -eq 'SUSPENDED') {
        throw 'Suspended processes exist. Resume with python .\pause_resume_acld_p0_6.py resume before switching.'
    }
}
if (Test-Path -LiteralPath $Lock) {
    throw "Throttle lock exists: $Lock. Verify no existing queue, then clear a stale lock manually."
}
if ($Processes.Count) {
    Write-Host 'Existing ACLD processes:' -ForegroundColor Yellow
    $Processes | Select-Object ProcessId,CommandLine | Format-List
    Write-Warning 'Current in-memory seed may restart. Saved CSV results are kept unchanged.'
    $Answer = Read-Host "Stop these processes and run $ActivePerCampaign active workers per campaign? Enter EVET"
    if ($Answer -cne 'EVET') { Write-Host 'Cancelled: no processes stopped'; return }
    # Stop controllers first so they cannot relaunch terminated workers.
    $Controllers = @($Processes | Where-Object {
        $_.CommandLine -match 'run_acld_p0_6_(parallel_queue|independent_queue|throttled_queue)\.py|run_acld_4day_queue\.py'
    })
    foreach ($p in $Controllers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    $Workers = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'run_acld_p0_6_unit_worker\.py' -and
        $_.CommandLine -like "*$Project*"
    })
    foreach ($p in $Workers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}
$Remaining = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and [string]$_.CommandLine -match $Pattern
})
if ($Remaining.Count) { throw 'Old P0.6 processes still exist; nothing new was launched.' }
if (Test-Path -LiteralPath $Lock) { throw 'Throttled lock exists; nothing new was launched.' }

New-Item -ItemType Directory -Path $Logs -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $Logs "throttled_${ActivePerCampaign}_${stamp}_stdout.log"
$stderr = Join-Path $Logs "throttled_${ActivePerCampaign}_${stamp}_stderr.log"
$Arguments = @('-u', ('"' + $NewQueue + '"'), '--project-root', ('"' + $Project + '"'),
    '--active-per-campaign', [string]$ActivePerCampaign, '--max-retries', '3')
$p = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Project `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content -LiteralPath (Join-Path $Logs 'QUEUE.pid') -Value $p.Id -Encoding ASCII
Write-Host "Throttled ACLD queue PID=$($p.Id). A2=$ActivePerCampaign, C1=$ActivePerCampaign, total=$($ActivePerCampaign * 2)." -ForegroundColor Green
Write-Host 'Shard count remains 6, existing CSVs preserved, C2 runs serially after both PASS.'
Write-Host "Status: $(Join-Path $Logs 'QUEUE_STATUS.json')"
