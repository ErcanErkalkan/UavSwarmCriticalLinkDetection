param(
    [ValidateSet(6)]
    [int]$WorkersPerCampaign = 6
)
$ErrorActionPreference = "Stop"

$Project = "E:\Google Drive\My Drive\01_ACTIVE_RESEARCH\FANET_UAV_RESEARCH\02_FANET_ADAPTIVE_CRITICAL_LINK_DETECTION"
$Python = (Get-Command python -ErrorAction Stop).Source
$QueueFile = Join-Path $Project "run_acld_p0_6_parallel_queue.py"
$MigrateFile = Join-Path $Project "acld_migrate_2_to_6.py"
$Required = @(
    "acld_migrate_2_to_6.py",
    "run_acld_p0_6_parallel_queue.py",
    "run_acld_p0_6_unit_worker.py",
    "merge_acld_p0_6_campaign.py",
    "acld_p0_6_common.py",
    "acld_full_experimental_campaign_P0_4_corrected.ipynb"
)

if (-not (Test-Path -LiteralPath $Project -PathType Container)) { throw "Project not found: $Project" }
foreach ($Name in $Required) {
    if (-not (Test-Path -LiteralPath (Join-Path $Project $Name))) {
        throw "Required project file is missing: $Name"
    }
}
Set-Location -LiteralPath $Project

# The old 2-worker queue may still be active under the previous C: Google Drive path.
# Never start new E: workers while a separate copy is active.
$Match = 'run_acld_p0_6_(parallel_queue|independent_queue|unit_worker)\.py|run_acld_4day_queue\.py'
$All = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and [string]$_.CommandLine -match $Match
})
$Other = @($All | Where-Object {
    [string]$_.CommandLine -notlike "*$Project*"
})
if ($Other.Count -gt 0) {
    Write-Host "P0.6 processes are running from another project path!" -ForegroundColor Red
    $Other | Select-Object ProcessId,Name,CommandLine | Format-List
    throw "STOP: Do not switch to E: until the other copy is stopped and E: is synchronized."
}

$Manifest = Join-Path $Project "acld_p0_6_logs_PARALLEL\SUSPENDED_PROCESSES.json"
if (Test-Path -LiteralPath $Manifest) {
    $State = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
    if ($All.Count -gt 0 -and $State.state -eq "SUSPENDED") {
        throw "P0.6 processes are suspended. Resume them first with: python .\pause_resume_acld_p0_6.py resume"
    }
}

if ($All.Count -gt 0) {
    Write-Host "These E: P0.6 processes must be stopped before resharding:" -ForegroundColor Yellow
    $All | Select-Object ProcessId,Name,CommandLine | Format-List
    Write-Warning "The seed currently being calculated may have to restart; fully saved CSVs are preserved."
    $Answer = Read-Host "To stop only these P0.6 processes and switch to 6 workers/campaign, type EVET"
    if ($Answer -cne "EVET") {
        Write-Host "Cancelled. No process stopped or file modified."
        exit 0
    }
    # Controller first so it cannot launch/retry workers after we stop them.
    $Controllers = @($All | Where-Object { $_.CommandLine -match 'parallel_queue\.py|independent_queue\.py|4day_queue\.py' })
    foreach ($p in $Controllers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
    $Workers = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -match 'run_acld_p0_6_unit_worker\.py' -and
        $_.CommandLine -like "*$Project*"
    })
    foreach ($p in $Workers) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}
$Remaining = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and [string]$_.CommandLine -match $Match
})
if ($Remaining.Count -gt 0) {
    $Remaining | Select-Object ProcessId,Name,CommandLine | Format-List
    throw "Old P0.6 processes still exist. No migration or relaunch was done."
}

# Finished campaigns must already be PASS on E:, otherwise the queue would rerun them.
foreach ($FinishedDir in @("B1_closed_loop_nominal", "B2_closed_loop_stress", "C3_gcs_sensitivity")) {
    $FinishedReport = Join-Path $Project "acld_full_experiment_outputs_P0_6\$FinishedDir\_merge_report.json"
    if (-not (Test-Path -LiteralPath $FinishedReport)) { throw "Missing PASS report: $FinishedReport" }
    $FinishedGate = (Get-Content -LiteralPath $FinishedReport -Raw | ConvertFrom-Json).gate
    if ($FinishedGate -ne "PASS") { throw "Finished campaign not PASS: $FinishedDir [$FinishedGate]" }
}

# Recreate up-to-date partial canonical merge from the existing TWO shard folders.
# An INCOMPLETE gate is expected until the missing seeds are finished.
foreach ($Campaign in @("A2", "C1")) {
    Write-Host "Canonical 2-worker merge: $Campaign" -ForegroundColor Cyan
    & $Python -u (Join-Path $Project "merge_acld_p0_6_campaign.py") `
        --project-root $Project --campaign $Campaign --workers 2
    if ($LASTEXITCODE -ne 0) { throw "2-worker merge failed for $Campaign" }
}

Write-Host "Repartitioning legacy data without modifying old CSVs..." -ForegroundColor Cyan
& $Python -u $MigrateFile --project-root $Project --min-a2 529 --min-c1 92
if ($LASTEXITCODE -ne 0) { throw "Resharding failed. No new queue was started." }

# New queue uses fresh six-worker folders; B1/B2/C3 remain PASS.
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:NUMEXPR_NUM_THREADS="1"

$Logs = Join-Path $Project "acld_p0_6_logs_PARALLEL"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$stamp=Get-Date -Format "yyyyMMdd_HHmmss"
$outFile=Join-Path $Logs "queue_6workers_${stamp}_stdout.log"
$errFile=Join-Path $Logs "queue_6workers_${stamp}_stderr.log"
$arguments=@(
    '-u', ('"' + $QueueFile + '"'),
    '--project-root', ('"' + $Project + '"'),
    '--workers-per-campaign', [string]$WorkersPerCampaign,
    '--max-retries', '3'
)
$p=Start-Process -FilePath $Python -ArgumentList $arguments `
    -WorkingDirectory $Project -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile
Set-Content -LiteralPath (Join-Path $Logs "QUEUE.pid") -Value $p.Id -Encoding ASCII
Write-Host "P0.6 queue started: PID=$($p.Id)" -ForegroundColor Green
Write-Host "A2=6 workers, C1=6 workers, max simultaneous=12"
Write-Host "Old 2-worker output folders are preserved. C2 runs serially afterward."
Write-Host "Logs: $outFile"
