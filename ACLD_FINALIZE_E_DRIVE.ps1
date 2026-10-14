# ACLD P0.6 FINALIZER - E DRIVE / Windows PowerShell 5.1
$ErrorActionPreference = 'Stop'
$Project = "E:\Google Drive\My Drive\01_ACTIVE_RESEARCH\FANET_UAV_RESEARCH\02_FANET_ADAPTIVE_CRITICAL_LINK_DETECTION"
if (-not (Test-Path -LiteralPath $Project -PathType Container)) { throw "Project missing: $Project" }
Set-Location -LiteralPath $Project

$processes = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    [string]$_.CommandLine -match 'run_acld_p0_5_worker\.py|run_acld_p0_6_(parallel_queue|unit_worker|independent_queue)\.py|run_acld_4day_queue\.py' -and
    [string]$_.CommandLine -like '*02_FANET_ADAPTIVE_CRITICAL_LINK_DETECTION*'
})
if ($processes.Count -gt 0) {
    $processes | Select-Object ProcessId,CommandLine | Format-List
    throw 'Workers or queue still active. Finalization refused.'
}

$qfile = Join-Path $Project 'acld_p0_6_logs_PARALLEL\QUEUE_STATUS.json'
if (-not (Test-Path -LiteralPath $qfile)) { throw "Missing queue status: $qfile" }
$q = Get-Content -LiteralPath $qfile -Raw | ConvertFrom-Json
if ($q.state -ne 'WAITING_FOR_A1_FINALIZATION') {
    throw "Queue not ready for finalization: $($q.state)"
}

foreach ($dir in @('A2_detection_rician','B1_closed_loop_nominal','B2_closed_loop_stress','C1_accuracy_scaling','C3_gcs_sensitivity')) {
    $report = Join-Path $Project "acld_full_experiment_outputs_P0_6\$dir\_merge_report.json"
    if (-not (Test-Path -LiteralPath $report)) { throw "Missing report: $report" }
    $gate = (Get-Content -LiteralPath $report -Raw | ConvertFrom-Json).gate
    if ($gate -ne 'PASS') { throw "$dir merge gate is $gate, not PASS" }
}
$c2 = Join-Path $Project 'acld_full_experiment_outputs_P0_6\C2_runtime_scaling\runtime_scaling_full.csv'
if (-not (Test-Path -LiteralPath $c2)) { throw "Missing C2 result: $c2" }

$py = (Get-Command python -ErrorAction Stop).Source
& $py -u (Join-Path $Project 'finalize_ACLD_P0_6_AFTER_A1.py') --project-root $Project --a1-workers 6 --a3-workers 8
if ($LASTEXITCODE -ne 0) { throw 'Finalization failed. Check acld_p0_6_logs_INDEPENDENT.' }

$final = Join-Path $Project 'acld_full_experiment_outputs_P0_6\E_manuscript_ready_FINAL\_postprocess_report.json'
if (-not (Test-Path -LiteralPath $final)) { throw "Missing final evidence report: $final" }
$gate = (Get-Content -LiteralPath $final -Raw | ConvertFrom-Json).gate
if ($gate -ne 'PASS') { throw "Final evidence gate: $gate" }
Write-Host 'ACLD P0.6 finalization: PASS' -ForegroundColor Green
