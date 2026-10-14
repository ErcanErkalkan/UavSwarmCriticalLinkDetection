param([switch]$Bootstrap)
$ErrorActionPreference = 'Stop'
$Project = 'E:\Google Drive\My Drive\01_ACTIVE_RESEARCH\FANET_UAV_RESEARCH\02_FANET_ADAPTIVE_CRITICAL_LINK_DETECTION'
$Remote = 'https://github.com/ErcanErkalkan/UavSwarmCriticalLinkDetection.git'
if (-not (Test-Path -LiteralPath $Project -PathType Container)) { throw "Project not found: $Project" }
Set-Location -LiteralPath $Project
$null = Get-Command git -ErrorAction Stop
$Required = @('README.md', '.gitignore', 'pyproject.toml', 'requirements.txt',
  'ACLD_6_WORKER_SAFE_RESUME.ps1', 'acld_migrate_2_to_6.py',
  'run_acld_p0_6_parallel_queue.py', 'run_acld_p0_6_unit_worker.py',
  'acld_full_experimental_campaign_P0_4_corrected.ipynb')
foreach ($File in $Required) {
    if (-not (Test-Path -LiteralPath (Join-Path $Project $File))) {
        throw "Google Drive sync incomplete; missing: $File"
    }
}
if (-not (Test-Path -LiteralPath '.git')) {
    if (-not $Bootstrap) { throw 'First synchronization: .\ACLD_SYNC_GITHUB.ps1 -Bootstrap' }
    & git init -b main
    if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
    & git remote add origin $Remote
    if ($LASTEXITCODE -ne 0) { throw 'git remote add failed' }
    & git fetch origin main
    if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
    # git reset --mixed updates Git tracking ONLY; it never overwrites experiment files.
    & git reset --mixed origin/main
    if ($LASTEXITCODE -ne 0) { throw 'git index setup failed' }
} else {
    $Origin = (& git remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or $Origin -notin @($Remote, 'git@github.com:ErcanErkalkan/UavSwarmCriticalLinkDetection.git')) {
        throw "Unexpected Git remote: $Origin"
    }
    & git fetch origin main
    if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
    $Dirty = @(& git status --porcelain --untracked-files=no)
    if ($Dirty.Count -gt 0) {
        $Dirty | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
        throw 'Local changes found. No automatic pull; reconcile manually.'
    }
    & git merge --ff-only origin/main
    if ($LASTEXITCODE -ne 0) { throw 'Fast-forward failed, review local history.' }
}
$Dirty = @(& git status --porcelain --untracked-files=no)
if ($Dirty.Count -gt 0) {
    $Dirty | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    throw 'Some tracked files are not the GitHub version. DO NOT run workers until reconciled.'
}
$Head = (& git rev-parse HEAD).Trim()
$RemoteHead = (& git rev-parse origin/main).Trim()
if ($Head -ne $RemoteHead) { throw 'Local commit differs from origin/main' }
Write-Host "Code is synchronized. GitHub commit: $Head" -ForegroundColor Green
Write-Host 'Experiment CSVs, status and logs remain only in Google Drive.'
