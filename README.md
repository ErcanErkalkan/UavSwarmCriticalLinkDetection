# FANET Adaptive Critical Link Detection — single source project

[Source on GitHub](https://github.com/ErcanErkalkan/UavSwarmCriticalLinkDetection)

The **GitHub `main` branch contains all active Python code** and the audited scientific notebook for the P0.5/P0.6 ACLD campaigns. The working folder under **Google Drive for Desktop** mirrors these source files and also stores **large experiment results, original worker shards, merge reports, logs, figures, manuscripts and historical records**. Those scientific outputs stay exclusively in Drive and are excluded from Git.

**Windows project root:**

```text
E:\Google Drive\My Drive\01_ACTIVE_RESEARCH\FANET_UAV_RESEARCH\02_FANET_ADAPTIVE_CRITICAL_LINK_DETECTION
```

## Unified project layout

The repository includes the root-level, interdependent Python modules `acld_p0_5_common.py`, `acld_p0_6_common.py`, `acld_p0_6_io.py`, `run_acld_p0_5_worker.py`, `run_acld_p0_6_unit_worker.py`, `run_acld_p0_6_parallel_queue.py`, `merge_acld_p0_5.py`, `merge_acld_p0_6_campaign.py`, `postprocess_acld_p0_6_FINAL.py`, `finalize_ACLD_P0_6_AFTER_A1.py`, `acld_migrate_2_to_6.py`, `pause_resume_acld_p0_6.py`, and `validate_acld_p0_4_outputs.py`. Scientific experiment definitions and algorithms remain in `acld_full_experimental_campaign_P0_4_corrected.ipynb`.

The root Python paths are **intentionally stable**: the audited notebook loader and experiment workers use relative paths to find the notebook and output directories. The project is **one modular Python package** with one code repository, **not one giant Python file**. The old GitHub `main.ipynb` is retained in Git history but is not the active notebook.

| Component | Source of truth | Stored here |
|---|---|---|
| Python source, PowerShell launcher, notebooks, project config | GitHub `main` | GitHub + E: Drive |
| Raw experiment CSVs, step rows, canonical merges, PASS gates | Google Drive | E: Drive only |
| Worker progress JSON, logs, suspended-process manifests | Google Drive | E: Drive only |
| Manuscript, figures, appendices, reviewer documents | Google Drive | E: Drive only |
| Historical scripts and ZIP archives | Google Drive archival folders | E: Drive only |

## Configure source synchronization on Windows

Wait for Google Drive for Desktop to complete syncing **all source files**. Open Windows PowerShell:

```powershell
$Project = 'E:\Google Drive\My Drive\01_ACTIVE_RESEARCH\FANET_UAV_RESEARCH\02_FANET_ADAPTIVE_CRITICAL_LINK_DETECTION'
Set-Location -LiteralPath $Project
.\ACLD_SYNC_GITHUB.ps1 -Bootstrap  # first use only
```

For later runs, use `.\ACLD_SYNC_GITHUB.ps1` without `-Bootstrap` (or `git pull --ff-only` on a clean branch). The script checks the remote and tracked content and blocks mismatches without deleting raw data. If you edit code locally, use standard `git add`, `git commit`, and `git push` after tests; this is a **controlled Git/GDrive integration**, not unattended Git pushes.

## Before restarting six workers per campaign

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
.\ACLD_SYNC_GITHUB.ps1
.\ACLD_6_WORKER_SAFE_RESUME.ps1
```

The launcher first checks that no conflicting processes are running, validates completed B1/B2/C3 canonical `PASS` gates, merges saved A2/C1 two-worker results, reshares completed seed results into six worker shards **without deleting old CSVs**, and starts **A2 with six and C1 with six**. Once their merges are `PASS`, it starts C2 serially.

**Caution:** 12 workers can overload RAM on C1's n=1000 simulations. A stopped mid-seed unit can start its current seed again; completed CSVs are reused. Never manually delete worker data or move output folders while running. Do not run the migration twice.

## Final postprocessing

After all canonical checks pass and C2 has completed, run `.\ACLD_FINALIZE_E_DRIVE.ps1`. The postprocess script writes canonical manuscript-ready evidence. It does not fabricate outcomes or declare incomplete experiments `PASS`.

## Provenance and maintenance

Scientific P0.4 engine definitions were not changed during repository consolidation. `tests/test_protocol.py` verifies A2=720 and C1=100 units and six-worker partitioning; `.github/workflows/code-integrity.yml` runs source syntax and protocol tests. Data are excluded using `.gitignore`. Do not commit credentials, worker log files, results or unpublished personal data to this public repository.
