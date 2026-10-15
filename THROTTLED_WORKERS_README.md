# ACLD throttled worker restart

**Do not change `--workers 6` to 2 or 3 directly.** P0.6 CSVs are already partitioned among six immutable worker-shard directories for each of A2 and C1. Changing shard count would create a new naming scheme and omit existing seed-level outputs from canonical six-shard merges.

The new queue keeps six shards but allows a **maximum of 2 or 3 live processes per campaign**. It processes remaining shard indices as earlier ones finish, merges A2 and C1 with `--workers 6`, checks `PASS`, and then runs C2 serially.

## On the Windows E: Google Drive machine

1. Ensure Google Drive sync has finished. Open PowerShell in the existing project folder.
2. Update repository code **before stopping existing experiments**: `./ACLD_SYNC_GITHUB.ps1` (without `-Bootstrap`). If it refuses due to tracked changes, resolve them; do not force-reset.
3. Run `./ACLD_RESTART_2_OR_3_WORKERS.ps1 -ActivePerCampaign 3` (6 active Python workers maximum), or choose `-ActivePerCampaign 2` (4 active maximum).
4. Check the console preflight. If old A2/C1 processes exist, the script asks you to enter `EVET`. It stops *only* P0.6 processes in the configured project, first the queue then its workers. Saved CSVs are **not deleted**, but a partly computed seed can be repeated. Force-stop can interrupt an append operation, so ensure local/Drive backup and inspect the resulting CSVs if recovery reports parser errors.
5. Monitor `acld_p0_6_logs_PARALLEL/QUEUE_STATUS.json` and per-shard logs in `A2/throttled/` and `C1/throttled/`. `workers_per_campaign` remains **6** because that is shard count; `active_workers_per_campaign` shows the real concurrency **2** or **3**.

Do not launch the previous `ACLD_6_WORKER_SAFE_RESUME.ps1` afterwards; it migrates from legacy 2-way shards and may collide with this queue.

**Limitations:** Success of the static Python/unit tests does not certify end-to-end recovery against actual Windows worker processes and gigabyte CSVs. The status file timestamp is written by the local running process. An existing suspended queue must first be resumed. Do not use this script while C2 is running. If first-run PowerShell commands encounter a stale lock, inspect running processes before removing it manually.
