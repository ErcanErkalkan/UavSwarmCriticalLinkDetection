# Reproduction & Windows process safety

**Do not run worker restart scripts out of the archive.** The original PowerShell launchers and live results are in the **active Drive root**. Restarting can stop active jobs, duplicate partially computed seeds, or collide with existing worker shards.

On the actual Windows machine, first synchronize source, inspect live processes and worker manifests, back up partial append CSVs, and run unit tests. Preserve immutable six-way shard naming; only reduce concurrency (2/3 active workers per campaign), not `--workers 6`. Run C2 only after workers have stopped and canonical checks pass.

See `00_GOVERNANCE/RUN_ORDER.md` for non-destructive execution order.
