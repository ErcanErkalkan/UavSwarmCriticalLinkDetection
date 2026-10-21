# Controlled execution sequence

1. Read `README_TR.md` and `PUBLICATION_GATE.md` before using any PowerShell launcher.
2. Let Google Drive for Desktop completely sync the original E: working directory. Verify no old queue or running C2 process before any operations.
3. On a **copy** of this snapshot, run `python 03_INDEPENDENT_AUDIT/verify_release.py`. A PASS only certifies episode-archive structure.
4. Run `python -m unittest discover -s tests -v` **from `01_ACTIVE_SOURCE`**; do not run from an arbitrary parent directory.
5. Use the original GitHub/Drive active project for any new experiments, NOT the frozen snapshot. Check SHA and changes before running.
6. Continue missing A2 130 scenario-seed units at six immutable shards with at most 2 or 3 live workers per campaign if system resources demand. Inspect source runbook; restarting can terminate existing Python worker processes.
7. Rerun C1 six-shard canonical merge; compare hashes / unit counts; then C2 serial when CPU free.
8. Only after coherent P0.6 gate status and A1/A3 provenance checks, compute D1-D4/E from verified **raw step files**.
9. Independently verify paired-by-scenario/seed comparisons, bootstrap CIs, failure prevalence, and runtime scope.
10. Rebuild manuscript and figures; do not promote legacy publication claims based on filename recency alone.

**Safety**: The copied launchers are REFERENCE ONLY and can stop jobs; do not execute them inside this release. Run against the active Windows project after inspecting process ownership and backups.
