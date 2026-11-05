# Independent audits — archival and current

- `verify_release.py` verifies **18 October 2026** frozen 8,070-row episode snapshot, only when the external raw snapshot files are supplied. The script is deliberately not rewritten to assert 8,720 rows.
- `../09_VALIDATED_EARLY_WARNING_20261105/verify_S2_onset_v2.py` verifies the **S2 40-unit ZIP**, each sidecar SHA256, paired step labels, onset confusion and whole-seed bootstrap. Original external ZIP required.
- S5.6.6/S5.6.7 exploratory controller checks are described in `../10_EXPLORATORY_SAFETY_CONTROLLER_20261104/`, without a safety or hard-real-time certification.

The November seven-campaign recorded integrity gate is not interchangeable with historical executed-worker code fingerprint completeness. Recompute from original artifacts when needed. Integrity of a data archive does not validate all scientific claims.

[Current evidence ledger](../00_GOVERNANCE/CURRENT_EVIDENCE_20261105.md).
