# FANET – Adaptive Critical Link Detection | Unified research hub

**Applied:** 2026-10-21. **Evidence cut-off:** 2026-10-18 (archival snapshot). **Scientific gate:** NOT READY.

* **Active source:** [https://github.com/ErcanErkalkan/UavSwarmCriticalLinkDetection](https://github.com/ErcanErkalkan/UavSwarmCriticalLinkDetection) (repository root; Python module paths unchanged)
* **Master scientific evidence:** [Drive unified release](https://drive.google.com/drive/folders/1lj_EmE6O2wTQU1oO0osqeyZRiiwsF7wd)
* **Original live working root:** [ACLD Drive project](https://drive.google.com/drive/folders/11BHmFCX2iTgPhU4M2F0v5yDGsEHKMz1p)

| Area | Drive physical folder |
|---|---|
| `00_GOVERNANCE` | [Drive directory](https://drive.google.com/drive/folders/1155clTncYrZKkUhpZLfq0bQsbg2XknMV) |
| `01_ACTIVE_SOURCE` | [Drive directory](https://drive.google.com/drive/folders/1f0YSA4OnjhJsUSMJx-h8cOBb9-z51Wnq) |
| `02_CANONICAL_SNAPSHOT_20261018` | [Drive directory](https://drive.google.com/drive/folders/1equBSQcn6vRgmNe3BGDlKlMawTQkMfG6) |
| `03_INDEPENDENT_AUDIT` | [Drive directory](https://drive.google.com/drive/folders/1EU39885NdVGWkICNXvvltuwZ414B45ZG) |
| `04_DESCRIPTIVE_ANALYSIS` | [Drive directory](https://drive.google.com/drive/folders/14G55FFAvd5kP1er26E0gwwSLTiZ6v0g9) |
| `05_MANUSCRIPT_WORKING` | [Drive directory](https://drive.google.com/drive/folders/1quHOw_xXA2gAeNO3dscG3Q00fmiV-SHQ) |
| `06_REPRODUCTION_RUNBOOK` | [Drive directory](https://drive.google.com/drive/folders/1ZVnibMypQnbfx6nfjkDXUEtI0lpLHDYg) |
| `07_LARGE_DATA_REFERENCES` | [Drive directory](https://drive.google.com/drive/folders/1qVOZJ8fBSZDavkBpZSlBe_Ck6mK8VleV) |
| `08_DELIVERABLES` | [Drive directory](https://drive.google.com/drive/folders/1-HnqRZiuaGg3c_LJ0znut0tDpwLWZHkQ) |

## Operating model

GitHub tracks executable source, notebook, tests and lightweight governance/reproduction documentation; original Drive root holds big raw experiment files, worker logs, intermediate results and the manuscript/figures. The nine subfolders here provide a navigable documentary architecture **without moving or duplicating working module paths**.

The frozen 18 October 2026 combined episode snapshot includes **8,070 method/controller rows**, comprising 6,870 detection and 1,200 mitigation rows; **1,690/1,820 seed units**. A2 is **590/720**, while C1 episode inventory is **100/100** but canonical merge PASS is pending. The manuscript remains a working revision, not a certified final submission. This architecture change did not execute simulations, restart Windows workers, or mutate existing experiment results.

### Validation

Run `python research_hub/03_INDEPENDENT_AUDIT/verify_release.py` **from the complete downloaded Drive release only** (the script requires a sibling `02_CANONICAL_SNAPSHOT_20261018` folder). GitHub's documentation-only checkout intentionally has no raw episode files, so this script should not be run on GitHub alone.
