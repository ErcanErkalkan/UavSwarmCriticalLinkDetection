# ACLD verified S2 early-warning evidence — 5 November 2026

The S2 validation file is a **40-unit synthetic archive**, not an on-drone or physical RF measurement. Ten unique seeds are shared across four fixed scenario strata. Every original CSV is checked against its archived SHA-256 sidecar. The at-risk onset evaluation is **post-hoc**.

- [Original S2 ZIP on Google Drive](https://drive.google.com/file/d/1qUnVkWwsHh0IOBIIhqrkZCFi_7FwSLe6/view)
- ZIP SHA256: `3295d3fdafd539e90f9ce9f75355b3dbd6d0296a4c0a330bf2e1c27051cb4e05`
- Recorded 159,960 transitions; 3,096 current-connected states; 46 next-step new partition onsets.
- At-risk negative support 3,050 observations. ACLD TP45/FN1/FP117/TN2933, predictive Tarjan TP46/FN0/FP712/TN2338, components-only TP40/FN6/FP4/TN3046.
- Bootstrap: whole-seed cluster resampling, 5,000 replicates, 10 seed clusters (do not resample 159,960 temporally dependent steps as IID).

To reproduce the recorded verification after downloading the unmodified original ZIP:

```bash
python -m pip install -r requirements.txt
python research_hub/09_VALIDATED_EARLY_WARNING_20261105/verify_S2_onset_v2.py --s2 ACLD_S2_40_OF_40_FINAL_20261102.zip --out ./audit_s2
```

Compare `audit_s2/S2_onset_verified_summary.json` with the checked-in `S2_onset_verified_summary.json`. The script verifies **recorded data**, not a new simulation run. Early warning is distinguished from bridge localization, where ACLD and guarded predictive Tarjan have identical forecast bridge sets.

The original ZIP (about 9.4 MB) is intentionally stored outside the public GitHub source to preserve the repository's Drive-only data policy.
