# Reproduction and Windows worker-process safety — 5 November 2026

1. **Do not restart “missing A2 130 units” or re-launch C2 based on the 18 October historical runbook.** More recent recorded campaigns contain their later results; inspect the active queue manifests, source hashes, process ownership and backup state before executing any worker launcher.
2. The original 18 October `../03_INDEPENDENT_AUDIT/verify_release.py` checks only that **dated snapshot** from an external Drive archive. Its expected 8,070 rows must not be silently edited to 8,720.
3. For the independent synthetic S2 40-unit archive, fetch the [Drive file](https://drive.google.com/file/d/1qUnVkWwsHh0IOBIIhqrkZCFi_7FwSLe6/view), ensure SHA256 `3295d3fdafd539e90f9ce9f75355b3dbd6d0296a4c0a330bf2e1c27051cb4e05` and run:
```bash
python -m pip install -r requirements.txt
python research_hub/09_VALIDATED_EARLY_WARNING_20261105/verify_S2_onset_v2.py --s2 ACLD_S2_40_OF_40_FINAL_20261102.zip --out ./audit_s2
python -m unittest discover -s tests -v
```
4. The S2 onset subset is **retrospective**; seed-cluster bootstrap resamples 10 IDs and not the individual time steps.
5. Source runner code, large Drive raw evidence, later S5 controller source and local manuscript artifacts have different public mirroring policies. A verified summary cannot substitute for absent executable source fingerprints.
6. No flight tests, RF packet trials, 50 ms large-swarm compliance, calibrated deployment probabilities or physical safety certification are claimed.
