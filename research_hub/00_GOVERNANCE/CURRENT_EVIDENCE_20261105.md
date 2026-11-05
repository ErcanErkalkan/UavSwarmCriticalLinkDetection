# ACLD scientific status — 5 November 2026

**Overall gate: NOT_ALL_PASS.** Recorded-data integrity, scientific generalization, calibration, operational controller safety and hard real-time execution are separate gates. The original 18 October snapshot must remain immutable.

## Validated review inventory

| Campaign | Recorded method/controller episode rows |
|---|---:|
| A1 | 2700 |
| A2 | 3600 |
| A3 | 240 |
| B1 | 600 |
| B2 | 600 |
| C1 | 500 |
| C3 | 480 |
| **Total** | **8720** |

All seven *recorded-campaign review integrity* gates passed. A1 includes corrected P0.4/P0.5 historical material and is **not** a newly run homogeneous P0.6 dataset. Historical worker-executable fingerprints are incomplete for some runs. The 18 October archival snapshot (8070 episode rows, 1690 seed units, A2 590/720) is dated evidence, not the latest completion state.

## Defensible early-warning results

- A2: 720 paired scenario–seed units; **identical** forecast-bridge edge sets for ACLD and matched predictive Tarjan; edge micro-F1 both 0.99605. This is **not** evidence of a novel bridge extractor.
- A2 pooled partition-warning false-positive rate: ACLD 2304/59607 = 0.03865; predictive Tarjan 12180/59607 = 0.20434. Episode-mean rates with zero-denominator convention are a different estimand.
- S2: 40 synthetic scenario–seed units, **10 distinct seed identities shared across four strata**, 159960 dependent transitions; 40/40 raw unit CSV SHA256 sidecars verified.
- S2 *post-hoc onset subset* (current graph connected; next step partition): 46 events, 3050 negative at-risk observations. ACLD TP45/FN1/FP117/TN2933; Tarjan TP46/FN0/FP712/TN2338; components-only TP40/FN6/FP4/TN3046. The onset calculation is retrospective, not preregistered.
- S2 source ZIP: [Drive](https://drive.google.com/file/d/1qUnVkWwsHh0IOBIIhqrkZCFi_7FwSLe6/view), SHA256 `3295d3fdafd539e90f9ce9f75355b3dbd6d0296a4c0a330bf2e1c27051cb4e05`.

## Other scientific bounds

- B1/B2 absolute connected-fraction gains: **+0.187** and **+0.046 percentage points**.
- C2 serial core benchmark: 5000 recorded measurements; n=50 median 10.976 ms; n=150 58.015 ms; n=1000 3751.072 ms. This is not large-swarm real-time performance.
- Separately fitted sigmoid mapping: aggregate reliability metrics are influenced by skewed labels; sparse intermediate warning scores are substantially miscalibrated.
- S5.6.6: 672 controlled pairs, 7 selected natural plans and **zero natural vetoes**.
- S5.6.7: 1188 dependent natural states; 334 probes; 13 unique viable episodes; **zero gate vetoes or risk/approved-action overlap**; untouched reserved holdout seeds 410–419. [Drive folder](https://drive.google.com/drive/folders/1g2GM3yYB2rbrRIbREgvRJG2_mCWZ7uR4). S5 is exploratory, **not** an operational safety or hard-real-time guarantee.

## Latest author-controlled manuscript, not hosted in public GitHub

- 5 November 2026 paper: *ACLD: Risk-Gated Partition Early Warning from Forecast Bridges in UAV Swarm Networks*, 49 pages, 34 bibliography entries, 10 newly added Ad Hoc Networks citations.
- PDF SHA256 `60a9b2c1de0a84afdf3c0993b111ce9ffc04928e6cfc757a51a82c181585db84`
- LaTeX SHA256 `27071523b692c5f2cdd755b68072d991cf9006bb0555c4be426a1be504497936`
- BibTeX SHA256 `e99d5d815a81c63ef04644ca2ea976b8daa7ca85ff7b06704a3425550dbf0168`
- Disclosures: no research funding; no competing interests; author-approved AI statement. Submission/acceptance is **not confirmed**.
- Public GitHub retains executable October source and *lightweight metadata*; manuscript PDF/LaTeX, huge raw experiments and current S5 source remain in author-controlled Drive/local releases. Their absence is a repository-mirroring limit, not evidence of a failed test.
- No public DOI has been verified for the complete seven-campaign archive.

## Safety of operations

**Do not follow October “remaining A2 130 units” restart instructions on current live data.** Before starting or resuming any workers, inspect active Drive logs, existing processes, manifests and code revisions. Integrity PASS is not physical safety PASS or universal scientific readiness.
