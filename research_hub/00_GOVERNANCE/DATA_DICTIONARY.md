# Dataset units and definitions

- **Episode-method row**: one `(campaign, scenario, seed, method_or_controller)` record. There are 8,070 unique rows.
- **Seed unit**: one `(campaign, scenario, seed)`; multiple methods/controllers per seed unit. There are 1,690 present and 1,820 planned across A1/A2/A3/B1/B2/C1/C3.
- **PEW-F1**: next-step partition early warning classification F1 (mean is macro across episode rows as here).
- **Edge micro-F1**: computed as `2 ΣTP / (2 ΣTP + ΣFP + ΣFN)` from the exact bridge-localization TP/FP/FN in one campaign/method. Do not substitute `mean(edge_f1_mean)`.
- **FPR**: false positive rate according to method's per-episode labeling scope; `mean(fpr)` is descriptive.
- **acld_pipeline_median_ms**: per-episode full ACLD pipeline median, available for ACLD; averages across episodes are NOT global p50 nor comparable to baselines timed after shared preprocessing.
- **partitions / connected_ratio**: closed-loop, controller-level metrics from B1/B2; not interchangeable with warning detection metrics.
- **protocol_version**: P0.4-reused, P0.5, P0.6. A1 contains 350 P0.4-reused method rows (70 seed units); provenance must not be overwritten.
- **Canonical PASS**: successful campaign merge, different from verified episode completeness or publication gate.

Missing values are intentionally blank when the metric does not apply. No imputation, interpolation, simulated filling, or statistical cross-protocol pooling was performed.

## Current evidence definitions (5 November 2026)

**The 8,070 rows and 1,690 units above describe the immutable 18 October snapshot only.** Current seven-campaign review inventory: 8,720 method/controller episode rows, with A1 corrected P0.4/P0.5 history retained.

- `S2 scenario–seed unit`: 40 paired scenario/seed units, **10 distinct seed IDs** reused across four fixed strata.
- `S2 onset`: current graph is connected (current lambda2 > 1e-10), and the next-step partition label is positive. Forty-six such onsets exist in 3,096 current-connected time steps.
- `S2 onset FPR`: false alarms / 3,050 current-connected, next-step *nonpartition* observations, not the full-step imbalance rate.
- `S2 onset confusion`: ACLD TP45 FN1 FP117 TN2933; guarded predictive Tarjan TP46 FN0 FP712 TN2338; components-only TP40 FN6 FP4 TN3046.
- `Full-label S2 metrics`: 159,960 temporally dependent next-step labels, with 156,904 partition-positive observations.
- `A2 pooled FPR`: summing false positives and negative labels across support-bearing steps (ACLD 2304/59607, Tarjan 12180/59607); not equal to convention-coded mean episode FPR.
- `Bridge localization`: ACLD and guarded predictive Tarjan use the same predicted viable bridge edge set; warning-policy differences are separate predictions.
- `Integrity PASS`: checksum/count consistency of *recorded* data; not complete source provenance, population generalization, hard-real-time, safety or calibrated probability guarantee.
