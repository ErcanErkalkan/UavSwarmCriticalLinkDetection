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
