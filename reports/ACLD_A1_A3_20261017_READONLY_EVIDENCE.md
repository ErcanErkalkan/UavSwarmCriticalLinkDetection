# ACLD A1 + A3 — 17 Oct 2026 read-only evidence summary

Scope: A1, A3 paired result audit. **This is NOT canonical P0.6 finalization, nor publication-ready inferential evidence.** No experiments rerun and no canonical results overwritten.

## Episode-level provenance
- A1: 27 scenarios x 20 seeds = 540 complete scenario-seed units, five methods = 2700 unique scenario-seed-method rows. 70 reused P0.4 units and 470 P0.5 worker units; P0.5 takes precedence on repeated keys.
- A3: 3 scenarios x 20 seeds = 60 complete scenario-seed units, four methods = 240 unique rows from eight P0.5 seed-shard workers.
- P0.6 A1/A3 canonical merge gates are **not yet reported PASS**. A2 is still running; C1 is 100/100 at the worker level, but canonical gate awaits campaign merge.
- B1/B2/C3 canonical PASS has been checked; C2 serial runtime benchmark remains pending.

## Mean raw episode metrics
| Campaign / method | PEW F1 | FPR | Edge F1 |
|---|---:|---:|---:|
| A1 ACLD | 0.9994456 | 0.0409318 | 0.9950866 |
| A1 Tarjan predict | 0.9972907 | 0.2570250 | 0.9950866 |
| A1 Tarjan snapshot | 0.9268178 | 0.2486585 | 0.9745473 |
| A3 ACLD full | 0.9994698 | 0.0326296 | 0.9951631 |
| A3 no guard | 0.9963315 | 0.0326296 | 0.7451941 |
| A3 no forecast bridge | 0.9994351 | 0.0343333 | 0.1073426 |
| A3 no adaptive threshold | 0.9961201 | 0.0000000 | 0.6935047 |

## Paired inference (exploratory)
Differences are ACLD minus comparator at identical scenario+seed. Repeated scenarios are averaged within each of 20 seed clusters; 12,000 seed-cluster bootstrap resamples estimate percentile 95% CIs. Two-sided seed-block sign flips use 40,000 Monte Carlo draws. Holm multiple-testing corrections applied within each campaign.

| Pair / metric | ACLD minus comparator | 95% cluster bootstrap CI | Approx Holm p |
|---|---:|---|---:|
| A1 vs Tarjan-predict / PEW F1 | +0.0021549 | [+0.0019482,+0.0023606] | 0.00015 |
| A1 vs Tarjan-predict / FPR | -0.2160932 | [-0.2340495,-0.1974634] | 0.00015 |
| A1 vs Tarjan-predict / Edge F1 | 0.0000000 | [0,0] | 1.0 |
| A3 vs no guard / Edge F1 | +0.2499690 | [+0.2466930,+0.2531360] | 0.000225 |
| A3 vs no forecast bridge / Edge F1 | +0.8878205 | [+0.8852496,+0.8904322] | 0.000225 |
| A3 vs no adaptive threshold / Edge F1 | +0.3016584 | [+0.2976499,+0.3060050] | 0.000225 |

Scientific interpretation: A1 supports reduced false-positive warnings with ACLD, **not a superior Edge F1 over Tarjan-predict**. A3 supports the importance of forecast bridge and the other tested components for edge-level detection. High positive-event prevalence can make PEW F1 less discriminating. B1/B2 closed-loop connectedness improvements are small and must not be overstated.

## Release gates
Do not rerun A1/A3. Do not run `finalize_ACLD_P0_6_AFTER_A1.py` while A2 or C2 is active. After A2/C1 canonical PASS and C2 serial completion, process A1 P0.5 merge, A1 P0.6 promotion, A3 P0.6 merge, then D1–D4 and E reporting. Existing finalizer already encodes this order. A3 is included in manuscript detection summaries but is **not automatically included in D1's A1+A2+C1 significance suite**; the paired A3 contrasts above are a separate evidence audit.

Source folders:
- [A1 P0.5](https://drive.google.com/drive/folders/1jkFAs-1osNGJs2m9wgRiDTThyNn4p8PU)
- [A1 P0.4](https://drive.google.com/drive/folders/1j15hg9TOVIa83J0mHAPZqIDbmRFVzwOC)
- [A3 P0.5 seed8](https://drive.google.com/drive/folders/1VOaRJOx8NzxtKR89782SpxUEZuQrkJT_)
- [P0.6 canonical folder](https://drive.google.com/drive/folders/1Yv3AZC9dHRHVXxYBewWvA_dsZy46x2lD)
