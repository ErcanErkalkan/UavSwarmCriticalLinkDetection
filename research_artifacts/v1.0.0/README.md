# ACLD — frozen paper-matched research version v1.0.0-paper

**Title:** ACLD: Risk-Gated Partition Early Warning from Forecast Bridges in UAV Swarm Networks  
**Author:** Ercan Erkalkan, Marmara University  
**Freeze date:** 2026-11-05  
**Version:** `v1.0.0-paper` (prepared; release/tag publication and Zenodo DOI not yet confirmed)

## One frozen research snapshot

The repository commit associated with this paper version contains:

1. `manuscript/ACLD_Manuscript.tex` and `manuscript/adaptive-critical-link-detection.bib`, byte-identical to the author-approved 49-page paper package.
2. A chunked, SHA-256-pinned `ACLD_Paper_Source_v1.0.0.zip` archive with the original PDF, LaTeX, figures, supplemental LaTeX/PDF and bibliography. Paper PDF SHA256: `60a9b2c1de0a84afdf3c0993b111ce9ffc04928e6cfc757a51a82c181585db84`.
3. A chunked, SHA-256-pinned `ACLD_S2_Zenodo_Dataset_v1.0.zip` archive with 40 original raw paired S2 CSVs, 40 SHA sidecars, metadata and retrospective reanalysis scripts.
4. The original October P0.5/P0.6 project Python code and notebook at repository root, retained as historical protocol source. **These files are not independently verified as the exact executed S2 engine for every one of the 40 units.**
5. `RELEASE_MANIFEST.json`, reconstruction code and automated CI checks so the files cannot be silently altered while preserving the same identity.

The two binary archives are stored as ordered chunks because GitHub's file-write interface could not accept the full archives in one transaction. This has **no effect on the bytes** after reconstruction. The source archives and release bundle are reconstructed and verified by CI.

## Reconstruct the exact files

From a fresh Git checkout at the final release tag:

```bash
python research_artifacts/v1.0.0/reconstruct.py --out dist
python research_artifacts/v1.0.0/build_single_release.py --out dist
```

No extra Python package is needed for archive reconstruction. An optional S2 reanalysis needs NumPy and pandas (see the S2 archive's `scripts/requirements.txt`). The combined `dist/ACLD_PaperMatched_v1.0.0-paper.zip` contains both frozen packages and repository Python source.

## Evidence scope / limitations

- Recorded S2 data comprise 40 scenario–seed units (10 distinct seeds shared across four scenario groups), 159,960 *paired* time-step observations and 46 next-step partition-onset events conditioned on the graph being currently connected.
- Retrospective S2 onset results: ACLD TP45/FN1/FP117/TN2933; guarded predictive Tarjan TP46/FN0/FP712/TN2338. The onset subgroup was defined **after** the original S2 runs, and is **not preregistered**.
- The S2 ZIP contains all 40 source CSVs; the complete A1–C3 seven-campaign raw step datasets and some historical executed-worker source fingerprints are **not** included. A complete independent rerun of all paper experiments is therefore **not** certified by this archive.
- ACLD and the matched predictive Tarjan baseline use identical predicted bridge edge sets; the paper's measured contribution is binary warning selection, not a new bridge-localization accuracy gain.
- The separate S5 safety-controller work is exploratory and **not part** of this paper release.
- Neither drone flight, packet-level field RF validation, hard real-time large-swarm performance nor an unconditional safety guarantee is asserted.

## DOI and publication status

**DOI not assigned.** Enable Zenodo's GitHub integration **before** publishing the GitHub Release for `v1.0.0-paper`, then verify the DOI record. GitHub Actions run artifacts have limited retention; a GitHub Release and Zenodo's resulting source archive preserve the paper-specific version. The Zenodo snapshot includes chunked data/source and reconstruction code, even if supplemental GitHub release assets are not imported. The author has not confirmed a formal journal submission.

The repository did not have a verified explicit license at freeze preparation. No reuse license has been invented here; obtain the author's approval before selecting a public software/data reuse license.

## Scientific and administrative declarations

Author-reported: no external funding and no competing interests. The manuscript's AI declaration reads: “Generative AI tools were used to assist with language editing, data analysis, and figure preparation. The author takes full responsibility for the final manuscript.”
