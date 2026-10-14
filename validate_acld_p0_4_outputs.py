from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROTOCOL = "P0.4-exact-bridge-microF1-runtime-scope"
CORE_CAMPAIGNS = {
    "A1_detection_core": "detection_episode_metrics_long.csv",
    "A2_detection_rician": "detection_episode_metrics_long.csv",
    "A3_ablation": "detection_episode_metrics_long.csv",
    "B1_closed_loop_nominal": "mitigation_episode_metrics_long.csv",
    "B2_closed_loop_stress": "mitigation_episode_metrics_long.csv",
    "C1_accuracy_scaling": "detection_episode_metrics_long.csv",
    "C2_runtime_scaling": "runtime_scaling_full.csv",
    "C3_gcs_sensitivity": "detection_episode_metrics_long.csv",
}

DETECTION_REQUIRED = {
    "scenario", "method", "seed", "pew_f1", "fpr",
    "edge_tp_sum", "edge_fp_sum", "edge_fn_sum", "edge_f1_mean",
    "edge_precision_micro", "edge_recall_micro", "edge_support_steps",
    "runtime_scope", "runtime_comparable_across_methods",
    "runtime_incremental_median_ms", "acld_pipeline_median_ms",
}


def _status_counts(status: dict):
    for a, b in [
        ("completed_seed_units", "total_seed_units"),
        ("completed_controller_seed_units", "total_controller_seed_units"),
        ("completed_repeat_units", "total_repeat_units"),
    ]:
        if a in status and b in status:
            return int(status[a]), int(status[b])
    return None


def _resume_status(campaign_dir: Path):
    files = sorted(campaign_dir.glob("_resume_status*.json"))
    if not files:
        return None, None
    p = files[0]
    return json.loads(p.read_text(encoding="utf-8")), p


def _expected_micro_f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.nan if den == 0 else (2 * tp) / den


def _validate_detection(path: Path, errors: list[str], warnings: list[str]):
    df = pd.read_csv(path)
    missing = DETECTION_REQUIRED - set(df.columns)
    if missing:
        errors.append(f"{path}: missing corrected columns: {sorted(missing)}")
        return

    # No fabricated F1=0 on no-support episodes.
    zero_support = (df["edge_tp_sum"].fillna(0) + df["edge_fp_sum"].fillna(0) + df["edge_fn_sum"].fillna(0)) == 0
    bad_zero = zero_support & df["edge_f1_mean"].notna()
    if bad_zero.any():
        errors.append(f"{path}: {int(bad_zero.sum())} no-support episodes have non-NaN Edge-F1.")

    # Recompute exact episode micro-F1 from TP/FP/FN.
    sample = df[["edge_tp_sum", "edge_fp_sum", "edge_fn_sum", "edge_f1_mean"]].copy()
    expected = [
        _expected_micro_f1(float(tp), float(fp), float(fn))
        for tp, fp, fn in zip(sample.edge_tp_sum, sample.edge_fp_sum, sample.edge_fn_sum)
    ]
    observed = sample.edge_f1_mean.to_numpy(dtype=float)
    expected = np.asarray(expected, dtype=float)
    mask = ~(np.isnan(observed) & np.isnan(expected))
    mismatch = mask & ~np.isclose(observed, expected, atol=1e-12, rtol=1e-10, equal_nan=True)
    if mismatch.any():
        errors.append(f"{path}: {int(mismatch.sum())} Edge-F1 values do not reconcile with exact TP/FP/FN micro-F1.")

    # Runtime scope guard.
    comparable = df["runtime_comparable_across_methods"].astype(str).str.lower().isin(["true", "1"])
    if comparable.any():
        warnings.append(f"{path}: {int(comparable.sum())} rows mark runtime as cross-method comparable; review timing boundaries manually.")

    non_acld_pipeline = (~df["method"].astype(str).str.contains("acld", case=False, na=False)) & df["acld_pipeline_median_ms"].notna()
    if non_acld_pipeline.any():
        errors.append(f"{path}: non-ACLD rows contain ACLD full-pipeline runtime values.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()

    project = Path(args.project_root).resolve()
    root = project / "acld_full_experiment_outputs_P0_corrected"
    errors: list[str] = []
    warnings: list[str] = []

    print(f"Project root: {project}")
    print(f"Output root : {root}")
    if not root.exists():
        print("FAIL: corrected output root is missing.")
        return 2

    manifest_path = root / "environment_manifest.json"
    if not manifest_path.exists():
        errors.append("environment_manifest.json is missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("p0_protocol_version") != PROTOCOL:
            errors.append(
                f"Unexpected protocol: {manifest.get('p0_protocol_version')!r}; expected {PROTOCOL!r}"
            )
        if not manifest.get("python_version"):
            errors.append("environment manifest lacks python_version")
        if not manifest.get("source_revision_env") and not manifest.get("git_head"):
            warnings.append("No Git/source revision was captured in the environment manifest.")

    all_complete = True
    print("\nCampaign completion")
    for name, expected_file in CORE_CAMPAIGNS.items():
        cdir = root / name
        if not cdir.exists():
            print(f"  {name:30s} MISSING")
            all_complete = False
            continue
        status, spath = _resume_status(cdir)
        counts = _status_counts(status) if status else None
        if counts:
            done, total = counts
            complete = done == total
            all_complete &= complete
            print(f"  {name:30s} {done:5d}/{total:<5d} {'PASS' if complete else 'INCOMPLETE'}")
        else:
            print(f"  {name:30s} status unavailable")
            all_complete = False

        f = cdir / expected_file
        if not f.exists():
            errors.append(f"{name}: expected output file missing: {expected_file}")
        elif expected_file.startswith("detection_episode"):
            _validate_detection(f, errors, warnings)

    if not all_complete and not args.allow_incomplete:
        errors.append("Core corrected campaigns are not all complete.")

    # Final downstream gate files.
    d1 = root / "D1_significance" / "scenario_balanced_paired_significance.csv"
    d2 = root / "D2_prevalence_control" / "balanced_prevalence_metrics.csv"
    d3 = root / "D3_calibration" / "calibration_by_scenario.csv"
    d4 = root / "D4_failure_modes" / "failure_mode_summary.csv"
    e = root / "E_manuscript_ready"
    final_files = [d1, d2, d3, d4]
    if all_complete and not args.allow_incomplete:
        for f in final_files:
            if not f.exists():
                errors.append(f"Final analysis output missing: {f.relative_to(project)}")
        if not e.exists() or not any(e.glob("*.csv")):
            errors.append("E_manuscript_ready is missing or contains no CSV outputs.")

    print("\nWarnings")
    if warnings:
        for w in warnings:
            print("  WARN:", w)
    else:
        print("  none")

    print("\nValidation")
    if errors:
        for e_ in errors:
            print("  FAIL:", e_)
        print("\nPUBLICATION GATE: FAIL")
        return 2

    if not all_complete:
        print("  Corrected schema/method checks passed on available outputs, but campaigns are incomplete.")
        print("\nPUBLICATION GATE: NOT READY")
        return 0 if args.allow_incomplete else 2

    print("  Corrected campaign and metric checks passed.")
    print("\nPUBLICATION GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
