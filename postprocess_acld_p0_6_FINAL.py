from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd
from acld_p0_6_common import load_p04_namespace, CAMPAIGN_DIRS

DETECTION = ("A1", "A2", "A3", "C1")
MITIGATION = ("B1", "B2")
REQUIRED_MERGE_GATES = ("A1", "A2", "A3", "B1", "B2", "C1", "C3")


def read_required(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"Required file missing/empty: {path}")
    return pd.read_csv(path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def protocol_values(df: pd.DataFrame):
    vals = set()
    for col in ("p06_protocol", "p05_protocol"):
        if col in df.columns:
            vals.update(str(x) for x in df[col].dropna().unique() if str(x) not in ("", "None", "nan"))
    return sorted(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    a = ap.parse_args()
    project = Path(a.project_root).resolve()
    outroot = project / "acld_full_experiment_outputs_P0_6"
    ns = load_p04_namespace(project / "acld_full_experimental_campaign_P0_4_corrected.ipynb")

    gate_reports = {}
    for camp in REQUIRED_MERGE_GATES:
        rp = outroot / CAMPAIGN_DIRS[camp] / "_merge_report.json"
        report = read_json(rp)
        gate_reports[camp] = report
        if report.get("gate") != "PASS":
            raise SystemExit(f"Finalization blocked: {camp} merge gate is not PASS: {rp}")

    c2_path = outroot / "C2_runtime_scaling" / "runtime_scaling_full.csv"
    c2 = read_required(c2_path)

    det = {}
    for camp in DETECTION:
        d = outroot / CAMPAIGN_DIRS[camp]
        det[camp] = {
            "episode": read_required(d / "detection_episode_metrics_long.csv"),
            "step": read_required(d / "detection_step_metrics_acld_only.csv"),
            "summary": read_required(d / "detection_summary_long.csv"),
        }

    mit = {}
    for camp in MITIGATION:
        d = outroot / CAMPAIGN_DIRS[camp]
        mit[camp] = {
            "episode": read_required(d / "mitigation_episode_metrics_long.csv"),
            "summary": read_required(d / "mitigation_summary_long.csv"),
        }

    c3dir = outroot / CAMPAIGN_DIRS["C3"]
    c3_episode = read_required(c3dir / "detection_episode_metrics_long.csv")
    c3_summary = read_required(c3dir / "detection_summary_long.csv")

    BASE_CFG = ns["ACLDConfig"](
        n_uavs=150, include_ground_station=True, density_scale_mode="medium",
        jnr_db=None, alpha=0.0, beta=0.10, gamma=0.20, delta=0.10,
        zeta=0.05, risk_band_fraction=0.97, seed=42,
    )

    # C4 payload-only communication lower-bound estimate.
    rows = []
    for n in (50, 150, 300, 500, 1000):
        payload_per_uav = 14 * 4
        payload_per_cycle = n * payload_per_uav
        rows.append({
            "n_uavs": n,
            "state_scalars_per_uav": 14,
            "scalar_bytes": 4,
            "payload_bytes_per_uav_per_cycle": payload_per_uav,
            "payload_bytes_per_cycle": payload_per_cycle,
            "payload_kbit_per_second": payload_per_cycle * 8.0 * (1.0 / float(BASE_CFG.dt_ctrl)) / 1000.0,
            "scope": "payload-only lower-bound estimate; excludes protocol overhead",
        })
    c4_path = outroot / "C4_payload_only_communication_overhead.csv"
    pd.DataFrame(rows).to_csv(c4_path, index=False)

    d1 = outroot / "D1_significance"; d1.mkdir(parents=True, exist_ok=True)
    det_ep_all = pd.concat([det["A1"]["episode"], det["A2"]["episode"], det["C1"]["episode"]], ignore_index=True)
    sig = ns["run_scenario_paired_significance_suite"](
        det_ep_all, ref_method="acld_v3", metrics=("pew_f1", "edge_f1_mean", "fpr"),
        outpath=d1 / "scenario_balanced_paired_significance.csv",
    )

    d2 = outroot / "D2_prevalence_control"; d2.mkdir(parents=True, exist_ok=True)
    step_prev = pd.concat([det["A1"]["step"], det["A2"]["step"]], ignore_index=True)
    prev = ns["prevalence_control_analysis"](
        step_prev, method="acld_v3", group_col="scenario", n_repeats=200, seed=42,
    )
    prev.to_csv(d2 / "balanced_prevalence_metrics.csv", index=False)

    d3 = outroot / "D3_calibration"; d3.mkdir(parents=True, exist_ok=True)
    cal_step = pd.concat([det["A1"]["step"], det["A2"]["step"], det["C1"]["step"]], ignore_index=True)
    cal_bins, cal_scen = ns["calibration_analysis"](cal_step, method="acld_v3", outdir=d3)
    ns["plot_reliability_diagram"](cal_bins, d3 / "reliability_diagram.png")

    d4 = outroot / "D4_failure_modes"; d4.mkdir(parents=True, exist_ok=True)
    fail = ns["failure_mode_analysis"](
        cal_step, method="acld_v3", high_cand_quantile=0.95,
        outpath=d4 / "failure_mode_summary.csv",
    )

    e = outroot / "E_manuscript_ready_FINAL"
    e.mkdir(parents=True, exist_ok=True)
    det_sum = pd.concat([det[c]["summary"] for c in DETECTION], ignore_index=True)
    mit_sum = pd.concat([mit[c]["summary"] for c in MITIGATION], ignore_index=True)
    ns["build_manuscript_ready_tables"](det_sum, mit_sum, sig, outdir=e)

    det_sum.to_csv(e / "all_detection_summaries_A1_A2_A3_C1.csv", index=False)
    mit_sum.to_csv(e / "all_mitigation_summaries_B1_B2.csv", index=False)
    sig.to_csv(e / "all_significance_A1_A2_C1.csv", index=False)
    prev.to_csv(e / "all_prevalence_control_A1_A2.csv", index=False)
    cal_scen.to_csv(e / "all_calibration_by_scenario_A1_A2_C1.csv", index=False)
    fail.to_csv(e / "all_failure_modes_A1_A2_C1.csv", index=False)
    c3_episode.to_csv(e / "C3_gcs_sensitivity_episode.csv", index=False)
    c3_summary.to_csv(e / "C3_gcs_sensitivity_summary.csv", index=False)
    c2.to_csv(e / "C2_runtime_scaling_full.csv", index=False)
    shutil.copy2(c4_path, e / c4_path.name)

    provenance = {
        camp: {
            "episode_protocols": protocol_values(det[camp]["episode"]) if camp in det else [],
            "merge_gate": gate_reports.get(camp, {}).get("gate"),
        }
        for camp in DETECTION
    }
    provenance.update({
        camp: {"merge_gate": gate_reports.get(camp, {}).get("gate")}
        for camp in MITIGATION
    })
    provenance["C3"] = {
        "episode_protocols": protocol_values(c3_episode),
        "merge_gate": gate_reports["C3"].get("gate"),
    }
    provenance["C2"] = {"rows": int(len(c2)), "file": str(c2_path)}

    report = {
        "gate": "PASS",
        "protocol_variant": "P0.6-final-after-A1",
        "campaigns": ["A1", "A2", "A3", "B1", "B2", "C1", "C2", "C3", "C4"],
        "d1_scope": "A1+A2+C1",
        "d2_scope": "A1+A2",
        "d3_d4_scope": "A1+A2+C1",
        "A3_policy": "existing completed corrected P0.5 seed8 output merged; not rerun",
        "A1_policy": "completed P0.5 worker output merged after completion",
        "provenance": provenance,
        "message": "All required campaign gates passed; D1-D4 and final manuscript-ready evidence generated.",
    }
    (e / "_postprocess_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
