from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run(cmd, cwd, log_out, log_err):
    with open(log_out, "ab", buffering=0) as so, open(log_err, "ab", buffering=0) as se:
        return subprocess.call(cmd, cwd=cwd, stdout=so, stderr=se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--a1-workers", type=int, default=6)
    ap.add_argument("--a3-workers", type=int, default=8)
    a = ap.parse_args()

    project = Path(a.project_root).resolve()
    py = sys.executable
    logs = project / "acld_p0_6_logs_INDEPENDENT"
    logs.mkdir(parents=True, exist_ok=True)
    outlog = logs / "finalize_stdout.log"
    errlog = logs / "finalize_stderr.log"
    status = logs / "FINALIZATION_STATUS.json"

    required = [
        project / "merge_acld_p0_5.py",
        project / "merge_acld_p0_6_campaign.py",
        project / "postprocess_acld_p0_6_FINAL.py",
        project / "acld_full_experimental_campaign_P0_4_corrected.ipynb",
    ]
    for f in required:
        if not f.exists():
            raise SystemExit(f"Required file missing: {f}")

    def write_state(state, **kw):
        payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "state": state, **kw}
        status.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 1) Merge completed A1 P0.5 worker shards. If A1 is not complete, this gate remains INCOMPLETE.
    write_state("MERGING_A1_P05")
    rc = run([py, "-u", str(project / "merge_acld_p0_5.py"),
              "--project-root", str(project), "--campaign", "A1", "--workers", str(a.a1_workers)],
             project, outlog, errlog)
    a1p05 = load_json(project / "acld_full_experiment_outputs_P0_5" / "A1_detection_core" / "_merge_report.json")
    if rc != 0 or a1p05.get("gate") != "PASS":
        write_state("WAITING_FOR_A1", a1_p05_report=a1p05,
                    message="A1 is not complete yet. Finalization stopped safely; rerun later.")
        raise SystemExit("A1 P0.5 merge gate is not PASS yet. Rerun finalizer after A1 completes.")

    # 2) Promote A1 into canonical P0.6 output by merge/dedup, without rerunning A1.
    write_state("PROMOTING_A1_TO_P06")
    rc = run([py, "-u", str(project / "merge_acld_p0_6_campaign.py"),
              "--project-root", str(project), "--campaign", "A1", "--workers", "8"],
             project, outlog, errlog)
    a1p06 = load_json(project / "acld_full_experiment_outputs_P0_6" / "A1_detection_core" / "_merge_report.json")
    if rc != 0 or a1p06.get("gate") != "PASS":
        write_state("ERROR", campaign="A1", report=a1p06)
        raise SystemExit("A1 P0.6 canonical merge did not PASS.")

    # 3) Merge existing completed A3 seed8 corrected output; do not rerun A3.
    write_state("MERGING_EXISTING_A3")
    rc = run([py, "-u", str(project / "merge_acld_p0_6_campaign.py"),
              "--project-root", str(project), "--campaign", "A3", "--workers", str(a.a3_workers)],
             project, outlog, errlog)
    a3p06 = load_json(project / "acld_full_experiment_outputs_P0_6" / "A3_ablation" / "_merge_report.json")
    if rc != 0 or a3p06.get("gate") != "PASS":
        write_state("ERROR", campaign="A3", report=a3p06)
        raise SystemExit("Existing A3 corrected output did not merge to PASS.")

    # 4) Verify all independent campaigns are already complete.
    checks = {
        "A2": project / "acld_full_experiment_outputs_P0_6" / "A2_detection_rician" / "_merge_report.json",
        "B1": project / "acld_full_experiment_outputs_P0_6" / "B1_closed_loop_nominal" / "_merge_report.json",
        "B2": project / "acld_full_experiment_outputs_P0_6" / "B2_closed_loop_stress" / "_merge_report.json",
        "C1": project / "acld_full_experiment_outputs_P0_6" / "C1_accuracy_scaling" / "_merge_report.json",
        "C3": project / "acld_full_experiment_outputs_P0_6" / "C3_gcs_sensitivity" / "_merge_report.json",
    }
    bad = {k: load_json(v) for k, v in checks.items() if load_json(v).get("gate") != "PASS"}
    c2 = project / "acld_full_experiment_outputs_P0_6" / "C2_runtime_scaling" / "runtime_scaling_full.csv"
    if bad or not c2.exists() or c2.stat().st_size == 0:
        write_state("ERROR", message="Independent campaign outputs are incomplete.", bad_reports=bad,
                    c2_exists=c2.exists())
        raise SystemExit("Independent P0.6 campaigns are not all complete.")

    # 5) Run final combined D1-D4/E post-processing only now.
    write_state("POSTPROCESS_FINAL")
    rc = run([py, "-u", str(project / "postprocess_acld_p0_6_FINAL.py"),
              "--project-root", str(project)], project, outlog, errlog)
    report_path = project / "acld_full_experiment_outputs_P0_6" / "E_manuscript_ready_FINAL" / "_postprocess_report.json"
    final_report = load_json(report_path)
    if rc != 0 or final_report.get("gate") != "PASS":
        write_state("ERROR", campaign="POSTPROCESS_FINAL", report=final_report)
        raise SystemExit("Final post-processing did not PASS.")

    write_state("ALL_PASS", report=str(report_path), message="A1+A2+A3+B1+B2+C1+C2+C3+C4 final evidence chain complete.")
    print(json.dumps({"gate": "PASS", "final_report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
