from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from acld_p0_6_io import atomic_json

PARALLEL_CAMPAIGNS = ("C3", "B1", "B2", "A2", "C1")
SERIAL_AFTER_PARALLEL = "C2"

CAMPAIGN_DIRS = {
    "A2": "A2_detection_rician",
    "B1": "B1_closed_loop_nominal",
    "B2": "B2_closed_loop_stress",
    "C1": "C1_accuracy_scaling",
    "C2": "C2_runtime_scaling",
    "C3": "C3_gcs_sensitivity",
}

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def campaign_passed(project: Path, campaign: str) -> bool:
    if campaign == "C2":
        return (project / "acld_full_experiment_outputs_P0_6" / CAMPAIGN_DIRS["C2"] /
                "runtime_scaling_full.csv").exists()
    report = project / "acld_full_experiment_outputs_P0_6" / CAMPAIGN_DIRS[campaign] / "_merge_report.json"
    return load_json(report).get("gate") == "PASS"

def launch_workers(project: Path, py: str, campaign: str, workers: int, attempt: int, env: dict, logs: Path):
    worker_script = project / "run_acld_p0_6_unit_worker.py"
    camp_log = logs / campaign / f"attempt_{attempt:02d}"
    camp_log.mkdir(parents=True, exist_ok=True)
    procs = []
    streams = []
    for i in range(1, workers + 1):
        cmd = [
            py, "-u", str(worker_script),
            "--project-root", str(project),
            "--campaign", campaign,
            "--worker-index", str(i),
            "--workers", str(workers),
        ]
        so = open(camp_log / f"worker_{i:02d}_stdout.log", "ab", buffering=0)
        se = open(camp_log / f"worker_{i:02d}_stderr.log", "ab", buffering=0)
        p = subprocess.Popen(cmd, cwd=project, env=env, stdout=so, stderr=se)
        (camp_log / f"worker_{i:02d}.pid").write_text(str(p.pid), encoding="ascii")
        procs.append(p)
        streams.append((so, se))
    return {"campaign": campaign, "attempt": attempt, "workers": workers,
            "procs": procs, "streams": streams, "logdir": camp_log}

def close_streams(state):
    for so, se in state["streams"]:
        try: so.close()
        except Exception: pass
        try: se.close()
        except Exception: pass

def merge_campaign(project: Path, py: str, campaign: str, workers: int, env: dict, logs: Path):
    merger = project / "merge_acld_p0_6_campaign.py"
    so_path = logs / campaign / "merge_stdout.log"
    se_path = logs / campaign / "merge_stderr.log"
    with open(so_path, "ab", buffering=0) as so, open(se_path, "ab", buffering=0) as se:
        rc = subprocess.call(
            [py, "-u", str(merger), "--project-root", str(project),
             "--campaign", campaign, "--workers", str(workers)],
            cwd=project, env=env, stdout=so, stderr=se
        )
    report = load_json(project / "acld_full_experiment_outputs_P0_6" /
                       CAMPAIGN_DIRS[campaign] / "_merge_report.json")
    return rc == 0 and report.get("gate") == "PASS", report

def run_c2(project: Path, py: str, env: dict, logs: Path):
    if campaign_passed(project, "C2"):
        return True
    d = logs / "C2" / "attempt_01"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "worker_01_stdout.log", "ab", buffering=0) as so, \
         open(d / "worker_01_stderr.log", "ab", buffering=0) as se:
        rc = subprocess.call([
            py, "-u", str(project / "run_acld_p0_6_unit_worker.py"),
            "--project-root", str(project), "--campaign", "C2",
            "--worker-index", "1", "--workers", "1"
        ], cwd=project, env=env, stdout=so, stderr=se)
    return rc == 0 and campaign_passed(project, "C2")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--workers-per-campaign", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    if args.workers_per_campaign < 1:
        raise SystemExit("--workers-per-campaign must be >= 1")

    project = Path(args.project_root).resolve()
    py = sys.executable
    required = [
        project / "run_acld_p0_6_unit_worker.py",
        project / "merge_acld_p0_6_campaign.py",
        project / "acld_p0_6_common.py",
        project / "acld_full_experimental_campaign_P0_4_corrected.ipynb",
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    logs = project / "acld_p0_6_logs_PARALLEL"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / "QUEUE_STATUS.json"
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"

    attempts = {c: 0 for c in PARALLEL_CAMPAIGNS}
    done = {c: campaign_passed(project, c) for c in PARALLEL_CAMPAIGNS}
    active = {}
    failed = {}

    def status(state, **extra):
        payload = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol_variant": "P0.6-campaign-parallel-A1-independent",
            "state": state,
            "parallel_campaigns": list(PARALLEL_CAMPAIGNS),
            "workers_per_campaign": args.workers_per_campaign,
            "C2_policy": "serial after C3/B1/B2/A2/C1 complete",
            "A1_policy": "external P0.5 result; not executed here",
            "A3_policy": "existing completed P0.5/A3 result; not rerun",
            "done": done,
            "active": {
                c: {
                    "attempt": st["attempt"],
                    "pids": [p.pid for p in st["procs"]],
                    "alive": sum(p.poll() is None for p in st["procs"]),
                } for c, st in active.items()
            },
            "failed": failed,
            **extra,
        }
        atomic_json(status_path, payload)

    # Launch all independent campaigns together.
    for camp in PARALLEL_CAMPAIGNS:
        if done[camp]:
            continue
        attempts[camp] += 1
        active[camp] = launch_workers(
            project, py, camp, args.workers_per_campaign,
            attempts[camp], env, logs
        )
    status("RUNNING_PARALLEL")

    while active:
        time.sleep(30)
        finished_campaigns = []
        for camp, st in list(active.items()):
            if any(p.poll() is None for p in st["procs"]):
                continue
            codes = [p.returncode for p in st["procs"]]
            close_streams(st)
            ok, report = merge_campaign(project, py, camp, st["workers"], env, logs)
            if ok:
                done[camp] = True
                finished_campaigns.append(camp)
            else:
                if attempts[camp] >= args.max_retries:
                    failed[camp] = {
                        "return_codes": codes,
                        "merge_report": report,
                        "message": "max retries exhausted",
                    }
                    finished_campaigns.append(camp)
                else:
                    attempts[camp] += 1
                    active[camp] = launch_workers(
                        project, py, camp, args.workers_per_campaign,
                        attempts[camp], env, logs
                    )
        for camp in finished_campaigns:
            # Do not delete a retried replacement.
            st = active.get(camp)
            if done.get(camp) or camp in failed:
                active.pop(camp, None)
        status("RUNNING_PARALLEL")

        if failed:
            # Stop any remaining children before exiting.
            for st in active.values():
                for p in st["procs"]:
                    if p.poll() is None:
                        try: p.terminate()
                        except Exception: pass
                close_streams(st)
            active.clear()
            status("ERROR", message="At least one campaign failed.")
            raise SystemExit("Parallel campaign failure: " + ", ".join(sorted(failed)))

    # Official runtime benchmark: deliberately after all parallel work.
    status("RUNNING_C2_SERIAL")
    if not run_c2(project, py, env, logs):
        status("ERROR", campaign="C2", message="C2 serial runtime benchmark failed.")
        raise SystemExit("C2 failed.")

    status(
        "WAITING_FOR_A1_FINALIZATION",
        c2_pass=True,
        message="C3/B1/B2/A2/C1 and C2 completed. Run finalizer after A1 P0.5 is complete."
    )
    print("P0.6 parallel independent campaigns complete; waiting for A1 finalization.", flush=True)

if __name__ == "__main__":
    main()
