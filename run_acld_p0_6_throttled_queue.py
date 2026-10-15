#!/usr/bin/env python3
"""Resume ACLD P0.6 6-way shards with a bounded number of active workers.

The original shard indices (--workers 6) remain unchanged.  This deliberately
avoids repartitioning or rewriting prior A2/C1 CSVs when lowering CPU/RAM load.
The original controller/worker processes MUST be stopped before launching.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from acld_p0_6_io import atomic_json

SHARDS = 6
CAMPAIGNS = {"A2": "A2_detection_rician", "C1": "C1_accuracy_scaling"}
FINISHED = {
    "B1": "B1_closed_loop_nominal",
    "B2": "B2_closed_loop_stress",
    "C3": "C3_gcs_sensitivity",
}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}


def worker_folder(project: Path, camp: str, index: int) -> Path:
    return (project / "acld_full_experiment_outputs_P0_6_workers" /
            CAMPAIGNS[camp] / f"worker_{index:02d}_of_{SHARDS:02d}")


def check_source(project: Path) -> None:
    for name in ("run_acld_p0_6_unit_worker.py", "merge_acld_p0_6_campaign.py",
                 "acld_p0_6_common.py", "acld_full_experimental_campaign_P0_4_corrected.ipynb"):
        if not (project / name).is_file():
            raise RuntimeError(f"Required source missing: {project / name}")
    for name, folder in FINISHED.items():
        report = read_json(project / "acld_full_experiment_outputs_P0_6" / folder / "_merge_report.json")
        if report.get("gate") != "PASS":
            raise RuntimeError(f"{name} canonical gate not PASS; refusing to restart finished campaign")
    for camp in CAMPAIGNS:
        for idx in range(1, SHARDS + 1):
            wd = worker_folder(project, camp, idx)
            if not (wd / "detection_episode_metrics_long.csv").is_file():
                raise RuntimeError(f"Missing existing 6-shard output: {wd}")
            manifest = read_json(wd / "worker_manifest.json")
            if manifest.get("workers") != SHARDS or manifest.get("worker_index") != idx:
                raise RuntimeError(f"Invalid six-way shard manifest: {wd}")


def ready_shards() -> list[int]:
    return list(range(1, SHARDS + 1))


def launch(project: Path, camp: str, idx: int, attempt: int, py: str, env: dict, logs: Path):
    outdir = logs / camp / "throttled" / f"shard_{idx:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    stdout = open(outdir / f"attempt_{attempt:02d}_stdout.log", "ab", buffering=0)
    stderr = open(outdir / f"attempt_{attempt:02d}_stderr.log", "ab", buffering=0)
    cmd = [py, "-u", str(project / "run_acld_p0_6_unit_worker.py"),
           "--project-root", str(project), "--campaign", camp,
           "--worker-index", str(idx), "--workers", str(SHARDS)]
    try:
        proc = subprocess.Popen(cmd, cwd=str(project), env=env, stdout=stdout, stderr=stderr)
    except Exception:
        stdout.close()
        stderr.close()
        raise
    return {"proc": proc, "stdout": stdout, "stderr": stderr, "attempt": attempt}


def finish_streams(info: dict) -> None:
    info["stdout"].close()
    info["stderr"].close()


def shard_complete(project: Path, camp: str, idx: int) -> bool:
    stat = read_json(worker_folder(project, camp, idx) / "_worker_status.json")
    return (stat.get("worker_index") == idx and stat.get("workers") == SHARDS and
            stat.get("campaign") == camp and
            isinstance(stat.get("assigned_units"), int) and
            stat.get("completed_or_reused_units") == stat["assigned_units"])


def merge(project: Path, camp: str, py: str, env: dict, logs: Path) -> None:
    folder = logs / camp / "throttled"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "merge_stdout.log", "ab", buffering=0) as so, \
         open(folder / "merge_stderr.log", "ab", buffering=0) as se:
        result = subprocess.run(
            [py, "-u", str(project / "merge_acld_p0_6_campaign.py"),
             "--project-root", str(project), "--campaign", camp, "--workers", str(SHARDS)],
            cwd=project, env=env, stdout=so, stderr=se, check=False,
        )
    report = read_json(project / "acld_full_experiment_outputs_P0_6" /
                       CAMPAIGNS[camp] / "_merge_report.json")
    if result.returncode != 0 or report.get("gate") != "PASS":
        raise RuntimeError(f"{camp} canonical merge failed: {report.get('gate')}; review merge log")


def run_c2(project: Path, py: str, env: dict, logs: Path) -> None:
    resultfile = project / "acld_full_experiment_outputs_P0_6" / "C2_runtime_scaling" / "runtime_scaling_full.csv"
    if resultfile.is_file() and resultfile.stat().st_size:
        return
    outdir = logs / "C2" / "throttled_serial"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "stdout.log", "ab", buffering=0) as so, \
         open(outdir / "stderr.log", "ab", buffering=0) as se:
        result = subprocess.run(
            [py, "-u", str(project / "run_acld_p0_6_unit_worker.py"),
             "--project-root", str(project), "--campaign", "C2",
             "--worker-index", "1", "--workers", "1"],
            cwd=project, env=env, stdout=so, stderr=se, check=False,
        )
    if result.returncode != 0 or not resultfile.is_file() or not resultfile.stat().st_size:
        raise RuntimeError("C2 serial benchmark did not finish; inspect C2 logs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--active-per-campaign", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=10.0, help=argparse.SUPPRESS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    project = Path(args.project_root).resolve()
    check_source(project)
    if args.check_only:
        print(json.dumps({"check": "PASS", "shards_per_campaign": SHARDS,
                          "active_per_campaign": args.active_per_campaign,
                          "campaigns": list(CAMPAIGNS)}, indent=2))
        return

    logs = project / "acld_p0_6_logs_PARALLEL"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = logs / "QUEUE_STATUS.json"
    lock_path = logs / "THROTTLED_QUEUE.lock"
    # Exclusive lock is an additional safeguard, not a substitute for Windows process checks.
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="ascii") as fp:
        fp.write(str(os.getpid()))
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    py = sys.executable
    pending = {camp: ready_shards() for camp in CAMPAIGNS}
    active = {camp: {} for camp in CAMPAIGNS}
    completed = {camp: [] for camp in CAMPAIGNS}
    tries: dict[tuple[str, int], int] = {}
    done = {"A2": False, "C1": False, "B1": True, "B2": True, "C3": True}
    failed: dict = {}

    def status(state: str, **extra):
        atomic_json(status_path, {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol_variant": "P0.6-throttled-six-shard", "state": state,
            "workers_per_campaign": SHARDS,
            "active_workers_per_campaign": args.active_per_campaign,
            "C2_policy": "serial after A2/C1/B1/B2/C3 complete",
            "done": done,
            "active": {camp: {"pids": [x["proc"].pid for x in jobs.values()],
                              "shards": list(jobs), "alive": sum(x["proc"].poll() is None for x in jobs.values())}
                       for camp, jobs in active.items()},
            "pending": pending, "completed_shards": completed,
            "failed": failed, **extra,
        })

    try:
        status("RUNNING_PARALLEL")
        while any(pending[c] or active[c] for c in CAMPAIGNS):
            for camp in CAMPAIGNS:
                while pending[camp] and len(active[camp]) < args.active_per_campaign:
                    idx = pending[camp].pop(0)
                    key = (camp, idx)
                    tries[key] = tries.get(key, 0) + 1
                    active[camp][idx] = launch(project, camp, idx, tries[key], py, env, logs)
            status("RUNNING_PARALLEL")
            time.sleep(args.poll_seconds)
            for camp in CAMPAIGNS:
                for idx, job in list(active[camp].items()):
                    ret = job["proc"].poll()
                    if ret is None:
                        continue
                    finish_streams(job)
                    del active[camp][idx]
                    if ret == 0 and shard_complete(project, camp, idx):
                        completed[camp].append(idx)
                    elif tries[(camp, idx)] < args.max_retries:
                        pending[camp].append(idx)
                    else:
                        failed[f"{camp}/shard_{idx:02d}"] = {"return_code": ret,
                                                               "message": "worker failed or did not complete all units"}
            if failed:
                raise RuntimeError("Workers failed after configured retries: " + str(failed))

        status("MERGING_CAMPAIGNS")
        for camp in CAMPAIGNS:
            merge(project, camp, py, env, logs)
            done[camp] = True
        status("RUNNING_C2_SERIAL")
        run_c2(project, py, env, logs)
        status("WAITING_FOR_A1_FINALIZATION", c2_pass=True,
               message="A2/C1 passed; C2 completed. Finalize when other gates ready.")
        print("Throttled queue complete, A2/C1 canonical PASS and C2 complete", flush=True)
    except Exception as exc:
        status("ERROR", error=str(exc))
        raise
    finally:
        for jobs in active.values():
            for job in jobs.values():
                if job["proc"].poll() is None:
                    job["proc"].terminate()
                finish_streams(job)
        try:
            lock_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
