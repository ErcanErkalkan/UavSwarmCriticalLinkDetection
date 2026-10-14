from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from acld_p0_5_common import (
    P05_PROTOCOL, CAMPAIGN_DIRS, DETECTION_CAMPAIGNS, MITIGATION_CAMPAIGNS,
    load_p04_namespace, build_specs, base_cfg, worker_specs,
)

CONTROLLERS = ("nh", "random", "tarjan_snapshot", "tarjan_predict", "acld")


def atomic_status(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def append_rows(path: Path, rows):
    rows = list(rows or [])
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=(not path.exists() or path.stat().st_size == 0), index=False)


def load_df(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def complete_detection_seed(df: pd.DataFrame, spec, seed: int) -> bool:
    if df.empty:
        return False
    sub = df[(df.get("scenario") == spec.name) & (pd.to_numeric(df.get("seed"), errors="coerce") == int(seed))]
    if sub.empty or "method" not in sub.columns:
        return False
    return set(str(x) for x in sub["method"].dropna()) >= set(str(x) for x in spec.methods)


def complete_mitigation_key(df: pd.DataFrame, spec, controller: str, seed: int) -> bool:
    if df.empty:
        return False
    seed_col = pd.to_numeric(df.get("seed"), errors="coerce")
    sub = df[(df.get("scenario") == spec.name) & (df.get("controller").astype(str) == str(controller)) & (seed_col == int(seed))]
    return len(sub) >= 1


def enrich_detection(cfg, spec, seed, ep_rows, st_rows):
    eps, steps = [], []
    for r in ep_rows:
        rr = dict(r)
        rr.update({
            "scenario": spec.name,
            "n_uavs": int(cfg.n_uavs),
            "density_scale_mode": str(cfg.density_scale_mode),
            "jnr_db": cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
            "alpha": float(cfg.alpha),
            "risk_band_fraction": float(cfg.risk_band_fraction),
            "p05_protocol": P05_PROTOCOL,
        })
        eps.append(rr)
    # P0.5 storage optimization: D2/D3/D4 use ACLD step rows only.
    for r in st_rows:
        if str(r.get("method")) != "acld_v3":
            continue
        rr = dict(r)
        rr.update({
            "scenario": spec.name,
            "seed": int(seed),
            "n_uavs": int(cfg.n_uavs),
            "density_scale_mode": str(cfg.density_scale_mode),
            "jnr_db": cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
            "alpha": float(cfg.alpha),
            "risk_band_fraction": float(cfg.risk_band_fraction),
            "p05_protocol": P05_PROTOCOL,
        })
        steps.append(rr)
    return eps, steps


def run_detection(ns, cfg0, specs, worker_dir: Path, canonical_dir: Path, args):
    run_episode = ns["run_detection_episode"]
    clone_cfg = ns["_clone_cfg"]
    aggregate = ns["_aggregate_detection_episode_df"]

    ep_path = worker_dir / "detection_episode_metrics_long.csv"
    st_path = worker_dir / "detection_step_metrics_acld_only.csv"
    sum_path = worker_dir / "detection_summary_long.csv"
    status_path = worker_dir / "_worker_status.json"

    worker_existing = load_df(ep_path)
    canonical_existing = load_df(canonical_dir / "detection_episode_metrics_long.csv")
    combined = pd.concat([canonical_existing, worker_existing], ignore_index=True) if not canonical_existing.empty else worker_existing.copy()

    total = sum(len(s.seeds) for s in specs)
    completed = 0
    for spec in specs:
        cfg = clone_cfg(cfg0, **spec.cfg_overrides)
        print(f"[P0.5][{args.campaign}][worker {args.worker_index}/{args.workers}] scenario={spec.name}", flush=True)
        for seed in spec.seeds:
            seed = int(seed)
            if complete_detection_seed(combined, spec, seed):
                completed += 1
                print(f"  skip seed={seed} (already canonical/worker-complete)", flush=True)
                continue
            if args.dry_run:
                print(f"  DRY-RUN would execute seed={seed}", flush=True)
                continue

            t0 = time.perf_counter()
            ep_rows, st_rows = run_episode(cfg, spec.methods, spec.horizon_s, seed, store_step_rows=True)
            eps, steps = enrich_detection(cfg, spec, seed, ep_rows, st_rows)
            append_rows(ep_path, eps)
            append_rows(st_path, steps)
            combined = pd.concat([combined, pd.DataFrame(eps)], ignore_index=True)
            completed += 1
            atomic_status(status_path, {
                "protocol": P05_PROTOCOL,
                "campaign": args.campaign,
                "worker_index": args.worker_index,
                "workers": args.workers,
                "assigned_seed_units": total,
                "completed_or_reused_seed_units": completed,
                "last_finished_scenario": spec.name,
                "last_finished_seed": seed,
                "elapsed_last_seed_s": time.perf_counter() - t0,
                "step_storage_policy": "acld_v3_only",
                "canonical_reuse_dir": str(canonical_dir),
            })
            print(f"  saved seed={seed}; worker-progress={completed}/{total}", flush=True)

    if not args.dry_run:
        ep = load_df(ep_path)
        if not ep.empty:
            aggregate(ep).to_csv(sum_path, index=False)


def run_mitigation(ns, cfg0, specs, worker_dir: Path, canonical_dir: Path, args):
    run_episode = ns["run_closed_loop_episode"]
    clone_cfg = ns["_clone_cfg"]
    aggregate = ns["_aggregate_closed_loop_episode_df"]

    ep_path = worker_dir / "mitigation_episode_metrics_long.csv"
    st_path = worker_dir / "mitigation_step_metrics_acld_only.csv"
    sum_path = worker_dir / "mitigation_summary_long.csv"
    status_path = worker_dir / "_worker_status.json"

    worker_existing = load_df(ep_path)
    canonical_existing = load_df(canonical_dir / "mitigation_episode_metrics_long.csv")
    combined = pd.concat([canonical_existing, worker_existing], ignore_index=True) if not canonical_existing.empty else worker_existing.copy()

    total = sum(len(s.seeds) * len(CONTROLLERS) for s in specs)
    completed = 0
    for spec in specs:
        cfg = clone_cfg(cfg0, **spec.cfg_overrides)
        print(f"[P0.5][{args.campaign}][worker {args.worker_index}/{args.workers}] scenario={spec.name}", flush=True)
        for controller in CONTROLLERS:
            for seed in spec.seeds:
                seed = int(seed)
                if complete_mitigation_key(combined, spec, controller, seed):
                    completed += 1
                    print(f"  skip controller={controller} seed={seed} (already canonical/worker-complete)", flush=True)
                    continue
                if args.dry_run:
                    print(f"  DRY-RUN would execute controller={controller} seed={seed}", flush=True)
                    continue

                t0 = time.perf_counter()
                st_rows, summ = run_episode(cfg, controller, spec.horizon_s, seed, intervention_mode="hybrid")
                rr = dict(summ)
                rr.update({
                    "scenario": spec.name,
                    "n_uavs": int(cfg.n_uavs),
                    "density_scale_mode": str(cfg.density_scale_mode),
                    "jnr_db": cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
                    "p05_protocol": P05_PROTOCOL,
                })
                append_rows(ep_path, [rr])
                # Storage optimization: retain per-step rows only for ACLD controller.
                if str(controller) == "acld":
                    payload = []
                    for r in st_rows:
                        x = dict(r)
                        x.update({
                            "scenario": spec.name,
                            "seed": seed,
                            "n_uavs": int(cfg.n_uavs),
                            "density_scale_mode": str(cfg.density_scale_mode),
                            "jnr_db": cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
                            "p05_protocol": P05_PROTOCOL,
                        })
                        payload.append(x)
                    append_rows(st_path, payload)
                combined = pd.concat([combined, pd.DataFrame([rr])], ignore_index=True)
                completed += 1
                atomic_status(status_path, {
                    "protocol": P05_PROTOCOL,
                    "campaign": args.campaign,
                    "worker_index": args.worker_index,
                    "workers": args.workers,
                    "assigned_controller_seed_units": total,
                    "completed_or_reused_controller_seed_units": completed,
                    "last_finished_scenario": spec.name,
                    "last_finished_controller": controller,
                    "last_finished_seed": seed,
                    "elapsed_last_unit_s": time.perf_counter() - t0,
                    "step_storage_policy": "acld_controller_only",
                    "canonical_reuse_dir": str(canonical_dir),
                })
                print(f"  saved controller={controller} seed={seed}; worker-progress={completed}/{total}", flush=True)

    if not args.dry_run:
        ep = load_df(ep_path)
        if not ep.empty:
            aggregate(ep).to_csv(sum_path, index=False)


def run_c2(ns, cfg0, outdir: Path, args):
    if args.workers != 1 or args.worker_index != 1:
        raise SystemExit("C2 runtime scaling MUST run with --workers 1 --worker-index 1 to avoid benchmark contention.")
    if args.dry_run:
        print("DRY-RUN: C2 serial benchmark sizes=(50,150,300,500,1000), repeats=1000")
        return
    rows = ns["benchmark_runtime_vs_n"](
        cfg0, sizes=(50, 150, 300, 500, 1000), repeats=1000,
        outdir=outdir, overwrite=False)
    pd.DataFrame(rows).to_csv(outdir / "runtime_scaling_full.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--campaign", required=True, choices=list(CAMPAIGN_DIRS))
    ap.add_argument("--worker-index", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    project = Path(args.project_root).resolve()
    notebook = project / "acld_full_experimental_campaign_P0_4_corrected.ipynb"
    if not notebook.exists():
        raise SystemExit(f"Notebook not found: {notebook}")
    if args.worker_index < 1 or args.worker_index > args.workers:
        raise SystemExit("worker-index must be within 1..workers")

    os.chdir(project)
    ns = load_p04_namespace(notebook)
    cfg0 = base_cfg(ns)

    camp_dir = CAMPAIGN_DIRS[args.campaign]
    canonical_dir = project / "acld_full_experiment_outputs_P0_corrected" / camp_dir
    workers_root = project / "acld_full_experiment_outputs_P0_5_workers" / camp_dir
    worker_dir = workers_root / f"worker_{args.worker_index:02d}_of_{args.workers:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": P05_PROTOCOL,
        "campaign": args.campaign,
        "worker_index": args.worker_index,
        "workers": args.workers,
        "project_root": str(project),
        "notebook_source": str(notebook),
        "canonical_reuse_dir": str(canonical_dir),
        "worker_output_dir": str(worker_dir),
        "thread_limits": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")},
    }
    (worker_dir / "worker_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.campaign == "C2":
        outdir = project / "acld_full_experiment_outputs_P0_5" / camp_dir
        outdir.mkdir(parents=True, exist_ok=True)
        run_c2(ns, cfg0, outdir, args)
        return

    specs = build_specs(ns, args.campaign)
    assigned = worker_specs(specs, args.worker_index, args.workers)
    print(f"Protocol: {P05_PROTOCOL}")
    print(f"Campaign: {args.campaign}")
    print(f"Worker: {args.worker_index}/{args.workers}")
    print(f"Assigned scenarios: {len(assigned)}/{len(specs)}")
    for s in assigned:
        print(f"  - {s.name}")

    if args.campaign in DETECTION_CAMPAIGNS:
        run_detection(ns, cfg0, assigned, worker_dir, canonical_dir, args)
    elif args.campaign in MITIGATION_CAMPAIGNS:
        run_mitigation(ns, cfg0, assigned, worker_dir, canonical_dir, args)


if __name__ == "__main__":
    main()
