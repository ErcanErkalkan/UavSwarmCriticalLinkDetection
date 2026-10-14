from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import pandas as pd
from acld_p0_6_io import atomic_json
from acld_p0_6_common import (
    P06_PROTOCOL, CAMPAIGN_DIRS, DETECTION_CAMPAIGNS, MITIGATION_CAMPAIGNS,
    CONTROLLERS, load_p04_namespace, build_specs, base_cfg, flatten_units, assigned_units
)

def atomic_status(path:Path, payload:dict):
    atomic_json(path, payload)

def append_rows(path:Path, rows):
    rows=list(rows or [])
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(path,mode="a",header=(not path.exists() or path.stat().st_size==0),index=False)

def read_csv(path:Path):
    if not path.exists() or path.stat().st_size==0: return pd.DataFrame()
    try: return pd.read_csv(path)
    except pd.errors.EmptyDataError: return pd.DataFrame()

def read_existing(project:Path,camp_dir:str,filename:str):
    frames=[]
    for base in (
        project/"acld_full_experiment_outputs_P0_corrected"/camp_dir,
        project/"acld_full_experiment_outputs_P0_5"/camp_dir,
        project/"acld_full_experiment_outputs_P0_6"/camp_dir,
    ):
        df=read_csv(base/filename)
        if not df.empty: frames.append(df)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def detection_complete(df,spec,seed):
    if df.empty or "method" not in df.columns: return False
    s=pd.to_numeric(df["seed"],errors="coerce")
    sub=df[(df["scenario"].astype(str)==spec.name)&(s==int(seed))]
    return set(sub["method"].dropna().astype(str)) >= set(map(str,spec.methods))

def mitigation_complete(df,spec,controller,seed):
    if df.empty or "controller" not in df.columns: return False
    s=pd.to_numeric(df["seed"],errors="coerce")
    sub=df[(df["scenario"].astype(str)==spec.name)&
           (df["controller"].astype(str)==str(controller))&(s==int(seed))]
    return len(sub)>=1

def enrich_detection(cfg,spec,seed,ep_rows,st_rows):
    eps=[]; steps=[]
    for r in ep_rows:
        x=dict(r); x.update({
            "scenario":spec.name,"n_uavs":int(cfg.n_uavs),
            "density_scale_mode":str(cfg.density_scale_mode),
            "jnr_db":cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
            "alpha":float(cfg.alpha),"risk_band_fraction":float(cfg.risk_band_fraction),
            "p06_protocol":P06_PROTOCOL
        }); eps.append(x)
    for r in st_rows:
        if str(r.get("method"))!="acld_v3": continue
        x=dict(r); x.update({
            "scenario":spec.name,"seed":int(seed),"n_uavs":int(cfg.n_uavs),
            "density_scale_mode":str(cfg.density_scale_mode),
            "jnr_db":cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
            "alpha":float(cfg.alpha),"risk_band_fraction":float(cfg.risk_band_fraction),
            "p06_protocol":P06_PROTOCOL
        }); steps.append(x)
    return eps,steps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",required=True)
    ap.add_argument("--campaign",required=True,choices=list(CAMPAIGN_DIRS))
    ap.add_argument("--worker-index",type=int,default=1)
    ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()

    project=Path(args.project_root).resolve()
    nb=project/"acld_full_experimental_campaign_P0_4_corrected.ipynb"
    if not nb.exists(): raise SystemExit(f"Notebook not found: {nb}")
    os.chdir(project)
    ns=load_p04_namespace(nb); cfg0=base_cfg(ns)
    camp=args.campaign; camp_dir=CAMPAIGN_DIRS[camp]

    if camp=="C2":
        if args.workers!=1 or args.worker_index!=1:
            raise SystemExit("C2 must run serially with one worker.")
        out=project/"acld_full_experiment_outputs_P0_6"/camp_dir
        out.mkdir(parents=True,exist_ok=True)
        rows=ns["benchmark_runtime_vs_n"](cfg0,sizes=(50,150,300,500,1000),repeats=1000,
                                           outdir=out,overwrite=False)
        pd.DataFrame(rows).to_csv(out/"runtime_scaling_full.csv",index=False)
        atomic_status(out/"_worker_status.json",{
            "protocol":P06_PROTOCOL,"campaign":"C2","assigned_units":5000,
            "completed_or_reused_units":5000,"gate":"PASS"
        })
        print("C2 serial runtime benchmark complete.",flush=True)
        return

    specs=build_specs(ns,camp)
    all_units=flatten_units(specs,camp)
    mine=assigned_units(all_units,args.worker_index,args.workers)
    workers_root=project/"acld_full_experiment_outputs_P0_6_workers"/camp_dir
    worker_dir=workers_root/f"worker_{args.worker_index:02d}_of_{args.workers:02d}"
    worker_dir.mkdir(parents=True,exist_ok=True)
    status_path=worker_dir/"_worker_status.json"

    manifest={
        "protocol":P06_PROTOCOL,"campaign":camp,"worker_index":args.worker_index,
        "workers":args.workers,"assigned_units":len(mine),"total_campaign_units":len(all_units),
        "project_root":str(project),"notebook_source":str(nb),
        "thread_limits":{k:os.environ.get(k) for k in
                         ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")}
    }
    (worker_dir/"worker_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    print(f"Protocol: {P06_PROTOCOL}",flush=True)
    print(f"Campaign: {camp} | worker {args.worker_index}/{args.workers} | assigned={len(mine)}/{len(all_units)}",flush=True)

    clone=ns["_clone_cfg"]
    if camp in DETECTION_CAMPAIGNS:
        run_episode=ns["run_detection_episode"]
        ep_path=worker_dir/"detection_episode_metrics_long.csv"
        st_path=worker_dir/"detection_step_metrics_acld_only.csv"
        existing=read_existing(project,camp_dir,"detection_episode_metrics_long.csv")
        own=read_csv(ep_path)
        combined=pd.concat([existing,own],ignore_index=True) if not own.empty else existing.copy()
        done=0
        for spec,seed in mine:
            if detection_complete(combined,spec,seed):
                done+=1
                atomic_status(status_path,{"protocol":P06_PROTOCOL,"campaign":camp,"worker_index":args.worker_index,
                    "workers":args.workers,"assigned_units":len(mine),"completed_or_reused_units":done,
                    "last_finished_scenario":spec.name,"last_finished_seed":seed,"reused":True})
                print(f"skip {spec.name} seed={seed}",flush=True)
                continue
            cfg=clone(cfg0,**spec.cfg_overrides)
            t0=time.perf_counter()
            ep_rows,st_rows=run_episode(cfg,spec.methods,spec.horizon_s,seed,store_step_rows=True)
            eps,steps=enrich_detection(cfg,spec,seed,ep_rows,st_rows)
            append_rows(ep_path,eps); append_rows(st_path,steps)
            combined=pd.concat([combined,pd.DataFrame(eps)],ignore_index=True)
            done+=1
            atomic_status(status_path,{"protocol":P06_PROTOCOL,"campaign":camp,"worker_index":args.worker_index,
                "workers":args.workers,"assigned_units":len(mine),"completed_or_reused_units":done,
                "last_finished_scenario":spec.name,"last_finished_seed":seed,
                "elapsed_last_unit_s":time.perf_counter()-t0,"reused":False})
            print(f"saved {spec.name} seed={seed} progress={done}/{len(mine)}",flush=True)
    else:
        run_episode=ns["run_closed_loop_episode"]
        ep_path=worker_dir/"mitigation_episode_metrics_long.csv"
        st_path=worker_dir/"mitigation_step_metrics_acld_only.csv"
        existing=read_existing(project,camp_dir,"mitigation_episode_metrics_long.csv")
        own=read_csv(ep_path)
        combined=pd.concat([existing,own],ignore_index=True) if not own.empty else existing.copy()
        done=0
        for spec,controller,seed in mine:
            if mitigation_complete(combined,spec,controller,seed):
                done+=1
                atomic_status(status_path,{"protocol":P06_PROTOCOL,"campaign":camp,"worker_index":args.worker_index,
                    "workers":args.workers,"assigned_units":len(mine),"completed_or_reused_units":done,
                    "last_finished_scenario":spec.name,"last_finished_controller":controller,
                    "last_finished_seed":seed,"reused":True})
                print(f"skip {spec.name} {controller} seed={seed}",flush=True)
                continue
            cfg=clone(cfg0,**spec.cfg_overrides)
            t0=time.perf_counter()
            st_rows,summ=run_episode(cfg,controller,spec.horizon_s,seed,intervention_mode="hybrid")
            x=dict(summ); x.update({
                "scenario":spec.name,"n_uavs":int(cfg.n_uavs),
                "density_scale_mode":str(cfg.density_scale_mode),
                "jnr_db":cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
                "p06_protocol":P06_PROTOCOL
            })
            append_rows(ep_path,[x])
            if controller=="acld":
                payload=[]
                for r in st_rows:
                    y=dict(r); y.update({
                        "scenario":spec.name,"seed":seed,"n_uavs":int(cfg.n_uavs),
                        "density_scale_mode":str(cfg.density_scale_mode),
                        "jnr_db":cfg.jnr_db if cfg.jnr_db is not None else "NoJam",
                        "p06_protocol":P06_PROTOCOL
                    }); payload.append(y)
                append_rows(st_path,payload)
            combined=pd.concat([combined,pd.DataFrame([x])],ignore_index=True)
            done+=1
            atomic_status(status_path,{"protocol":P06_PROTOCOL,"campaign":camp,"worker_index":args.worker_index,
                "workers":args.workers,"assigned_units":len(mine),"completed_or_reused_units":done,
                "last_finished_scenario":spec.name,"last_finished_controller":controller,
                "last_finished_seed":seed,"elapsed_last_unit_s":time.perf_counter()-t0,"reused":False})
            print(f"saved {spec.name} {controller} seed={seed} progress={done}/{len(mine)}",flush=True)

if __name__=="__main__":
    main()
