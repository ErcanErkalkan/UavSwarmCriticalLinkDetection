from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from acld_p0_5_common import (
    P05_PROTOCOL, CAMPAIGN_DIRS, DETECTION_CAMPAIGNS, MITIGATION_CAMPAIGNS,
    load_p04_namespace, build_specs,
)

CONTROLLERS = ("nh", "random", "tarjan_snapshot", "tarjan_predict", "acld")


def read_many(paths):
    dfs=[]
    for p in paths:
        p=Path(p)
        if p.exists() and p.stat().st_size:
            try:
                dfs.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
    return pd.concat(dfs,ignore_index=True) if dfs else pd.DataFrame()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def merge_detection(project: Path, campaign: str, ns, workers: int):
    camp_dir=CAMPAIGN_DIRS[campaign]
    old=project/"acld_full_experiment_outputs_P0_corrected"/camp_dir
    wr=project/"acld_full_experiment_outputs_P0_5_workers"/camp_dir
    out=project/"acld_full_experiment_outputs_P0_5"/camp_dir
    out.mkdir(parents=True,exist_ok=True)

    ep_paths=[old/"detection_episode_metrics_long.csv"]
    st_paths=[old/"detection_step_metrics_long.csv"]
    for i in range(1,workers+1):
        d=wr/f"worker_{i:02d}_of_{workers:02d}"
        ep_paths.append(d/"detection_episode_metrics_long.csv")
        st_paths.append(d/"detection_step_metrics_acld_only.csv")
    ep=read_many(ep_paths)
    st=read_many(st_paths)
    if ep.empty:
        raise SystemExit("No detection episode rows found to merge.")
    if "p05_protocol" not in ep.columns:
        ep["p05_protocol"] = None
    # Prefer P0.5 rows if a duplicate unexpectedly exists.
    ep["_priority"]=(ep["p05_protocol"].astype(str)==P05_PROTOCOL).astype(int)
    ep=ep.sort_values("_priority").drop_duplicates(["scenario","seed","method"],keep="last").drop(columns="_priority")

    if not st.empty:
        # D2/D3/D4 require ACLD rows only; filter old P0.4 all-method step file here.
        if "method" in st.columns:
            st=st[st["method"].astype(str)=="acld_v3"].copy()
        if "p05_protocol" not in st.columns:
            st["p05_protocol"] = None
        st["_priority"]=(st["p05_protocol"].astype(str)==P05_PROTOCOL).astype(int)
        step_keys=[c for c in ["scenario","seed","method","step_idx"] if c in st.columns]
        st=st.sort_values("_priority").drop_duplicates(step_keys,keep="last").drop(columns="_priority")

    specs=build_specs(ns,campaign)
    missing=[]
    for spec in specs:
        for seed in spec.seeds:
            sub=ep[(ep["scenario"]==spec.name)&(pd.to_numeric(ep["seed"],errors="coerce")==int(seed))]
            got=set(sub["method"].astype(str)) if not sub.empty else set()
            req=set(str(m) for m in spec.methods)
            if not req.issubset(got):
                missing.append({"scenario":spec.name,"seed":int(seed),"missing_methods":sorted(req-got)})

    ep.to_csv(out/"detection_episode_metrics_long.csv",index=False)
    if not st.empty:
        st.to_csv(out/"detection_step_metrics_acld_only.csv",index=False)
    agg=ns["_aggregate_detection_episode_df"](ep)
    agg.to_csv(out/"detection_summary_long.csv",index=False)
    report={
        "protocol":P05_PROTOCOL,"campaign":campaign,"workers":workers,
        "episode_rows":int(len(ep)),"step_rows_acld_only":int(len(st)),
        "missing_seed_units":len(missing),"missing_examples":missing[:25],
        "gate":"PASS" if not missing else "INCOMPLETE"
    }
    (out/"_merge_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


def merge_mitigation(project: Path, campaign: str, ns, workers: int):
    camp_dir=CAMPAIGN_DIRS[campaign]
    old=project/"acld_full_experiment_outputs_P0_corrected"/camp_dir
    wr=project/"acld_full_experiment_outputs_P0_5_workers"/camp_dir
    out=project/"acld_full_experiment_outputs_P0_5"/camp_dir
    out.mkdir(parents=True,exist_ok=True)

    ep_paths=[old/"mitigation_episode_metrics_long.csv"]
    st_paths=[old/"mitigation_step_metrics_long.csv"]
    for i in range(1,workers+1):
        d=wr/f"worker_{i:02d}_of_{workers:02d}"
        ep_paths.append(d/"mitigation_episode_metrics_long.csv")
        st_paths.append(d/"mitigation_step_metrics_acld_only.csv")
    ep=read_many(ep_paths)
    st=read_many(st_paths)
    if ep.empty:
        raise SystemExit("No mitigation episode rows found to merge.")
    if "p05_protocol" not in ep.columns:
        ep["p05_protocol"] = None
    ep["_priority"]=(ep["p05_protocol"].astype(str)==P05_PROTOCOL).astype(int)
    ep=ep.sort_values("_priority").drop_duplicates(["scenario","controller","seed"],keep="last").drop(columns="_priority")

    if not st.empty and "controller" in st.columns:
        st=st[st["controller"].astype(str)=="acld"].copy()
        keys=[c for c in ["scenario","seed","controller","step_idx"] if c in st.columns]
        if keys:
            st=st.drop_duplicates(keys,keep="last")

    specs=build_specs(ns,campaign)
    missing=[]
    for spec in specs:
        for controller in CONTROLLERS:
            for seed in spec.seeds:
                sub=ep[(ep["scenario"]==spec.name)&(ep["controller"].astype(str)==controller)&(pd.to_numeric(ep["seed"],errors="coerce")==int(seed))]
                if len(sub)<1:
                    missing.append({"scenario":spec.name,"controller":controller,"seed":int(seed)})

    ep.to_csv(out/"mitigation_episode_metrics_long.csv",index=False)
    if not st.empty:
        st.to_csv(out/"mitigation_step_metrics_acld_only.csv",index=False)
    ns["_aggregate_closed_loop_episode_df"](ep).to_csv(out/"mitigation_summary_long.csv",index=False)
    report={
        "protocol":P05_PROTOCOL,"campaign":campaign,"workers":workers,
        "episode_rows":int(len(ep)),"step_rows_acld_only":int(len(st)),
        "missing_controller_seed_units":len(missing),"missing_examples":missing[:25],
        "gate":"PASS" if not missing else "INCOMPLETE"
    }
    (out/"_merge_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",required=True)
    ap.add_argument("--campaign",required=True,choices=["A1","A2","A3","B1","B2","C1","C3"])
    ap.add_argument("--workers",type=int,default=3)
    args=ap.parse_args()
    project=Path(args.project_root).resolve()
    nb=project/"acld_full_experimental_campaign_P0_4_corrected.ipynb"
    ns=load_p04_namespace(nb)
    if args.campaign in DETECTION_CAMPAIGNS:
        merge_detection(project,args.campaign,ns,args.workers)
    else:
        merge_mitigation(project,args.campaign,ns,args.workers)

if __name__=="__main__":
    main()
