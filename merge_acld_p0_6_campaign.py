from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from acld_p0_6_common import (
    P06_PROTOCOL,CAMPAIGN_DIRS,DETECTION_CAMPAIGNS,MITIGATION_CAMPAIGNS,
    CONTROLLERS,load_p04_namespace,build_specs
)

def read_many(paths):
    frames=[]
    for p in paths:
        p=Path(p)
        if p.exists() and p.stat().st_size:
            try: frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError: pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",required=True)
    ap.add_argument("--campaign",required=True,choices=["A1","A2","A3","B1","B2","C1","C3"])
    ap.add_argument("--workers",type=int,default=8)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    ns=load_p04_namespace(project/"acld_full_experimental_campaign_P0_4_corrected.ipynb")
    camp=a.campaign; camp_dir=CAMPAIGN_DIRS[camp]
    out=project/"acld_full_experiment_outputs_P0_6"/camp_dir
    out.mkdir(parents=True,exist_ok=True)

    if camp in DETECTION_CAMPAIGNS:
        ep_paths=[]; st_paths=[]
        for base in (
            project/"acld_full_experiment_outputs_P0_corrected"/camp_dir,
            project/"acld_full_experiment_outputs_P0_5"/camp_dir,
        ):
            ep_paths.append(base/"detection_episode_metrics_long.csv")
            st_paths += [base/"detection_step_metrics_acld_only.csv",base/"detection_step_metrics_long.csv"]
        wr=project/"acld_full_experiment_outputs_P0_6_workers"/camp_dir
        for i in range(1,a.workers+1):
            d=wr/f"worker_{i:02d}_of_{a.workers:02d}"
            ep_paths.append(d/"detection_episode_metrics_long.csv")
            st_paths.append(d/"detection_step_metrics_acld_only.csv")
        if camp=="A3":
            special=project/"acld_full_experiment_outputs_P0_5_A3_seed8_workers"/camp_dir
            for i in range(1,9):
                d=special/f"worker_{i:02d}_of_08"
                ep_paths.append(d/"detection_episode_metrics_long.csv")
                st_paths.append(d/"detection_step_metrics_acld_only.csv")
        ep=read_many(ep_paths); st=read_many(st_paths)
        if ep.empty: raise SystemExit("No episode rows found.")
        if "p06_protocol" not in ep.columns: ep["p06_protocol"]=None
        if "p05_protocol" not in ep.columns: ep["p05_protocol"]=None
        ep["_priority"]=(ep["p06_protocol"].notna().astype(int)*2 + ep["p05_protocol"].notna().astype(int))
        ep=ep.sort_values("_priority").drop_duplicates(["scenario","seed","method"],keep="last").drop(columns="_priority")
        if not st.empty:
            if "method" in st.columns: st=st[st["method"].astype(str)=="acld_v3"].copy()
            keys=[c for c in ("scenario","seed","method","step_idx") if c in st.columns]
            if keys: st=st.drop_duplicates(keys,keep="last")
        specs=build_specs(ns,camp); missing=[]
        for spec in specs:
            for seed in spec.seeds:
                sub=ep[(ep["scenario"].astype(str)==spec.name)&(pd.to_numeric(ep["seed"],errors="coerce")==int(seed))]
                got=set(sub["method"].dropna().astype(str)); req=set(map(str,spec.methods))
                if not req.issubset(got):
                    missing.append({"scenario":spec.name,"seed":int(seed),"missing_methods":sorted(req-got)})
        ep.to_csv(out/"detection_episode_metrics_long.csv",index=False)
        if not st.empty: st.to_csv(out/"detection_step_metrics_acld_only.csv",index=False)
        ns["_aggregate_detection_episode_df"](ep).to_csv(out/"detection_summary_long.csv",index=False)
        report={"protocol":P06_PROTOCOL,"campaign":camp,"episode_rows":len(ep),
                "step_rows_acld_only":len(st),"missing_units":len(missing),
                "missing_examples":missing[:20],"gate":"PASS" if not missing else "INCOMPLETE"}
    else:
        ep_paths=[]; st_paths=[]
        for base in (
            project/"acld_full_experiment_outputs_P0_corrected"/camp_dir,
            project/"acld_full_experiment_outputs_P0_5"/camp_dir,
        ):
            ep_paths.append(base/"mitigation_episode_metrics_long.csv")
            st_paths += [base/"mitigation_step_metrics_acld_only.csv",base/"mitigation_step_metrics_long.csv"]
        wr=project/"acld_full_experiment_outputs_P0_6_workers"/camp_dir
        for i in range(1,a.workers+1):
            d=wr/f"worker_{i:02d}_of_{a.workers:02d}"
            ep_paths.append(d/"mitigation_episode_metrics_long.csv")
            st_paths.append(d/"mitigation_step_metrics_acld_only.csv")
        ep=read_many(ep_paths); st=read_many(st_paths)
        if ep.empty: raise SystemExit("No mitigation rows found.")
        if "p06_protocol" not in ep.columns: ep["p06_protocol"]=None
        if "p05_protocol" not in ep.columns: ep["p05_protocol"]=None
        ep["_priority"]=(ep["p06_protocol"].notna().astype(int)*2 + ep["p05_protocol"].notna().astype(int))
        ep=ep.sort_values("_priority").drop_duplicates(["scenario","controller","seed"],keep="last").drop(columns="_priority")
        if not st.empty and "controller" in st.columns:
            st=st[st["controller"].astype(str)=="acld"].copy()
            keys=[c for c in ("scenario","seed","controller","step_idx") if c in st.columns]
            if keys: st=st.drop_duplicates(keys,keep="last")
        specs=build_specs(ns,camp); missing=[]
        for spec in specs:
            for controller in CONTROLLERS:
                for seed in spec.seeds:
                    sub=ep[(ep["scenario"].astype(str)==spec.name)&
                           (ep["controller"].astype(str)==controller)&
                           (pd.to_numeric(ep["seed"],errors="coerce")==int(seed))]
                    if len(sub)<1: missing.append({"scenario":spec.name,"controller":controller,"seed":int(seed)})
        ep.to_csv(out/"mitigation_episode_metrics_long.csv",index=False)
        if not st.empty: st.to_csv(out/"mitigation_step_metrics_acld_only.csv",index=False)
        ns["_aggregate_closed_loop_episode_df"](ep).to_csv(out/"mitigation_summary_long.csv",index=False)
        report={"protocol":P06_PROTOCOL,"campaign":camp,"episode_rows":len(ep),
                "step_rows_acld_only":len(st),"missing_units":len(missing),
                "missing_examples":missing[:20],"gate":"PASS" if not missing else "INCOMPLETE"}

    (out/"_merge_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
