from __future__ import annotations
import json, sys, types
from pathlib import Path
from typing import Dict, Tuple

P06_PROTOCOL = "P0.6-unit-shards-exact-bridge-microF1"

CAMPAIGN_DIRS = {
    "A1": "A1_detection_core",
    "A2": "A2_detection_rician",
    "A3": "A3_ablation",
    "B1": "B1_closed_loop_nominal",
    "B2": "B2_closed_loop_stress",
    "C1": "C1_accuracy_scaling",
    "C2": "C2_runtime_scaling",
    "C3": "C3_gcs_sensitivity",
}
DETECTION_CAMPAIGNS = {"A1","A2","A3","C1","C3"}
MITIGATION_CAMPAIGNS = {"B1","B2"}
CONTROLLERS = ("nh","random","tarjan_snapshot","tarjan_predict","acld")

def load_p04_namespace(notebook_path: Path) -> Dict[str, object]:
    # The exact bridge DFS can visit every vertex in the 1000-UAV graph.
    # Keep the same algorithm while allowing its traversal to complete.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    notebook_path = Path(notebook_path)
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    module = types.ModuleType("__acld_p06_core__")
    module.__file__ = str(notebook_path)
    sys.modules[module.__name__] = module
    ns = module.__dict__
    for idx in (2,3,4):
        cell = nb["cells"][idx]
        src = cell.get("source","")
        if isinstance(src,list):
            src = "".join(src)
        exec(compile(src, f"{notebook_path.name}:cell{idx}", "exec"), ns, ns)
    return ns

def build_specs(ns, campaign: str):
    campaign = campaign.upper()
    CampaignSpec = ns["CampaignSpec"]
    rician = ns["rician_kdb_to_nakagami_m"]
    MAIN_METHODS = tuple(ns["MAIN_METHODS"])
    ABLATION_METHODS = tuple(ns["ABLATION_METHODS"])
    seeds_full = tuple(range(42,62))
    seeds_scale = tuple(range(142,152))
    horizon_det = 200.0
    horizon_cl = 200.0

    def make_spec(name, *, n_uavs, density, jnr=None, horizon_s=horizon_det,
                  seeds=seeds_full, methods=MAIN_METHODS, rician_k_db=None,
                  include_ground_station=None):
        overrides = dict(n_uavs=n_uavs, density_scale_mode=density, jnr_db=jnr)
        if include_ground_station is not None:
            overrides["include_ground_station"] = bool(include_ground_station)
        if rician_k_db is not None:
            m = rician(float(rician_k_db))
            overrides["m_fading"] = m
            overrides["gs_m_fading"] = m
        return CampaignSpec(name=name, cfg_overrides=overrides, horizon_s=horizon_s,
                            seeds=tuple(seeds), methods=tuple(methods))
    specs=[]
    if campaign=="A1":
        for n in (150,300,500):
            for density in ("sparse","medium","dense"):
                for jnr in (None,5.0,15.0):
                    tag="NoJam" if jnr is None else f"J{int(jnr)}"
                    specs.append(make_spec(f"A1_n{n}_{density}_{tag}_Nakagami",
                                           n_uavs=n,density=density,jnr=jnr))
    elif campaign=="A2":
        for n in (150,300,500):
            for density in ("sparse","medium","dense"):
                for kdb in (3.0,0.0):
                    for jnr in (None,15.0):
                        tag="NoJam" if jnr is None else f"J{int(jnr)}"
                        specs.append(make_spec(f"A2_n{n}_{density}_{tag}_RicianK{int(kdb)}",
                                               n_uavs=n,density=density,jnr=jnr,rician_k_db=kdb))
    elif campaign=="A3":
        specs=[
            make_spec("A3_medium_NoJam",n_uavs=150,density="medium",jnr=None,methods=ABLATION_METHODS),
            make_spec("A3_dense_NoJam",n_uavs=150,density="dense",jnr=None,methods=ABLATION_METHODS),
            make_spec("A3_medium_J15",n_uavs=150,density="medium",jnr=15.0,methods=ABLATION_METHODS),
        ]
    elif campaign=="B1":
        for n in (150,300):
            for density in ("sparse","medium","dense"):
                specs.append(make_spec(f"B1_n{n}_{density}_NoJam",
                                       n_uavs=n,density=density,jnr=None,horizon_s=horizon_cl))
    elif campaign=="B2":
        for n in (150,300):
            for density in ("sparse","medium","dense"):
                specs.append(make_spec(f"B2_n{n}_{density}_J15_RicianK0",
                                       n_uavs=n,density=density,jnr=15.0,horizon_s=horizon_cl,
                                       rician_k_db=0.0))
    elif campaign=="C1":
        for n in (50,150,300,500,1000):
            for jnr in (None,15.0):
                tag="NoJam" if jnr is None else f"J{int(jnr)}"
                specs.append(make_spec(f"C1_n{n}_medium_{tag}",
                                       n_uavs=n,density="medium",jnr=jnr,seeds=seeds_scale))
    elif campaign=="C3":
        for n in (150,300):
            for jnr in (None,15.0):
                tag="NoJam" if jnr is None else f"J{int(jnr)}"
                for include_gs in (True,False):
                    gs="GCS_ON" if include_gs else "GCS_OFF"
                    specs.append(make_spec(f"C3_n{n}_medium_{tag}_{gs}",
                                           n_uavs=n,density="medium",jnr=jnr,
                                           methods=("acld_v3","tarjan_predict_guard","tarjan_snapshot_guard"),
                                           include_ground_station=include_gs))
    elif campaign=="C2":
        return []
    else:
        raise ValueError(campaign)
    return specs

def base_cfg(ns):
    ACLDConfig=ns["ACLDConfig"]
    return ACLDConfig(
        n_uavs=150, include_ground_station=True, density_scale_mode="medium",
        jnr_db=None, alpha=0.0, beta=0.10, gamma=0.20, delta=0.10,
        zeta=0.05, risk_band_fraction=0.97, seed=42
    )

def flatten_units(specs, campaign):
    units=[]
    if campaign in DETECTION_CAMPAIGNS:
        for spec in specs:
            for seed in spec.seeds:
                units.append((spec, int(seed)))
    elif campaign in MITIGATION_CAMPAIGNS:
        for spec in specs:
            for controller in CONTROLLERS:
                for seed in spec.seeds:
                    units.append((spec, controller, int(seed)))
    return units

def assigned_units(units, worker_index:int, workers:int):
    if worker_index<1 or worker_index>workers:
        raise ValueError("worker_index must be in 1..workers")
    return [u for i,u in enumerate(units) if i % workers == (worker_index-1)]
