"""Retrospective S2 onset audit (no new simulation and no parameter tuning).
Usage: python verify_S2_onset_v2.py --s2 path/to/ACLD_S2_40_OF_40_FINAL_20261102.zip --out ./audit
Requires NumPy/pandas from requirements.txt; checks raw unit CSVs and SHA sidecars.
"""
import argparse
import hashlib
import io
import json
import pathlib
import zipfile
import numpy as np
import pandas as pd

EXPECTED_SHA = "3295d3fdafd539e90f9ce9f75355b3dbd6d0296a4c0a330bf2e1c27051cb4e05"
METHODS = ("acld_v3", "tarjan_predict_guard")

def div(a, b):
    return float(a / b) if b else None

def main(zip_path, outdir):
    zip_path = pathlib.Path(zip_path)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == EXPECTED_SHA, "S2 archive SHA256 mismatch"
    units = []
    verified = 0
    with zipfile.ZipFile(zip_path) as z:
        files = sorted(n for n in z.namelist() if "/units/" in n and n.endswith(".csv"))
        assert len(files) == 40, f"Expected 40 paired unit CSV files; got {len(files)}"
        for name in files:
            payload = z.read(name)
            sidecar = json.loads(z.read(name[:-4] + ".json"))
            assert hashlib.sha256(payload).hexdigest() == sidecar["sha256"], name
            df = pd.read_csv(io.BytesIO(payload), low_memory=False)
            assert len(df) == 7998 and set(df.method) == set(METHODS), name
            a = df[df.method == METHODS[0]].sort_values("step_idx")
            t = df[df.method == METHODS[1]].sort_values("step_idx")
            assert len(a) == len(t) == 3999
            assert np.array_equal(a.step_idx.to_numpy(), np.arange(3999))
            assert np.array_equal(t.step_idx.to_numpy(), np.arange(3999))
            for field in ("label_partition_next", "lambda2_current", "n_components_pred_viable"):
                assert np.array_equal(a[field].to_numpy(), t[field].to_numpy(), equal_nan=True), (name, field)
            verified += 1
            for method, df2 in ((METHODS[0], a), (METHODS[1], t), ("components_only", a)):
                current_connected = df2.lambda2_current.to_numpy() > 1e-10
                actual = df2.label_partition_next.astype(int).to_numpy() > 0
                pred = (df2.n_components_pred_viable.to_numpy() > 1) if method == "components_only" else (df2.alert.astype(int).to_numpy() > 0)
                positive = current_connected & actual
                negative = current_connected & ~actual
                units.append({
                    "unit": name.split("/")[-1].removesuffix(".csv"),
                    "scenario": str(df2.scenario.iloc[0]), "seed": int(df2.seed.iloc[0]),
                    "method": method, "total": len(df2),
                    "connected_current": int(current_connected.sum()),
                    "already_disconnected_current": int((~current_connected).sum()),
                    "at_risk_onsets": int(positive.sum()), "at_risk_negatives": int(negative.sum()),
                    "TP": int((positive & pred).sum()), "FN": int((positive & ~pred).sum()),
                    "FP": int((negative & pred).sum()), "TN": int((negative & ~pred).sum()),
                    "all_label_FP": int(((~actual) & pred).sum()),
                    "all_label_FN": int((actual & ~pred).sum())
                })
    df = pd.DataFrame(units)
    assert len(df) == 120 and df.seed.nunique() == 10 and df.scenario.nunique() == 4
    df.to_csv(outdir / "S2_onset_verified_per_unit.csv", index=False)
    summary = {}
    keys = ("total", "connected_current", "already_disconnected_current", "at_risk_onsets",
            "at_risk_negatives", "TP", "FN", "FP", "TN", "all_label_FP", "all_label_FN")
    for method in (*METHODS, "components_only"):
        sums = df[df.method == method].select_dtypes(include="number").sum()
        summary[method] = {k: int(sums[k]) for k in keys}
        summary[method].update({
            "FPR_at_risk": div(sums["FP"], sums["at_risk_negatives"]),
            "recall_at_risk": div(sums["TP"], sums["at_risk_onsets"]),
            "precision_at_risk": div(sums["TP"], sums["TP"] + sums["FP"])
        })
    assert (summary["acld_v3"]["TP"], summary["acld_v3"]["FP"]) == (45, 117)
    assert (summary["tarjan_predict_guard"]["TP"], summary["tarjan_predict_guard"]["FP"]) == (46, 712)
    assert (summary["components_only"]["TP"], summary["components_only"]["FP"]) == (40, 4)
    arr = {}
    for method in METHODS:
        arr[method] = df[df.method == method].groupby("seed")[
            ["TP", "FN", "FP", "TN", "at_risk_onsets", "at_risk_negatives"]
        ].sum().sort_index().to_numpy(dtype=float)
    rng = np.random.default_rng(20261104)
    samples = []
    for _ in range(5000):
        ix = rng.integers(0, 10, 10)
        a = arr[METHODS[0]][ix].sum(axis=0)
        t = arr[METHODS[1]][ix].sum(axis=0)
        samples.append((a[2] / a[5], t[2] / t[5], a[2] / a[5] - t[2] / t[5]))
    samples = np.asarray(samples)
    bootstrap = {
        key: {
            "lower_95": float(np.percentile(samples[:, i], 2.5)),
            "upper_95": float(np.percentile(samples[:, i], 97.5))
        } for i, key in enumerate(("FPR_ACLD", "FPR_Tarjan", "FPR_difference_ACLD_minus_Tarjan"))
    }
    result = {
        "status": "PASS_RECORDED_S2_INTEGRITY_AND_RETROSPECTIVE_REANALYSIS",
        "interpretation": "Post-hoc conditional sensitivity; not preregistered or field validated",
        "verified_unit_sha256": verified, "unique_seed_clusters": 10, "fixed_scenario_strata": 4,
        "source_archive_sha256": EXPECTED_SHA,
        "counts_and_metrics": summary, "bootstrap_5000_whole_seed_clusters": bootstrap
    }
    (outdir / "S2_onset_verified_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.s2, args.out)
