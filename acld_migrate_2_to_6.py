#!/usr/bin/env python3
"""Repartition ACLD A2/C1 P0.6 worker CSVs from 2 to 6 shards, non-destructively.

Run ONLY when all original P0.6 queue/worker processes have stopped.
Run the ORIGINAL merge_acld_p0_6_campaign.py --workers 2 for A2 and C1 first.
This program does not modify legacy shard files, canonical outputs, or project code.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path


def specs_from_project(project: Path, campaign: str):
    sys.path.insert(0, str(project))
    from acld_p0_6_common import build_specs, flatten_units

    class Spec:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    ns = {
        "CampaignSpec": Spec,
        "rician_kdb_to_nakagami_m": lambda k: 1.0,
        "MAIN_METHODS": ("acld_v3", "cut_predict_guard", "ebc_snapshot_guard",
                         "tarjan_predict_guard", "tarjan_snapshot_guard"),
        "ABLATION_METHODS": ("acld_v3",),
    }
    units = flatten_units(build_specs(ns, campaign), campaign)
    expected = {"A2": 720, "C1": 100}[campaign]
    if len(units) != expected:
        raise RuntimeError(f"{campaign} specification changed: {len(units)} != {expected}")
    locations = {}
    for i, (spec, seed) in enumerate(units):
        key = (spec.name, int(seed))
        if key in locations:
            raise RuntimeError(f"Duplicate protocol unit: {key}")
        locations[key] = (i % 6) + 1
    return locations


def repartition_csv(campaign: str, filename: str, old_base: Path, temp_base: Path, locations):
    sources = [old_base / f"worker_{j:02d}_of_02" / filename for j in (1, 2)]
    if not all(p.is_file() and p.stat().st_size > 0 for p in sources):
        raise RuntimeError(f"{campaign}: missing old 2-worker CSV: {sources}")

    writers = {}
    outs = []
    expected_header = None
    incoming_rows = 0
    outgoing_rows = 0
    covered = set()
    try:
        for src in sources:
            with src.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames
                if not header or not {"scenario", "seed"}.issubset(set(header)):
                    raise RuntimeError(f"Invalid CSV header: {src}")
                if expected_header is None:
                    expected_header = list(header)
                    for j in range(1, 7):
                        dest = temp_base / f"worker_{j:02d}_of_06" / filename
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        fout = dest.open("w", newline="", encoding="utf-8")
                        outs.append(fout)
                        writer = csv.DictWriter(fout, fieldnames=expected_header, lineterminator="\n")
                        writer.writeheader()
                        writers[j] = writer
                elif list(header) != expected_header:
                    raise RuntimeError(f"Inconsistent columns between old shards: {src}")
                for row_no, row in enumerate(reader, start=2):
                    if None in row or any(v is None for v in row.values()):
                        raise RuntimeError(f"Malformed/truncated CSV row: {src}, line {row_no}")
                    try:
                        seed = float(row["seed"])
                        if not seed.is_integer():
                            raise ValueError("seed not integral")
                        key = (str(row["scenario"]), int(seed))
                        dst_worker = locations[key]
                    except (ValueError, KeyError, TypeError) as ex:
                        raise RuntimeError(f"Invalid/out-of-protocol unit in {src}:{row_no}: {ex}") from ex
                    writers[dst_worker].writerow(row)
                    incoming_rows += 1
                    outgoing_rows += 1
                    covered.add(key)
    finally:
        for f in outs:
            f.close()
    if incoming_rows != outgoing_rows:
        raise RuntimeError(f"Row mismatch for {filename}")
    return {"filename": filename, "rows": incoming_rows, "unit_keys": len(covered)}


def migrate_one(project: Path, campaign: str, min_complete: int):
    campdir = {"A2": "A2_detection_rician", "C1": "C1_accuracy_scaling"}[campaign]
    root = project / "acld_full_experiment_outputs_P0_6_workers" / campdir
    canon = project / "acld_full_experiment_outputs_P0_6" / campdir
    report_file = canon / "_merge_report.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("campaign") != campaign:
        raise RuntimeError(f"Unexpected canonical report: {report_file}")
    complete = {"A2": 720, "C1": 100}[campaign] - int(report.get("missing_units", -1))
    if complete < min_complete:
        raise RuntimeError(
            f"{campaign} source incomplete/unsynced: canonical {complete} < required {min_complete}. "
            "Do not resume from this E: copy until Drive sync is complete."
        )
    for j in range(1, 7):
        d = root / f"worker_{j:02d}_of_06"
        if d.exists():
            raise RuntimeError(f"6-worker folder already exists, refusing overwrite: {d}")

    locations = specs_from_project(project, campaign)
    parent = root.parent
    stage = Path(tempfile.mkdtemp(prefix=f"_{campdir}_migrate_", dir=parent))
    try:
        metrics = [
            repartition_csv(campaign, "detection_episode_metrics_long.csv", root, stage, locations),
            repartition_csv(campaign, "detection_step_metrics_acld_only.csv", root, stage, locations),
        ]
        if not metrics[0]["rows"] or not metrics[1]["rows"]:
            raise RuntimeError(f"Empty episode/step metrics for {campaign}")
        for j in range(1, 7):
            target = root / f"worker_{j:02d}_of_06"
            shutil.move(str(stage / target.name), str(target))
        print(f"{campaign}: {complete} canonical units preserved; legacy worker rows repartitioned: {metrics}")
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--min-a2", type=int, default=529)
    ap.add_argument("--min-c1", type=int, default=92)
    args = ap.parse_args()
    project = args.project_root.resolve()
    if not project.is_dir():
        raise SystemExit(f"Project not found: {project}")
    # Both campaigns are validated before any destination is written.
    for c, min_count in (("A2", args.min_a2), ("C1", args.min_c1)):
        campdir = {"A2": "A2_detection_rician", "C1": "C1_accuracy_scaling"}[c]
        base = project / "acld_full_experiment_outputs_P0_6_workers" / campdir
        out = project / "acld_full_experiment_outputs_P0_6" / campdir
        rep = json.loads((out / "_merge_report.json").read_text(encoding="utf-8"))
        if {"A2": 720, "C1": 100}[c] - int(rep["missing_units"]) < min_count:
            raise SystemExit(f"{c} canonical merge has fewer completed units than last verified cloud state")
        for w in range(1, 7):
            if (base / f"worker_{w:02d}_of_06").exists():
                raise SystemExit(f"Found existing six-worker shard ({c}/{w}). No files modified.")
        for w in (1, 2):
            for file in ("detection_episode_metrics_long.csv", "detection_step_metrics_acld_only.csv"):
                if not (base / f"worker_{w:02d}_of_02" / file).is_file():
                    raise SystemExit(f"Missing old two-worker input: {c}/worker_{w:02d}_of_02/{file}")
    migrate_one(project, "A2", args.min_a2)
    migrate_one(project, "C1", args.min_c1)
    print("OK: Ready to start P0.6 parallel queue with --workers-per-campaign 6")


if __name__ == "__main__":
    main()
