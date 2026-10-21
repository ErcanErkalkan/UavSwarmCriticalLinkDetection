#!/usr/bin/env python3
"""Verify immutable episode snapshot integrity and completeness; does not launch simulations.
Usage: python 03_INDEPENDENT_AUDIT/verify_release.py
"""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,sys
P=Path(__file__).resolve().parents[1]/"02_CANONICAL_SNAPSHOT_20261018"
v=json.loads((P/"ACLD_P05_P06_MERGE_VALIDATION_20261018.json").read_text(encoding="utf-8"))
errors=[]
for name,digest in v["output_sha256"].items():
 path=P/name
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
  errors.append("HASH_MISMATCH_OR_MISSING: "+name)
def load(name):
 with (P/name).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
rows=load("ACLD_P05_P06_ALL_EPISODES_20261018.csv")
det=load("ACLD_P05_P06_DETECTION_EPISODES_20261018.csv")
mit=load("ACLD_P06_MITIGATION_EPISODES_20261018.csv")
status=load("ACLD_P05_P06_CAMPAIGN_STATUS_20261018.csv")
counts=Counter(x["campaign"] for x in rows)
if len(rows)!=len(det)+len(mit):errors.append("RECORD_TYPE_COUNT_MISMATCH")
if len(rows)!=8070:errors.append("UNEXPECTED_SNAPSHOT_ROW_COUNT")
keys=[(x["campaign"],x["scenario"],x["seed"],x["method_or_controller"]) for x in rows]
if len(keys)!=len(set(keys)):errors.append("DUPLICATED_EPISODE_KEY")
for s in status:
 c=s["campaign"]
 if counts[c]!=int(s["episode_rows"]):errors.append("CAMPAIGN_COUNT_MISMATCH:"+c)
units=len({(x["campaign"],x["scenario"],x["seed"]) for x in rows})
if units!=1690:errors.append("UNEXPECTED_SEED_UNIT_COUNT")
print("INTEGRITY:","PASS" if not errors else "FAIL")
print("RECORDS:",len(rows),"DETECTION:",len(det),"MITIGATION:",len(mit),"UNITS:",units)
print("PUBLICATION GATE: NOT_READY (A2/C1/C2/D1-D4/E and protocol comparability)")
for e in errors:print("ERROR:",e)
sys.exit(1 if errors else 0)
