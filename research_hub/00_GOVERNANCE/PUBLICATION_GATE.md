# Publication Gate - 20 October 2026 (snapshot from 18 October)

**Publication readiness: NOT READY**. Passing ZIP hashes is necessary but not enough.

| Task | Status | Closure criterion |
|---|---|---|
| Episode integrity | PASS | Verified SHA256, 8070 unique episode-method rows, 1690 units |
| A1 exact-bridge and reuse audit | PARTIAL | Confirm 70 reused P0.4 units/source revision, step-level consistency, then canonical gate |
| A2 Rician | FAIL / INCOMPLETE | Recover remaining 130 scenario-seed units, rerun canonical merge to 720/720 |
| A3 ablation | PARTIAL | Verified 60/60 episode-unit rows, step-level/compatibility/source audit |
| B1 and B2 closed-loop | PASS canonical | Confirm frozen source revision and paired controller evaluation |
| C1 scaling | PARTIAL | Rerun current six-shard merge report with PASS and independent step-level checks |
| C2 wall-clock runtime | PENDING | Serialize, isolate from concurrent workers, keep exact full-pipeline measurement boundary |
| C3 GCS sensitivity | PASS canonical | Interpret as network-node inclusion effect, not compute-node location |
| D1-D4 and E | PENDING | Paired CIs; prevalence control; calibration; failure modes; generate tables/figures |
| Revised paper | WORKING ONLY | Replace placeholders only after all cited data gates pass |

**No claimed field UAV/HIL validation; no distributed implementation; no 10-ms guarantee at n=1000.**
