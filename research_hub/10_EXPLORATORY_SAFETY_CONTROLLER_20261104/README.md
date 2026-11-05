# S5.6 experimental controller audit (4 November 2026)

**Status: EXPLORATORY ONLY; NO VERIFIED SAFETY OR HARD REAL-TIME CLAIM.**

This experimental controller stream must not be used to strengthen the already evaluated S1–S4 partition-warning claim without new independent tests.

- **S5.6.6:** 672 controlled pairs, 26,880 paired future kinematic steps; natural-state branch: 594 positions, 120 checkpoints, 7 viable selected plans, **0 natural safety vetoes**. Integrity validation passed; operational utility not established.
- **S5.6.7:** 1,188 natural states across 12 dependent trajectories, 334 probes, 13 unique approved plan episodes and 39 repeated rollouts. **0 approved plan/risk coincidence**, **0 natural vetoes**, 2 unique next-step connectivity benefits outside risk events. Integrity validation passed, not a population-level safety outcome.
- Reserved holdout seeds **410–419 were not used**; preserve the holdout lock.
- Current S5.6.7 working evidence folder: [Google Drive](https://drive.google.com/drive/folders/1g2GM3yYB2rbrRIbREgvRJG2_mCWZ7uR4).

The public GitHub source tree is an October P0.5/P0.6 experimental-runner source line and **does not contain an audited, source-hash-verified copy of the latest S5 controller engine**. Source hashes in the reports identify artifacts in Drive; no file copy or independent physical validation is claimed here.

Remaining research gates: natural risk/action overlap, repeated independent holdout evaluation after controller freeze, safety feasibility, controller-level longitudinal effect, actual packet overhead and hardware-timed deadlines.
