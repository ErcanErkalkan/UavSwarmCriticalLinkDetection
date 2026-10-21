# Independent read-only checks

`verify_release.py` checks the archived episode snapshot file hashes, row counts, distinct episode identifiers and campaign totals. Run it from the full downloaded Drive release (with `02_CANONICAL_SNAPSHOT_20261018/` as its sibling).

```bash
python 03_INDEPENDENT_AUDIT/verify_release.py
```

The script reports **INTEGRITY: PASS** when data match the archived 18 October manifest, but still reports **PUBLICATION GATE: NOT_READY**. It does not run experiments or verify physical UAV deployment.
