"""Atomic status writes resilient to temporary Windows file locks."""
import json
import os
import tempfile
import time
from pathlib import Path


def atomic_json(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        for attempt in range(60):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 59:
                    raise
                time.sleep(min(0.1 * (attempt + 1), 1.0))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except PermissionError:
                pass
