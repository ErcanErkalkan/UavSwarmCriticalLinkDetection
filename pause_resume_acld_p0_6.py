"""Suspend/resume this project's P0.6 processes without terminating them."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from acld_p0_6_io import atomic_json

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "acld_p0_6_logs_PARALLEL" / "SUSPENDED_PROCESSES.json"
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
native = ctypes.WinDLL("ntdll")
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
for fn in (native.NtSuspendProcess, native.NtResumeProcess):
    fn.argtypes = [wintypes.HANDLE]
    fn.restype = wintypes.LONG


def identity(handle):
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    code = wintypes.DWORD()
    if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
        raise RuntimeError("Process has exited")
    return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime


def discover():
    query = "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match 'run_acld_p0_6_(parallel_queue|unit_worker)\\.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, check=True)
    raw = result.stdout.decode("utf-8", errors="replace").strip()
    rows = json.loads(raw) if raw else []
    if isinstance(rows, dict):
        rows = [rows]
    # Restrict to this project and exclude any A1 worker explicitly.
    return [r for r in rows if str(ROOT).casefold() in r["CommandLine"].casefold()
            and "--campaign A1" not in r["CommandLine"]]


def change(row, resume=False):
    pid = row["pid"]
    handle = kernel.OpenProcess(0x0800 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        created = identity(handle)
        if row.get("creation_filetime", created) != created:
            raise RuntimeError(f"PID {pid} has been reused; refusing to change it")
        row["creation_filetime"] = created
        result = (native.NtResumeProcess if resume else native.NtSuspendProcess)(handle)
        if result != 0:
            raise RuntimeError(f"PID {pid}: NTSTATUS {result:#x}")
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pause", "resume"])
    args = parser.parse_args()
    state = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"processes": []}
    if args.action == "pause":
        state["paused_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["state"] = "PAUSING"
        atomic_json(MANIFEST, state)
        # Suspend the queue first, then rediscover workers so it cannot launch more.
        for queue_only in (True, False):
            for item in discover():
                is_queue = "run_acld_p0_6_parallel_queue.py" in item["CommandLine"]
                if is_queue != queue_only:
                    continue
                pid = item["ProcessId"]
                if any(r["pid"] == pid and r.get("suspended") for r in state["processes"]):
                    continue
                row = {"pid": pid, "command_line": item["CommandLine"], "queue": is_queue}
                change(row)
                row["suspended"] = True
                state["processes"].append(row)
                atomic_json(MANIFEST, state)
                print(f"Suspended PID {pid}", flush=True)
        state["state"] = "SUSPENDED"
    else:
        # Workers first, scheduler last. Validate PID creation time before resuming.
        for row in sorted(state["processes"], key=lambda r: r["queue"]):
            if not row.get("suspended"):
                continue
            change(row, resume=True)
            row["suspended"] = False
            atomic_json(MANIFEST, state)
            print(f"Resumed PID {row['pid']}", flush=True)
        state["state"] = "RESUMED"
        state["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(MANIFEST, state)
    print(state["state"], flush=True)


if __name__ == "__main__":
    main()
