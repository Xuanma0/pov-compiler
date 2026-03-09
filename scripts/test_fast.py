from __future__ import annotations

import subprocess
import sys
from typing import Sequence


_XDIST_MISSING_PATTERNS = (
    "no module named xdist",
    "unrecognized arguments: -n",
    "unrecognized arguments: --numprocesses",
    "unknown argument: -n",
)


def _render_cmd(cmd: Sequence[str]) -> str:
    return " ".join(str(x) for x in cmd)


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def _is_xdist_missing(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return any(pattern in text for pattern in _XDIST_MISSING_PATTERNS)


def main() -> int:
    cmd_fast = [sys.executable, "-m", "pytest", "-q", "-n", "auto"]
    proc_fast = _run(cmd_fast)
    if proc_fast.returncode == 0:
        print("xdist_enabled=true")
        print(f"cmd={_render_cmd(cmd_fast)}")
        print(f"returncode={int(proc_fast.returncode)}")
        if proc_fast.stdout:
            print(proc_fast.stdout, end="" if proc_fast.stdout.endswith("\n") else "\n")
        if proc_fast.stderr:
            print(proc_fast.stderr, file=sys.stderr, end="" if proc_fast.stderr.endswith("\n") else "\n")
        return 0

    if _is_xdist_missing(proc_fast.stdout or "", proc_fast.stderr or ""):
        cmd_fallback = [sys.executable, "-m", "pytest", "-q"]
        proc_fallback = _run(cmd_fallback)
        print("xdist_enabled=false")
        print(f"cmd={_render_cmd(cmd_fallback)}")
        print(f"returncode={int(proc_fallback.returncode)}")
        if proc_fallback.stdout:
            print(proc_fallback.stdout, end="" if proc_fallback.stdout.endswith("\n") else "\n")
        if proc_fallback.stderr:
            print(proc_fallback.stderr, file=sys.stderr, end="" if proc_fallback.stderr.endswith("\n") else "\n")
        return int(proc_fallback.returncode)

    print("xdist_enabled=true")
    print(f"cmd={_render_cmd(cmd_fast)}")
    print(f"returncode={int(proc_fast.returncode)}")
    if proc_fast.stdout:
        print(proc_fast.stdout, end="" if proc_fast.stdout.endswith("\n") else "\n")
    if proc_fast.stderr:
        print(proc_fast.stderr, file=sys.stderr, end="" if proc_fast.stderr.endswith("\n") else "\n")
    return int(proc_fast.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
