from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def has_command(name: str) -> bool:
    return shutil.which(str(name)) is not None


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout_s: float | None = None,
    check: bool = False,
) -> CommandResult:
    proc = subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    result = CommandResult(
        command=[str(x) for x in command],
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(result.command)}\n{result.stderr.strip()}")
    return result


def ffprobe_readable(video_path: Path, timeout_s: float = 20.0) -> bool:
    if not has_command("ffprobe"):
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = run_command(cmd, timeout_s=timeout_s, check=False)
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())

