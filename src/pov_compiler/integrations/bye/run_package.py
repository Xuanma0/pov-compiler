from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pov_compiler.integrations.bye.entrypoints import EntryPointResolver


def resolve_bye_root(cli_arg: str | None, env_var: str = "BYE_ROOT") -> Path | None:
    candidates: list[Path] = []
    if cli_arg:
        candidates.append(Path(cli_arg))
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value))

    repo_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        [
            repo_root.parent / "Project-Be-your-eyes",
            repo_root.parent / "project-be-your-eyes",
            repo_root.parent / "be-your-eyes",
            repo_root.parent / "BYE",
            repo_root.parent / "paperbye",
        ]
    )

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.exists() or not path.is_dir():
            continue
        if _looks_like_bye_root(path):
            return path.resolve()
    return None


def _looks_like_bye_root(path: Path) -> bool:
    marker_files = [
        "Gateway/scripts/lint_run_package.py",
        "scripts/lint_run_package.py",
        "report_run.py",
        "run_regression_suite.py",
        "pyproject.toml",
    ]
    matches = 0
    for rel in marker_files:
        if (path / rel).exists():
            matches += 1
    return matches >= 1


def find_bye_entrypoints(bye_root: Path) -> dict[str, str | None]:
    resolution = EntryPointResolver(bye_root=Path(bye_root)).resolve()

    def _to_rel(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return str(Path(p).resolve().relative_to(Path(bye_root).resolve())).replace("\\", "/")
        except Exception:
            return str(p)

    return {
        "lint": _to_rel(resolution.lint),
        "report": _to_rel(resolution.report),
        "regression": _to_rel(resolution.regression),
    }


def build_run_package(
    out_dir: str | Path,
    video_id: str,
    events_jsonl_path: str | Path,
    video_path: str | Path | None = None,
    video_mode: str = "copy",
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    out_root = Path(out_dir)
    run_pkg = out_root / "run_package"
    run_pkg.mkdir(parents=True, exist_ok=True)

    events_dir = run_pkg / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    src_events = Path(events_jsonl_path)
    dst_events = events_dir / "events_v1.jsonl"
    shutil.copyfile(src_events, dst_events)

    manifest: dict[str, Any] = {
        "version": "v1",
        "video_id": str(video_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events_relpath": "events/events_v1.jsonl",
    }

    if video_path:
        src_video = Path(video_path)
        if src_video.exists() and src_video.is_file():
            video_dir = run_pkg / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            dst_video = video_dir / src_video.name
            if str(video_mode).lower() == "link":
                try:
                    if dst_video.exists():
                        dst_video.unlink()
                    os.link(str(src_video), str(dst_video))
                except Exception:
                    shutil.copyfile(src_video, dst_video)
            else:
                shutil.copyfile(src_video, dst_video)
            manifest["video_relpath"] = str(Path("video") / src_video.name).replace("\\", "/")
            manifest["video_name"] = src_video.name

    if extra_meta:
        manifest["extra_meta"] = dict(extra_meta)

    (run_pkg / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    git_commit = get_git_commit_short(Path(__file__).resolve().parents[4])
    try:
        from pov_compiler import __version__ as pov_version
    except Exception:
        pov_version = "unknown"

    (run_pkg / "metadata.json").write_text(
        json.dumps(
            {
                "video_id": str(video_id),
                "source": "pov-compiler",
                "events_format": "events_v1.jsonl",
                "created_at_utc": manifest["created_at_utc"],
                "created_utc": manifest["created_at_utc"],
                "git_commit": git_commit,
                "pov_compiler_version": str(pov_version),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_pkg


def run_bye_tool(
    py: str,
    bye_root: str | Path,
    module_or_script_path: str,
    args_list: list[str],
    cwd: str | Path | None = None,
    log_dir: str | Path | None = None,
    log_prefix: str | None = None,
) -> dict[str, Any]:
    root = Path(bye_root)
    script_rel = Path(module_or_script_path)
    script_path = script_rel if script_rel.is_absolute() else (root / script_rel)

    logs_root = Path(log_dir) if log_dir else root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    stem = log_prefix or script_path.stem
    stdout_log = logs_root / f"{stem}.stdout.log"
    stderr_log = logs_root / f"{stem}.stderr.log"

    cmd = [str(py), str(script_path), *[str(x) for x in args_list]]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_log.write_text(proc.stdout or "", encoding="utf-8")
    stderr_log.write_text(proc.stderr or "", encoding="utf-8")

    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else str(root),
        "returncode": int(proc.returncode),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "script_path": str(script_path),
    }


def collect_report_candidates(base_dir: str | Path) -> list[Path]:
    root = Path(base_dir)
    candidates: list[Path] = []
    for name in ("report.json", "run_report.json", "summary.json"):
        for p in root.rglob(name):
            if p.is_file():
                candidates.append(p)
    candidates.sort(key=lambda p: (len(str(p)), str(p).lower()))
    return candidates


def infer_args_for_tool(tool: str, run_package_dir: Path, report_out_dir: Path) -> list[list[str]]:
    rp = str(run_package_dir.resolve())
    rd = str(report_out_dir.resolve())
    if tool == "lint":
        return [
            ["--run-package", rp],
            ["--run_package", rp],
            ["--package", rp],
            [rp],
        ]
    if tool == "report":
        return [
            ["--run-package", rp, "--out-dir", rd],
            ["--run_package", rp, "--out_dir", rd],
            ["--package", rp, "--out", rd],
            [rp, rd],
            ["--run-package", rp],
            [rp],
        ]
    if tool == "regression":
        return [
            ["--run-package", rp],
            ["--run_package", rp],
            ["--package", rp],
            [rp],
        ]
    return [[]]


def get_git_commit_short(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[4]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        value = (proc.stdout or "").strip()
        return value or None
    except Exception:
        return None


def python_executable() -> str:
    return sys.executable
