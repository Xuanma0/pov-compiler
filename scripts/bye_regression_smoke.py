from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.exporter import export_bye_events_from_output_dict, write_jsonl
from pov_compiler.integrations.bye.run_package import (
    build_run_package,
    collect_report_candidates,
    find_bye_entrypoints,
    get_git_commit_short,
    infer_args_for_tool,
    python_executable,
    resolve_bye_root,
    run_bye_tool,
)


def _infer_video_id(output_dict: dict[str, Any], json_path: Path, cli_video_id: str | None) -> str:
    if cli_video_id:
        return str(cli_video_id)
    vid = output_dict.get("video_id")
    if isinstance(vid, str) and vid:
        return vid
    stem = json_path.stem
    for suffix in ("_v03_decisions", "_v02_token", "_v01_decision", "_v01", "_v0"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POV->BYE regression smoke (offline injection loop)")
    parser.add_argument("--pov_json", required=True, help="Path to POV output json (e.g., *_v03_decisions.json)")
    parser.add_argument("--video_id", default=None, help="Optional video id override")
    parser.add_argument("--video", default=None, help="Optional video path for BYE run package")
    parser.add_argument("--out_dir", required=True, help="Output directory for run package/logs/snapshot")
    parser.add_argument("--bye_root", default=None, help="External BYE repo root")
    parser.add_argument("--skip_lint", action="store_true", help="Skip BYE lint step")
    parser.add_argument("--skip_report", action="store_true", help="Skip BYE report step")
    parser.add_argument("--skip_regression", action="store_true", help="Skip BYE regression step")
    parser.add_argument(
        "--include",
        default="events_v1,highlights,tokens,decisions",
        help="Sections to export to BYE events jsonl",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing BYE or failed step")
    return parser.parse_args()


def _write_snapshot(snapshot: dict[str, Any], out_dir: Path) -> Path:
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return snapshot_path


def _to_rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _copy_report_outputs(out_dir: Path, candidates: list[Path]) -> list[str]:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for idx, src in enumerate(candidates):
        name = src.name if idx == 0 else f"{src.stem}_{idx}{src.suffix}"
        dst = report_dir / name
        if src.resolve() == dst.resolve():
            copied.append(str(dst))
            continue
        shutil.copyfile(src, dst)
        copied.append(str(dst))
    return copied


def _run_tool_with_attempts(
    *,
    tool_name: str,
    script_rel: str,
    bye_root: Path,
    run_package_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    logs_dir = out_dir / "logs"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    attempts_meta: list[dict[str, Any]] = []
    arg_candidates = infer_args_for_tool(tool_name, run_package_dir=run_package_dir, report_out_dir=report_dir)
    last_result: dict[str, Any] | None = None
    for i, args_list in enumerate(arg_candidates):
        prefix = f"{tool_name}.attempt_{i+1:02d}"
        result = run_bye_tool(
            py=python_executable(),
            bye_root=bye_root,
            module_or_script_path=script_rel,
            args_list=args_list,
            cwd=bye_root,
            log_dir=logs_dir,
            log_prefix=prefix,
        )
        attempt = {
            "attempt": i + 1,
            "args": [str(x) for x in args_list],
            "returncode": int(result.get("returncode", 1)),
            "stdout_log": str(result.get("stdout_log", "")),
            "stderr_log": str(result.get("stderr_log", "")),
        }
        attempts_meta.append(attempt)
        last_result = result
        if int(result.get("returncode", 1)) == 0:
            break

    return {
        "tool": tool_name,
        "script_relpath": script_rel.replace("\\", "/"),
        "attempts": attempts_meta,
        "ok": bool(last_result and int(last_result.get("returncode", 1)) == 0),
        "returncode": int(last_result.get("returncode", 1)) if last_result else 1,
        "stdout_log": str(last_result.get("stdout_log", "")) if last_result else "",
        "stderr_log": str(last_result.get("stderr_log", "")) if last_result else "",
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pov_json_path = Path(args.pov_json)
    output_dict = json.loads(pov_json_path.read_text(encoding="utf-8"))
    if not isinstance(output_dict, dict):
        raise ValueError("pov_json must contain a JSON object")

    video_id = _infer_video_id(output_dict, pov_json_path, args.video_id)
    include = tuple([x.strip() for x in str(args.include).split(",") if x.strip()])

    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    exported_events = export_bye_events_from_output_dict(output_dict, video_id=video_id, include=include, sort=True)
    events_jsonl_path = events_dir / "events_v1.jsonl"
    write_jsonl(exported_events, str(events_jsonl_path))

    run_pkg = build_run_package(
        out_dir=out_dir,
        video_id=video_id,
        events_jsonl_path=events_jsonl_path,
        video_path=args.video,
        extra_meta={"pov_json": str(pov_json_path), "include": list(include)},
    )

    snapshot: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_short(ROOT),
        "args": vars(args),
        "video_id": video_id,
        "input": {
            "pov_json": str(pov_json_path),
            "video": str(args.video) if args.video else None,
        },
        "outputs": {
            "events_jsonl": str(events_jsonl_path),
            "run_package_dir": str(run_pkg),
            "logs_dir": str(out_dir / "logs"),
            "report_dir": str(out_dir / "report"),
        },
        "counts": {"events_total": int(len(exported_events))},
        "bye": {
            "root": None,
            "found_entrypoints": {},
            "steps": [],
            "status": "not_resolved",
            "warnings": [],
        },
    }

    bye_root = resolve_bye_root(args.bye_root)
    if bye_root is None:
        msg = "BYE root not found. Export-only mode completed."
        print(f"WARN: {msg}")
        snapshot["bye"]["warnings"].append(msg)
        snapshot["bye"]["status"] = "missing_bye_root"
        snapshot["outputs_rel"] = {
            "events_jsonl": _to_rel(events_jsonl_path, out_dir),
            "run_package_dir": _to_rel(run_pkg, out_dir),
            "logs_dir": _to_rel(out_dir / "logs", out_dir),
            "report_dir": _to_rel(out_dir / "report", out_dir),
        }
        snapshot_path = _write_snapshot(snapshot, out_dir)
        print(f"video_id={video_id}")
        print(f"events_total={len(exported_events)}")
        print(f"saved_events={events_jsonl_path}")
        print(f"saved_snapshot={snapshot_path}")
        if args.strict:
            return 2
        return 0

    snapshot["bye"]["root"] = str(bye_root)
    entrypoints = find_bye_entrypoints(bye_root)
    snapshot["bye"]["found_entrypoints"] = dict(entrypoints)
    snapshot["bye"]["status"] = "resolved"

    plan = [
        ("lint", bool(args.skip_lint)),
        ("report", bool(args.skip_report)),
        ("regression", bool(args.skip_regression)),
    ]
    strict_failure = False

    for tool_name, skip in plan:
        if skip:
            snapshot["bye"]["steps"].append({"tool": tool_name, "skipped": True, "reason": "cli_skip"})
            continue

        rel = entrypoints.get(tool_name)
        if not rel:
            msg = f"BYE {tool_name} entrypoint not found."
            print(f"WARN: {msg}")
            snapshot["bye"]["warnings"].append(msg)
            snapshot["bye"]["steps"].append({"tool": tool_name, "skipped": True, "reason": "entrypoint_missing"})
            if args.strict:
                strict_failure = True
            continue

        result = _run_tool_with_attempts(
            tool_name=tool_name,
            script_rel=rel,
            bye_root=bye_root,
            run_package_dir=run_pkg,
            out_dir=out_dir,
        )
        snapshot["bye"]["steps"].append(result)
        print(f"{tool_name}_returncode={result.get('returncode')}")
        if not result.get("ok", False):
            msg = f"BYE {tool_name} failed."
            print(f"WARN: {msg}")
            snapshot["bye"]["warnings"].append(msg)
            if args.strict:
                strict_failure = True

    report_candidates = collect_report_candidates(out_dir / "report")
    if not report_candidates:
        report_candidates = collect_report_candidates(run_pkg)
    copied_reports = _copy_report_outputs(out_dir, report_candidates) if report_candidates else []
    snapshot["outputs"]["report_files"] = copied_reports
    snapshot["outputs_rel"] = {
        "events_jsonl": _to_rel(events_jsonl_path, out_dir),
        "run_package_dir": _to_rel(run_pkg, out_dir),
        "logs_dir": _to_rel(out_dir / "logs", out_dir),
        "report_dir": _to_rel(out_dir / "report", out_dir),
        "report_files": [_to_rel(x, out_dir) for x in copied_reports],
    }

    snapshot_path = _write_snapshot(snapshot, out_dir)
    print(f"video_id={video_id}")
    print(f"events_total={len(exported_events)}")
    print(f"bye_root={bye_root}")
    print(f"saved_events={events_jsonl_path}")
    print(f"saved_snapshot={snapshot_path}")
    print(f"report_files={len(copied_reports)}")

    if strict_failure:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
