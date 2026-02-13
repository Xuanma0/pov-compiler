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

from pov_compiler.integrations.bye.budget_filter import Budget, apply_budget
from pov_compiler.integrations.bye.entrypoints import EntryPointResolver
from pov_compiler.integrations.bye.exporter import export_bye_events_from_output_dict, write_jsonl
from pov_compiler.integrations.bye.metrics import parse_bye_report, save_bye_metrics
from pov_compiler.integrations.bye.run_package import (
    build_run_package,
    collect_report_candidates,
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
    parser.add_argument(
        "--video-mode",
        choices=["copy", "link"],
        default="copy",
        help="How to place video in run_package when --video is provided",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for run package/logs/snapshot")
    parser.add_argument("--bye_root", default=None, help="External BYE repo root")
    parser.add_argument("--bye-lint", default=None, help="Override path for BYE lint script (relative to BYE root or absolute)")
    parser.add_argument(
        "--bye-report", default=None, help="Override path for BYE report script (relative to BYE root or absolute)"
    )
    parser.add_argument(
        "--bye-regression",
        default=None,
        help="Override path for BYE regression script (relative to BYE root or absolute)",
    )
    parser.add_argument("--skip_lint", action="store_true", help="Skip BYE lint step")
    parser.add_argument("--skip_report", action="store_true", help="Skip BYE report step")
    parser.add_argument("--skip_regression", action="store_true", help="Skip BYE regression step")
    parser.add_argument(
        "--include",
        default="events_v1,highlights,tokens,decisions",
        help="Sections to export to BYE events jsonl",
    )
    parser.add_argument(
        "--budget-mode",
        choices=["none", "filter"],
        default="none",
        help="Budget filter mode for exported events. Any budget value auto-enables filter mode.",
    )
    parser.add_argument("--budget-max-total-s", type=float, default=None, help="Max kept context duration (seconds)")
    parser.add_argument("--budget-max-tokens", type=int, default=None, help="Max kept pov.token events")
    parser.add_argument("--budget-max-decisions", type=int, default=None, help="Max kept pov.decision events")
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
    script_path: Path,
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
            module_or_script_path=str(script_path),
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
        "script_path": str(script_path),
        "attempts": attempts_meta,
        "ok": bool(last_result and int(last_result.get("returncode", 1)) == 0),
        "returncode": int(last_result.get("returncode", 1)) if last_result else 1,
        "stdout_log": str(last_result.get("stdout_log", "")) if last_result else "",
        "stderr_log": str(last_result.get("stderr_log", "")) if last_result else "",
    }


def _build_metrics_payload(status: str, report_dir: Path) -> dict[str, Any]:
    if status == "ok":
        return parse_bye_report(report_dir)
    if status == "skipped":
        return {"status": "skipped", "report_path": str(report_dir / "report.json"), "summary_keys": [], "numeric_metrics": {}}
    if status == "failed":
        return {"status": "failed", "report_path": str(report_dir / "report.json"), "summary_keys": [], "numeric_metrics": {}}
    return {
        "status": "missing_report",
        "report_path": str(report_dir / "report.json"),
        "summary_keys": [],
        "numeric_metrics": {},
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
    budget = Budget(
        max_total_s=args.budget_max_total_s,
        max_tokens=args.budget_max_tokens,
        max_decisions=args.budget_max_decisions,
    )
    budget_requested = any(x is not None for x in (args.budget_max_total_s, args.budget_max_tokens, args.budget_max_decisions))
    budget_mode = "filter" if args.budget_mode == "filter" or budget_requested else "none"

    events_dir = out_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    exported_events = export_bye_events_from_output_dict(output_dict, video_id=video_id, include=include, sort=True)
    budget_stats: dict[str, Any] = {
        "mode": budget_mode,
        "applied": False,
        "before_total": len(exported_events),
        "after_total": len(exported_events),
    }
    if budget_mode == "filter":
        exported_events, stats = apply_budget(exported_events, budget, keep_base_events=True)
        budget_stats = {"mode": "filter", "applied": True, **stats}
    events_jsonl_path = events_dir / "events_v1.jsonl"
    write_jsonl(exported_events, str(events_jsonl_path))

    run_pkg = build_run_package(
        out_dir=out_dir,
        video_id=video_id,
        events_jsonl_path=events_jsonl_path,
        video_path=args.video,
        video_mode=str(args.video_mode),
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
        "budget": budget_stats,
        "bye": {
            "root": None,
            "entrypoints_resolved": {"lint": None, "report": None, "regression": None},
            "entrypoints_method": {"lint": "missing", "report": "missing", "regression": "missing"},
            "steps": [],
            "status": "not_resolved",
            "warnings": [],
            "notes": [],
        },
    }

    # Minimal pre-BYE self checks.
    if not events_jsonl_path.exists() or events_jsonl_path.stat().st_size <= 0:
        msg = "events/events_v1.jsonl missing or empty"
        snapshot["bye"]["warnings"].append(msg)
        metrics_payload = _build_metrics_payload("failed", out_dir / "report")
        metrics_payload["status"] = "failed"
        metrics_payload["reason"] = msg
        metrics_json, metrics_csv = save_bye_metrics(metrics_payload, out_dir)
        snapshot["outputs"]["bye_metrics_json"] = str(metrics_json)
        snapshot["outputs"]["bye_metrics_csv"] = str(metrics_csv)
        snapshot["outputs_rel"] = {
            "events_jsonl": _to_rel(events_jsonl_path, out_dir),
            "run_package_dir": _to_rel(run_pkg, out_dir),
            "bye_metrics_json": _to_rel(metrics_json, out_dir),
            "bye_metrics_csv": _to_rel(metrics_csv, out_dir),
        }
        snapshot_path = _write_snapshot(snapshot, out_dir)
        print(f"WARN: {msg}")
        print(f"saved_snapshot={snapshot_path}")
        return 4
    metadata_path = run_pkg / "metadata.json"
    if not metadata_path.exists():
        msg = "run_package/metadata.json missing"
        snapshot["bye"]["warnings"].append(msg)
        metrics_payload = _build_metrics_payload("failed", out_dir / "report")
        metrics_payload["status"] = "failed"
        metrics_payload["reason"] = msg
        metrics_json, metrics_csv = save_bye_metrics(metrics_payload, out_dir)
        snapshot["outputs"]["bye_metrics_json"] = str(metrics_json)
        snapshot["outputs"]["bye_metrics_csv"] = str(metrics_csv)
        snapshot["outputs_rel"] = {
            "events_jsonl": _to_rel(events_jsonl_path, out_dir),
            "run_package_dir": _to_rel(run_pkg, out_dir),
            "bye_metrics_json": _to_rel(metrics_json, out_dir),
            "bye_metrics_csv": _to_rel(metrics_csv, out_dir),
        }
        snapshot_path = _write_snapshot(snapshot, out_dir)
        print(f"WARN: {msg}")
        print(f"saved_snapshot={snapshot_path}")
        return 4

    bye_root = resolve_bye_root(args.bye_root)
    if bye_root is None:
        msg = "BYE root not found. Export-only mode completed."
        print(f"WARN: {msg}")
        snapshot["bye"]["warnings"].append(msg)
        snapshot["bye"]["status"] = "missing_bye_root"
        metrics_payload = _build_metrics_payload("missing", out_dir / "report")
        metrics_payload["reason"] = "missing_bye_root"
        metrics_json, metrics_csv = save_bye_metrics(metrics_payload, out_dir)
        snapshot["outputs"]["bye_metrics_json"] = str(metrics_json)
        snapshot["outputs"]["bye_metrics_csv"] = str(metrics_csv)
        snapshot["outputs_rel"] = {
            "events_jsonl": _to_rel(events_jsonl_path, out_dir),
            "run_package_dir": _to_rel(run_pkg, out_dir),
            "logs_dir": _to_rel(out_dir / "logs", out_dir),
            "report_dir": _to_rel(out_dir / "report", out_dir),
            "bye_metrics_json": _to_rel(metrics_json, out_dir),
            "bye_metrics_csv": _to_rel(metrics_csv, out_dir),
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
    resolver = EntryPointResolver(
        bye_root=bye_root,
        overrides={"lint": args.bye_lint, "report": args.bye_report, "regression": args.bye_regression},
    )
    resolution = resolver.resolve()
    resolved_dict = resolution.as_dict(bye_root=bye_root)
    snapshot["bye"]["entrypoints_resolved"] = {
        "lint": resolved_dict.get("lint"),
        "report": resolved_dict.get("report"),
        "regression": resolved_dict.get("regression"),
    }
    snapshot["bye"]["entrypoints_method"] = dict(resolution.methods)
    snapshot["bye"]["notes"] = list(resolution.notes)
    snapshot["bye"]["status"] = "resolved"

    strict_failure = False
    if args.strict and not args.skip_report and resolution.report is None:
        strict_failure = True
        snapshot["bye"]["warnings"].append("strict: missing report entrypoint")

    plan = [
        ("lint", bool(args.skip_lint), resolution.lint),
        ("report", bool(args.skip_report), resolution.report),
        ("regression", bool(args.skip_regression), resolution.regression),
    ]
    report_step_status = "skipped" if args.skip_report else "missing"
    for tool_name, skip, script_path in plan:
        if skip:
            snapshot["bye"]["steps"].append({"tool": tool_name, "skipped": True, "reason": "cli_skip"})
            continue
        if script_path is None:
            msg = f"BYE {tool_name} entrypoint not found."
            print(f"WARN: {msg}")
            snapshot["bye"]["warnings"].append(msg)
            snapshot["bye"]["steps"].append({"tool": tool_name, "skipped": True, "reason": "entrypoint_missing"})
            if tool_name == "report":
                report_step_status = "missing"
            if args.strict and tool_name == "report":
                strict_failure = True
            continue

        result = _run_tool_with_attempts(
            tool_name=tool_name,
            script_path=script_path,
            bye_root=bye_root,
            run_package_dir=run_pkg,
            out_dir=out_dir,
        )
        snapshot["bye"]["steps"].append(result)
        print(f"{tool_name}_returncode={result.get('returncode')}")
        if tool_name == "report":
            report_step_status = "ok" if result.get("ok", False) else "failed"
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

    metrics_payload = _build_metrics_payload(report_step_status, out_dir / "report")
    if report_step_status != "ok":
        metrics_payload["reason"] = report_step_status
    metrics_json, metrics_csv = save_bye_metrics(metrics_payload, out_dir)
    snapshot["outputs"]["bye_metrics_json"] = str(metrics_json)
    snapshot["outputs"]["bye_metrics_csv"] = str(metrics_csv)
    snapshot["outputs_rel"] = {
        "events_jsonl": _to_rel(events_jsonl_path, out_dir),
        "run_package_dir": _to_rel(run_pkg, out_dir),
        "logs_dir": _to_rel(out_dir / "logs", out_dir),
        "report_dir": _to_rel(out_dir / "report", out_dir),
        "report_files": [_to_rel(x, out_dir) for x in copied_reports],
        "bye_metrics_json": _to_rel(metrics_json, out_dir),
        "bye_metrics_csv": _to_rel(metrics_csv, out_dir),
    }

    snapshot_path = _write_snapshot(snapshot, out_dir)
    print(f"video_id={video_id}")
    print(f"events_total={len(exported_events)}")
    if budget_stats.get("applied"):
        print(f"budget_kept_duration_s={budget_stats.get('kept_duration_s', 0.0)}")
        print(f"budget_compression_ratio={budget_stats.get('compression_ratio', 1.0)}")
    print(f"bye_root={bye_root}")
    print(f"saved_events={events_jsonl_path}")
    print(f"saved_snapshot={snapshot_path}")
    print(f"report_files={len(copied_reports)}")
    print(f"bye_metrics_json={metrics_json}")
    print(f"bye_metrics_csv={metrics_csv}")

    if strict_failure:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
