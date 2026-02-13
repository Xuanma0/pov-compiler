from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.runner import StreamingConfig, run_streaming


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = dict(payload.get("summary", {}))
    step_rows = list(payload.get("step_rows", []))
    lines: list[str] = []
    lines.append("# Streaming Smoke Report")
    lines.append("")
    lines.append(f"- video_id: {summary.get('video_id', '')}")
    lines.append(f"- duration_s: {float(summary.get('duration_s', 0.0)):.3f}")
    lines.append(f"- steps: {int(summary.get('steps', 0))}")
    lines.append(f"- queries_total: {int(summary.get('queries_total', 0))}")
    lines.append(f"- events_v1_indexed: {int(summary.get('events_v1_indexed', 0))}")
    lines.append(f"- latency_p50_ms: {float(summary.get('latency_p50_ms', 0.0)):.3f}")
    lines.append(f"- latency_p95_ms: {float(summary.get('latency_p95_ms', 0.0)):.3f}")
    lines.append(f"- throughput_qps_mean: {float(summary.get('throughput_qps_mean', 0.0)):.3f}")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append("| step | end_t | events_v1_indexed | latency_p50_ms | latency_p95_ms | throughput_qps |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in step_rows:
        lines.append(
            f"| {int(row.get('step_idx', 0))} | {float(row.get('end_t', 0.0)):.3f} | "
            f"{int(row.get('events_v1_indexed', 0))} | {float(row.get('latency_p50_ms', 0.0)):.3f} | "
            f"{float(row.get('latency_p95_ms', 0.0)):.3f} | {float(row.get('throughput_qps', 0.0)):.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming runner smoke: incremental index + online retrieve")
    parser.add_argument("--json", required=True, help="Offline output JSON path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Config path")
    parser.add_argument("--step-s", type=float, default=5.0, help="Streaming step window in seconds")
    parser.add_argument("--top-k", type=int, default=6, help="Query top_k")
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Query text (repeatable). Example: --query \"anchor=turn_head top_k=6\"",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    streaming_cfg = dict(cfg.get("streaming", {}))
    queries = list(args.query)
    if not queries:
        queries = list(streaming_cfg.get("queries", []))

    payload = run_streaming(
        Path(args.json),
        config=StreamingConfig(
            step_s=float(args.step_s),
            top_k=int(args.top_k),
            queries=queries,
            retrieval_config=retrieval_cfg,
        ),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    steps_csv = out_dir / "streaming_steps.csv"
    queries_csv = out_dir / "streaming_queries.csv"
    progressive_jsonl = out_dir / "progressive_ir.jsonl"
    report_md = out_dir / "report.md"

    _write_csv(summary_csv, [dict(payload.get("summary", {}))])
    _write_csv(steps_csv, list(payload.get("step_rows", [])))
    _write_csv(queries_csv, list(payload.get("query_rows", [])))
    progressive_lines = [
        json.dumps(dict(row), ensure_ascii=False) for row in list(payload.get("progressive_rows", []))
    ]
    progressive_jsonl.write_text("\n".join(progressive_lines), encoding="utf-8")
    _write_report(report_md, payload)

    summary = dict(payload.get("summary", {}))
    print(f"video_id={summary.get('video_id', '')}")
    print(f"steps={int(summary.get('steps', 0))}")
    print(f"queries_total={int(summary.get('queries_total', 0))}")
    print(f"events_v1_indexed={int(summary.get('events_v1_indexed', 0))}")
    print(f"latency_p50_ms={float(summary.get('latency_p50_ms', 0.0)):.3f}")
    print(f"latency_p95_ms={float(summary.get('latency_p95_ms', 0.0)):.3f}")
    print(f"throughput_qps_mean={float(summary.get('throughput_qps_mean', 0.0)):.3f}")
    print(f"saved_summary={summary_csv}")
    print(f"saved_steps={steps_csv}")
    print(f"saved_queries={queries_csv}")
    print(f"saved_progressive={progressive_jsonl}")
    print(f"saved_report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

