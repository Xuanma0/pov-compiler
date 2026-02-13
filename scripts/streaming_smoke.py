from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

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
    lines.append(f"- retrieval_latency_p50_ms: {float(summary.get('retrieval_latency_p50_ms', 0.0)):.3f}")
    lines.append(f"- retrieval_latency_p95_ms: {float(summary.get('retrieval_latency_p95_ms', 0.0)):.3f}")
    lines.append(f"- e2e_latency_p50_ms: {float(summary.get('e2e_latency_p50_ms', 0.0)):.3f}")
    lines.append(f"- e2e_latency_p95_ms: {float(summary.get('e2e_latency_p95_ms', 0.0)):.3f}")
    lines.append(f"- e2e_with_io_p50_ms: {float(summary.get('e2e_with_io_p50_ms', 0.0)):.3f}")
    lines.append(f"- e2e_with_io_p95_ms: {float(summary.get('e2e_with_io_p95_ms', 0.0)):.3f}")
    lines.append(f"- write_ms_total: {float(summary.get('write_ms_total', 0.0)):.3f}")
    lines.append(f"- throughput_qps_mean: {float(summary.get('throughput_qps_mean', 0.0)):.3f}")
    lines.append(
        "- e2e_includes: step slicing + events_v1 IR update + index update + retrieval execution + "
        "serialization/write amortized"
    )
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append(
        "| step | end_t | index_size | events_v1_added | events_v1_indexed | "
        "retrieval_latency_p50_ms | retrieval_latency_p95_ms | e2e_ms | throughput_qps |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in step_rows:
        lines.append(
            f"| {int(row.get('step_idx', 0))} | {float(row.get('end_t', 0.0)):.3f} | "
            f"{int(row.get('index_size', 0))} | {int(row.get('events_v1_added', 0))} | "
            f"{int(row.get('events_v1_indexed', 0))} | {float(row.get('retrieval_latency_p50_ms', 0.0)):.3f} | "
            f"{float(row.get('retrieval_latency_p95_ms', 0.0)):.3f} | {float(row.get('e2e_ms', 0.0)):.3f} | "
            f"{float(row.get('throughput_qps', 0.0)):.3f} |"
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

    write_started = time.perf_counter()
    _write_csv(summary_csv, [dict(payload.get("summary", {}))])
    _write_csv(steps_csv, list(payload.get("step_rows", [])))
    _write_csv(queries_csv, list(payload.get("query_rows", [])))
    progressive_lines = [json.dumps(dict(row), ensure_ascii=False) for row in list(payload.get("progressive_rows", []))]
    progressive_jsonl.write_text("\n".join(progressive_lines), encoding="utf-8")
    write_ms_total = float((time.perf_counter() - write_started) * 1000.0)

    # Amortize write overhead into step e2e to expose end-to-end latency with I/O.
    step_rows = list(payload.get("step_rows", []))
    if step_rows:
        write_per_step = float(write_ms_total / len(step_rows))
        for row in step_rows:
            row["e2e_ms_with_io"] = float(row.get("e2e_ms", 0.0)) + write_per_step
        payload["step_rows"] = step_rows
        summary = dict(payload.get("summary", {}))
        summary["write_ms_total"] = write_ms_total
        summary["e2e_with_io_p50_ms"] = float(
            np.percentile([float(r["e2e_ms_with_io"]) for r in step_rows], 50.0)
        )
        summary["e2e_with_io_p95_ms"] = float(
            np.percentile([float(r["e2e_ms_with_io"]) for r in step_rows], 95.0)
        )
        payload["summary"] = summary
        _write_csv(summary_csv, [summary])
        _write_csv(steps_csv, step_rows)
    _write_report(report_md, payload)

    summary = dict(payload.get("summary", {}))
    print(f"video_id={summary.get('video_id', '')}")
    print(f"steps={int(summary.get('steps', 0))}")
    print(f"queries_total={int(summary.get('queries_total', 0))}")
    print(f"events_v1_indexed={int(summary.get('events_v1_indexed', 0))}")
    print(f"retrieval_latency_p50_ms={float(summary.get('retrieval_latency_p50_ms', 0.0)):.3f}")
    print(f"retrieval_latency_p95_ms={float(summary.get('retrieval_latency_p95_ms', 0.0)):.3f}")
    print(f"e2e_latency_p50_ms={float(summary.get('e2e_latency_p50_ms', 0.0)):.3f}")
    print(f"e2e_latency_p95_ms={float(summary.get('e2e_latency_p95_ms', 0.0)):.3f}")
    print(f"e2e_with_io_p50_ms={float(summary.get('e2e_with_io_p50_ms', 0.0)):.3f}")
    print(f"e2e_with_io_p95_ms={float(summary.get('e2e_with_io_p95_ms', 0.0)):.3f}")
    print(f"write_ms_total={float(summary.get('write_ms_total', 0.0)):.3f}")
    print(f"throughput_qps_mean={float(summary.get('throughput_qps_mean', 0.0)):.3f}")
    print(f"saved_summary={summary_csv}")
    print(f"saved_steps={steps_csv}")
    print(f"saved_queries={queries_csv}")
    print(f"saved_progressive={progressive_jsonl}")
    print(f"saved_report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
