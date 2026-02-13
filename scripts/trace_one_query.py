from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.trace import trace_query


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _render_markdown(trace: dict[str, Any]) -> str:
    plan = dict(trace.get("plan", {}))
    ctrace = dict(trace.get("constraint_trace", {}))
    hits = list(trace.get("hits", []))
    lines: list[str] = []
    lines.append("# Query Trace Report")
    lines.append("")
    lines.append(f"- video_id: {trace.get('video_id', '')}")
    lines.append(f"- query: `{trace.get('query', '')}`")
    lines.append(f"- chosen_plan_intent: {plan.get('intent', '')}")
    lines.append(f"- rerank_cfg_hash: {trace.get('rerank_cfg_hash', '')}")
    lines.append(f"- top1_kind: {trace.get('top1_kind', '')}")
    lines.append("")
    lines.append("## Plan")
    lines.append("")
    lines.append(f"- constraints: `{json.dumps(plan.get('constraints', {}), ensure_ascii=False, sort_keys=True)}`")
    lines.append(f"- candidates_count: {len(plan.get('candidates', []))}")
    for item in plan.get("candidates", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- candidate: priority={int(item.get('priority', 0))} "
            f"query=`{item.get('query', '')}` reason=`{item.get('reason', '')}`"
        )
    lines.append("")
    lines.append("## Constraints")
    lines.append("")
    lines.append(f"- applied_constraints: {ctrace.get('applied_constraints', [])}")
    lines.append(f"- constraints_relaxed: {ctrace.get('constraints_relaxed', [])}")
    lines.append(f"- filtered_hits_before: {int(ctrace.get('filtered_hits_before', 0))}")
    lines.append(f"- filtered_hits_after: {int(ctrace.get('filtered_hits_after', 0))}")
    lines.append(f"- used_fallback: {str(bool(ctrace.get('used_fallback', False))).lower()}")
    lines.append("")
    lines.append("## Top Hits")
    lines.append("")
    lines.append(
        "| rank | kind | id | span | score | distractor_flag | source_query | "
        "score_breakdown | linked_events_v1 |"
    )
    lines.append("|---:|---|---|---|---:|---:|---|---|---|")
    for hit in hits:
        span = f"{float(hit.get('t0', 0.0)):.3f}-{float(hit.get('t1', 0.0)):.3f}"
        sb = hit.get("score_breakdown", {})
        lines.append(
            f"| {int(hit.get('rank', 0))} | {hit.get('kind', '')} | {hit.get('id', '')} | {span} | "
            f"{float(hit.get('score', 0.0)):.4f} | {int(bool(hit.get('distractor_flag', False)))} | "
            f"`{hit.get('source_query', '')}` | `{json.dumps(sb, ensure_ascii=False, sort_keys=True)}` | "
            f"`{json.dumps(hit.get('linked_events_v1', []), ensure_ascii=False)}` |"
        )
    lines.append("")
    lines.append("## Evidence Spans")
    lines.append("")
    for hit in hits:
        lines.append(f"### rank={int(hit.get('rank', 0))} {hit.get('kind', '')}:{hit.get('id', '')}")
        evidence = hit.get("evidence_spans", [])
        if not evidence:
            lines.append("- evidence_spans: []")
            lines.append("")
            continue
        lines.append("| event_v1_id | evidence_id | evidence_type | span | conf |")
        lines.append("|---|---|---|---|---:|")
        for item in evidence:
            span = f"{float(item.get('t0', 0.0)):.3f}-{float(item.get('t1', 0.0)):.3f}"
            lines.append(
                f"| {item.get('event_v1_id', '')} | {item.get('evidence_id', '')} | "
                f"{item.get('evidence_type', '')} | {span} | {float(item.get('conf', 0.0)):.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace one query: plan -> constraints -> rerank -> evidence")
    parser.add_argument("--json", required=True, help="Output JSON path")
    parser.add_argument("--query", required=True, help="Natural language or structured query")
    parser.add_argument("--index", default=None, help="Index prefix")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Config path")
    parser.add_argument("--top-k", type=int, default=6, help="Top-k hits")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    hard_cfg = dict(cfg.get("hard_constraints", {}))
    rerank_cfg = cfg.get("reranker", {})

    trace = trace_query(
        output_json=Path(args.json),
        query=str(args.query),
        index_prefix=Path(args.index) if args.index else None,
        retrieval_config=retrieval_cfg,
        hard_constraints_cfg=hard_cfg,
        rerank_cfg=rerank_cfg,
        top_k=int(args.top_k),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = out_dir / "trace_report.md"
    trace_json = out_dir / "trace.json"
    report_md.write_text(_render_markdown(trace), encoding="utf-8")
    trace_json.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"video_id={trace.get('video_id', '')}")
    print(f"query={trace.get('query', '')}")
    print(f"chosen_plan_intent={trace.get('plan', {}).get('intent', '')}")
    print(f"applied_constraints={trace.get('constraint_trace', {}).get('applied_constraints', [])}")
    print(f"filtered_hits_before={trace.get('constraint_trace', {}).get('filtered_hits_before', 0)}")
    print(f"filtered_hits_after={trace.get('constraint_trace', {}).get('filtered_hits_after', 0)}")
    print(f"top1_kind={trace.get('top1_kind', '')}")
    print(f"saved_report={report_md}")
    print(f"saved_trace={trace_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

