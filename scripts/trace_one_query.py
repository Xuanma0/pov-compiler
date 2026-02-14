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
    chain = trace.get("chain", {})
    is_chain = bool(isinstance(chain, dict) and chain.get("is_chain", False))
    lines: list[str] = []
    lines.append("# Query Trace Report")
    lines.append("")
    lines.append(f"- video_id: {trace.get('video_id', '')}")
    lines.append(f"- query: `{trace.get('query', '')}`")
    lines.append(f"- is_chain: {str(is_chain).lower()}")
    if is_chain:
        lines.append(f"- chain_steps: 2")
        lines.append(f"- chain_rel: {chain.get('chain_rel', 'after')}")
        lines.append(f"- chain_window_s: {float(chain.get('window_s', 30.0)):.3f}")
        lines.append(f"- chain_top1_only: {str(bool(chain.get('top1_only', True))).lower()}")
    lines.append(f"- chosen_plan_intent: {plan.get('intent', '')}")
    lines.append(f"- rerank_cfg_hash: {trace.get('rerank_cfg_hash', '')}")
    lines.append(f"- top1_kind: {trace.get('top1_kind', '')}")
    lines.append(f"- parsed_constraints: `{json.dumps(plan.get('constraints', {}), ensure_ascii=False, sort_keys=True)}`")
    lines.append(f"- enable_constraints: {str(bool(ctrace.get('enable_constraints', True))).lower()}")
    lines.append(f"- applied_constraints: {ctrace.get('applied_constraints', [])}")
    lines.append(f"- filtered_hits_before: {int(ctrace.get('filtered_hits_before', 0))}")
    lines.append(f"- filtered_hits_after: {int(ctrace.get('filtered_hits_after', 0))}")
    lines.append(f"- relax_steps: {ctrace.get('constraints_relaxed', [])}")
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
    steps = ctrace.get("constraint_steps", [])
    if isinstance(steps, list) and steps:
        lines.append("")
        lines.append("| constraint | before | after | satisfied | details |")
        lines.append("|---|---:|---:|---:|---|")
        for step in steps:
            if not isinstance(step, dict):
                continue
            details = json.dumps(step.get("details", {}), ensure_ascii=False, sort_keys=True)
            lines.append(
                f"| {step.get('name', '')} | {int(step.get('before', 0))} | {int(step.get('after', 0))} | "
                f"{int(bool(step.get('satisfied', False)))} | `{details}` |"
            )
    lines.append("")
    if is_chain:
        lines.append("## Chain Step 1")
        lines.append("")
        step1 = chain.get("step1", {}) if isinstance(chain, dict) else {}
        lines.append(f"- query: `{step1.get('query', '')}`")
        lines.append(f"- parsed_constraints: `{json.dumps(step1.get('parsed_constraints', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append(f"- applied_constraints: {step1.get('applied_constraints', [])}")
        lines.append(
            f"- filtered_hits_before/after: {int(step1.get('filtered_hits_before', 0))}->{int(step1.get('filtered_hits_after', 0))}"
        )
        lines.append(f"- top1_kind: {step1.get('top1_kind', '')}")
        lines.append("")
        step1_hits = step1.get("topk_hits", [])
        if isinstance(step1_hits, list) and step1_hits:
            lines.append("| rank | kind | id | span | score |")
            lines.append("|---:|---|---|---|---:|")
            for item in step1_hits[:10]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"| {int(item.get('rank', 0))} | {item.get('kind', '')} | {item.get('id', '')} | "
                    f"{float(item.get('t0', 0.0)):.3f}-{float(item.get('t1', 0.0)):.3f} | {float(item.get('score', 0.0)):.4f} |"
                )
        lines.append("")
        lines.append("## Derived Constraints")
        lines.append("")
        derived = chain.get("derived_constraints", {}) if isinstance(chain, dict) else {}
        lines.append(f"- raw: `{json.dumps(derived, ensure_ascii=False, sort_keys=True)}`")
        d_time = derived.get("time", {}) if isinstance(derived, dict) else {}
        d_place = derived.get("place", {}) if isinstance(derived, dict) else {}
        d_object = derived.get("object", {}) if isinstance(derived, dict) else {}
        lines.append("")
        lines.append("| type | value | mode | source | enabled |")
        lines.append("|---|---|---|---|---:|")
        lines.append(
            f"| time | {json.dumps({'t_min_s': d_time.get('t_min_s'), 't_max_s': d_time.get('t_max_s')}, ensure_ascii=False)} | "
            f"{d_time.get('mode', '')} | {d_time.get('source', '')} | {int(bool(d_time.get('enabled', False)))} |"
        )
        lines.append(
            f"| place | {d_place.get('value', '')} | {d_place.get('mode', '')} | {d_place.get('source', '')} | "
            f"{int(bool(d_place.get('enabled', False)))} |"
        )
        lines.append(
            f"| object | {d_object.get('value', '')} | {d_object.get('mode', '')} | {d_object.get('source', '')} | "
            f"{int(bool(d_object.get('enabled', False)))} |"
        )
        lines.append("")
        lines.append("## Chain Step 2")
        lines.append("")
        step2 = chain.get("step2", {}) if isinstance(chain, dict) else {}
        lines.append(f"- query: `{step2.get('query', '')}`")
        lines.append(f"- query_derived: `{step2.get('query_derived', '')}`")
        lines.append(f"- parsed_constraints: `{json.dumps(step2.get('parsed_constraints', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append(f"- applied_constraints: {step2.get('applied_constraints', [])}")
        lines.append(
            f"- filtered_hits_before/after: {int(step2.get('filtered_hits_before', 0))}->{int(step2.get('filtered_hits_after', 0))}"
        )
        lines.append(f"- top1_kind: {step2.get('top1_kind', '')}")
        lines.append("")

    lines.append("## Object Memory V0")
    lines.append("")
    obj_rows = trace.get("object_memory_summary", [])
    if isinstance(obj_rows, list) and obj_rows:
        lines.append("| object_name | last_seen_t_ms | last_contact_t_ms | last_place_id | score |")
        lines.append("|---|---:|---:|---|---:|")
        for item in obj_rows:
            if not isinstance(item, dict):
                continue
            lc = item.get("last_contact_t_ms", "")
            lines.append(
                f"| {item.get('object_name', '')} | {int(item.get('last_seen_t_ms', 0))} | "
                f"{'' if lc is None else int(lc)} | {item.get('last_place_id', '')} | "
                f"{float(item.get('score', 0.0)):.4f} |"
            )
    else:
        lines.append("- object_memory_summary: []")
    lines.append("")

    lines.append("## Place Segment")
    lines.append("")
    place_dist = trace.get("place_segment_distribution", [])
    if isinstance(place_dist, list) and place_dist:
        lines.append("| place_segment_id | count |")
        lines.append("|---|---:|")
        for item in place_dist:
            if not isinstance(item, dict):
                continue
            lines.append(f"| {item.get('place_segment_id', '')} | {int(item.get('count', 0))} |")
    else:
        lines.append("- place_segment_distribution: []")
    lines.append("")

    lines.append("## Interaction TopK")
    lines.append("")
    interaction_rows = trace.get("interaction_topk", [])
    if isinstance(interaction_rows, list) and interaction_rows:
        lines.append("| rank | kind | id | interaction_score | interaction_primary_object | place_segment_id |")
        lines.append("|---:|---|---|---:|---|---|")
        for item in interaction_rows:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {int(item.get('rank', 0))} | {item.get('kind', '')} | {item.get('id', '')} | "
                f"{float(item.get('interaction_score', 0.0)):.4f} | {item.get('interaction_primary_object', '')} | "
                f"{item.get('place_segment_id', '')} |"
            )
    else:
        lines.append("- interaction_topk: []")
    lines.append("")

    lines.append("## Repo selection (query-aware)")
    lines.append("")
    repo_selection = trace.get("repo_selection", {})
    if isinstance(repo_selection, dict) and repo_selection.get("enabled"):
        rtrace = dict(repo_selection.get("trace", {}))
        sel = list(repo_selection.get("selected_chunks", []))
        lines.append(f"- policy_name: `{rtrace.get('selection_trace', {}).get('policy_name', '')}`")
        lines.append(f"- policy_hash: `{rtrace.get('selection_trace', {}).get('policy_hash', '')}`")
        lines.append(f"- selected_chunks: {len(sel)}")
        lines.append(f"- selected_breakdown_by_level: `{json.dumps(rtrace.get('selection_trace', {}).get('selected_breakdown_by_level', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append(f"- dropped_topN: `{json.dumps(rtrace.get('selection_trace', {}).get('dropped_topN', []), ensure_ascii=False)}`")
        if sel:
            lines.append("")
            lines.append("| chunk_id | level | span | preview |")
            lines.append("|---|---|---|---|")
            for item in sel[:20]:
                if not isinstance(item, dict):
                    continue
                preview = str(item.get("text", "")).replace("|", " ").strip()
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                lines.append(
                    f"| {item.get('id', item.get('chunk_id', ''))} | {item.get('level', item.get('scale', ''))} | "
                    f"{float(item.get('t0', 0.0)):.3f}-{float(item.get('t1', 0.0)):.3f} | {preview} |"
                )
    else:
        lines.append("- repo_selection: disabled")
    lines.append("")

    lines.append("## Top Hits")
    lines.append("")
    lines.append(
        "| rank | kind | id | span | score | semantic | decision_align | final | "
        "distractor_flag | source_query | score_breakdown | linked_events_v1 |"
    )
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---|---|---|")
    for hit in hits:
        span = f"{float(hit.get('t0', 0.0)):.3f}-{float(hit.get('t1', 0.0)):.3f}"
        sb = hit.get("score_breakdown", {})
        semantic = float(sb.get("semantic_score", 0.0))
        decision_align = float(sb.get("decision_align_score", 0.0))
        final = float(sb.get("total", hit.get("score", 0.0)))
        lines.append(
            f"| {int(hit.get('rank', 0))} | {hit.get('kind', '')} | {hit.get('id', '')} | {span} | "
            f"{float(hit.get('score', 0.0)):.4f} | {semantic:.4f} | {decision_align:.4f} | {final:.4f} | "
            f"{int(bool(hit.get('distractor_flag', False)))} | "
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable-constraints", dest="enable_constraints", action="store_true")
    group.add_argument("--no-enable-constraints", dest="enable_constraints", action="store_false")
    parser.set_defaults(enable_constraints=True)
    parser.add_argument("--use-repo", action="store_true", help="Enable repo-only context trace with query-aware selection")
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
        enable_constraints=bool(args.enable_constraints),
        use_repo=bool(args.use_repo),
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
    chain = trace.get("chain", {})
    is_chain = bool(isinstance(chain, dict) and chain.get("is_chain", False))
    if is_chain:
        print(f"is_chain=true chain_steps=2 chain_rel={chain.get('chain_rel', 'after')}")
        print(f"derived_constraints={chain.get('derived_constraints', {})}")
        step1 = chain.get("step1", {}) if isinstance(chain, dict) else {}
        step2 = chain.get("step2", {}) if isinstance(chain, dict) else {}
        print(
            "step1_filtered_hits_before="
            f"{int(step1.get('filtered_hits_before', 0))} step1_filtered_hits_after={int(step1.get('filtered_hits_after', 0))}"
        )
        print(
            "step2_filtered_hits_before="
            f"{int(step2.get('filtered_hits_before', 0))} step2_filtered_hits_after={int(step2.get('filtered_hits_after', 0))}"
        )
        print(f"step2_applied_constraints={step2.get('applied_constraints', [])}")
        print(f"step1_top1_kind={step1.get('top1_kind', '')} step2_top1_kind={step2.get('top1_kind', '')}")
    print(f"parsed_constraints={trace.get('plan', {}).get('constraints', {})}")
    print(f"applied_constraints={trace.get('constraint_trace', {}).get('applied_constraints', [])}")
    print(f"filtered_hits_before={trace.get('constraint_trace', {}).get('filtered_hits_before', 0)}")
    print(f"filtered_hits_after={trace.get('constraint_trace', {}).get('filtered_hits_after', 0)}")
    print(f"relax_steps={trace.get('constraint_trace', {}).get('constraints_relaxed', [])}")
    step_rows = trace.get("constraint_trace", {}).get("constraint_steps", [])
    if isinstance(step_rows, list) and step_rows:
        summary_parts: list[str] = []
        for step in step_rows:
            if not isinstance(step, dict):
                continue
            summary_parts.append(
                f"{step.get('name', '')} {int(step.get('before', 0))}->{int(step.get('after', 0))}"
            )
        if summary_parts:
            print(f"constraint_steps_summary={'; '.join(summary_parts)}")
    print(f"top1_kind={trace.get('top1_kind', '')}")
    repo_sel = trace.get("repo_selection", {})
    if isinstance(repo_sel, dict) and repo_sel.get("enabled"):
        rtrace = repo_sel.get("trace", {})
        if isinstance(rtrace, dict):
            st = rtrace.get("selection_trace", {})
            if isinstance(st, dict):
                print(f"repo_policy={st.get('policy_name', '')}")
                print(f"repo_selected_chunks={len(repo_sel.get('selected_chunks', []))}")
    print(f"saved_report={report_md}")
    print(f"saved_trace={trace_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
