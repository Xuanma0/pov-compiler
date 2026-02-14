from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pov_compiler.context.context_builder import build_context
from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import QueryPlan, plan as plan_query
from pov_compiler.retrieval.reranker import rerank
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.retrieval.rerank_debug import explain_scores
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Output


def _as_output(output_json: str | Path | dict[str, Any] | Output) -> Output:
    if isinstance(output_json, Output):
        return output_json
    if isinstance(output_json, (str, Path)):
        data = json.loads(Path(output_json).read_text(encoding="utf-8"))
    elif isinstance(output_json, dict):
        data = output_json
    else:
        raise TypeError("output_json must be Output, dict, or path")
    if hasattr(Output, "model_validate"):
        return Output.model_validate(data)  # type: ignore[attr-defined]
    return Output.parse_obj(data)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(float(a0), float(b0)) <= min(float(a1), float(b1))


def trace_query(
    *,
    output_json: str | Path | dict[str, Any] | Output,
    query: str,
    index_prefix: str | Path | None = None,
    retrieval_config: dict[str, Any] | None = None,
    hard_constraints_cfg: HardConstraintConfig | dict[str, Any] | str | Path | None = None,
    rerank_cfg: WeightConfig | dict[str, Any] | str | Path | None = None,
    top_k: int = 6,
    enable_constraints: bool = True,
    use_repo: bool = False,
) -> dict[str, Any]:
    output = ensure_events_v1(_as_output(output_json))
    retriever = Retriever(output_json=output, index=index_prefix, config=dict(retrieval_config or {}))
    plan: QueryPlan = plan_query(str(query))
    candidates = [dict(c) for c in plan.candidates]
    candidate_queries = [str(c["query"]) for c in candidates]
    raw_hits = retriever.retrieve_multi(candidate_queries)

    if bool(enable_constraints):
        constraint_result = apply_constraints_detailed(
            hits=raw_hits,
            query_plan=plan,
            cfg=hard_constraints_cfg,
            output=output,
        )
    else:
        from pov_compiler.retrieval.constraints import ConstraintApplyResult

        constraint_result = ConstraintApplyResult(
            hits=list(raw_hits),
            applied=[],
            relaxed=[],
            used_fallback=False,
            filtered_before=len(raw_hits),
            filtered_after=len(raw_hits),
        )
    weights = resolve_weight_config(rerank_cfg)
    reranked_hits = rerank(
        constraint_result.hits,
        plan=plan,
        context=output,
        cfg=weights,
        distractors=[],
        constraint_trace={
            "applied_constraints": list(constraint_result.applied),
            "constraints_relaxed": list(constraint_result.relaxed),
            "filtered_hits_before": int(constraint_result.filtered_before),
            "filtered_hits_after": int(constraint_result.filtered_after),
            "used_fallback": bool(constraint_result.used_fallback),
        },
    )
    scored = explain_scores(
        reranked_hits,
        plan=plan,
        cfg=weights,
        context=output,
        distractors=[],
    )

    hit_rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(reranked_hits[: max(1, int(top_k))], start=1):
        hit_id = str(hit["id"])
        t0 = float(hit["t0"])
        t1 = float(hit["t1"])
        breakdown = next((x for x in scored if str(x["id"]) == hit_id and str(x["kind"]) == str(hit["kind"])), {})
        linked_events = []
        evidence_spans = []
        for event in output.events_v1:
            if _overlap(t0, t1, float(event.t0), float(event.t1)):
                linked_events.append(event.id)
                for evd in event.evidence:
                    if _overlap(t0, t1, float(evd.t0), float(evd.t1)):
                        evidence_spans.append(
                            {
                                "event_v1_id": event.id,
                                "evidence_id": evd.id,
                                "evidence_type": evd.type,
                                "t0": float(evd.t0),
                                "t1": float(evd.t1),
                                "conf": float(evd.conf),
                            }
                        )
        hit_rows.append(
            {
                "rank": rank,
                "kind": str(hit["kind"]),
                "id": hit_id,
                "t0": t0,
                "t1": t1,
                "score": float(hit["score"]),
                "source_query": str(hit.get("source_query", "")),
                "linked_events_v1": linked_events,
                "distractor_flag": bool(float(breakdown.get("distractor_penalty", 0.0)) < 0.0),
                "score_breakdown": breakdown,
                "score_decomposition": {
                    "semantic_score": float(breakdown.get("semantic_score", 0.0)),
                    "decision_align_score": float(breakdown.get("decision_align_score", 0.0)),
                    "final_score": float(breakdown.get("total", 0.0)),
                },
                "evidence_spans": evidence_spans,
            }
        )

    selected_events_for_repo: list[str] = []
    selected_highlights_for_repo: list[str] = []
    selected_tokens_for_repo: list[str] = []
    selected_decisions_for_repo: list[str] = []
    for hit in hit_rows[: max(1, int(top_k))]:
        kind = str(hit.get("kind", ""))
        hid = str(hit.get("id", ""))
        if not hid:
            continue
        if kind == "event":
            selected_events_for_repo.append(hid)
        elif kind == "highlight":
            selected_highlights_for_repo.append(hid)
        elif kind == "token":
            selected_tokens_for_repo.append(hid)
        elif kind == "decision":
            selected_decisions_for_repo.append(hid)
    repo_selection: dict[str, Any] = {}
    if bool(use_repo):
        repo_ctx = build_context(
            output_json=output,
            mode="repo_only",
            budget={
                "use_repo": True,
                "repo_read_policy": "query_aware",
                "repo_strategy": "query_aware",
                "repo_query": str(query),
                "max_repo_chunks": max(6, int(top_k) * 4),
                "max_repo_tokens": 240,
            },
            selected_events=sorted(set(selected_events_for_repo)) or None,
            selected_highlights=sorted(set(selected_highlights_for_repo)) or None,
            selected_tokens=sorted(set(selected_tokens_for_repo)) or None,
            selected_decisions=sorted(set(selected_decisions_for_repo)) or None,
            query_info={
                "query": str(query),
                "plan_intent": str(plan.intent),
                "parsed_constraints": dict(plan.constraints),
                "top_k": int(top_k),
            },
        )
        repo_selection = {
            "enabled": True,
            "selected_chunks": list(repo_ctx.get("repo_chunks", [])),
            "trace": dict(repo_ctx.get("repo_trace", {})),
        }

    place_counts: dict[str, int] = {}
    interaction_rows: list[dict[str, Any]] = []
    for hit in hit_rows[: max(1, int(top_k))]:
        sb = dict(hit.get("score_breakdown", {}))
        place_id = str(sb.get("place_segment_id", hit.get("score_breakdown", {}).get("place_segment_id", "")))
        # source place/object fields are stored in hit meta via explain_scores payload.
        meta = {}
        for rh in reranked_hits:
            if str(rh.get("id", "")) == str(hit.get("id", "")) and str(rh.get("kind", "")) == str(hit.get("kind", "")):
                meta = dict(rh.get("meta", {}))
                break
        place_id = str(meta.get("place_segment_id", place_id)).strip()
        if place_id:
            place_counts[place_id] = int(place_counts.get(place_id, 0)) + 1
        interaction_rows.append(
            {
                "rank": int(hit.get("rank", 0)),
                "kind": str(hit.get("kind", "")),
                "id": str(hit.get("id", "")),
                "interaction_score": float(meta.get("interaction_score", 0.0) or 0.0),
                "interaction_primary_object": str(meta.get("interaction_primary_object", "")),
                "place_segment_id": place_id,
            }
        )

    place_distribution = [
        {"place_segment_id": key, "count": int(value)}
        for key, value in sorted(place_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    ]
    object_memory_summary = [
        {
            "object_name": str(item.object_name),
            "last_seen_t_ms": int(item.last_seen_t_ms),
            "last_contact_t_ms": int(item.last_contact_t_ms) if item.last_contact_t_ms is not None else None,
            "last_place_id": str(item.last_place_id or ""),
            "score": float(item.score),
        }
        for item in sorted(
            list(output.object_memory_v0),
            key=lambda x: (
                int(x.last_contact_t_ms or 0),
                int(x.last_seen_t_ms or 0),
                float(x.score),
            ),
            reverse=True,
        )[:10]
    ]

    return {
        "video_id": output.video_id,
        "query": str(query),
        "plan": {
            "intent": plan.intent,
            "constraints": dict(plan.constraints),
            "candidates": candidates,
            "debug": dict(plan.debug),
        },
        "constraint_trace": {
            "enable_constraints": bool(enable_constraints),
            "applied_constraints": list(constraint_result.applied),
            "constraints_relaxed": list(constraint_result.relaxed),
            "filtered_hits_before": int(constraint_result.filtered_before),
            "filtered_hits_after": int(constraint_result.filtered_after),
            "used_fallback": bool(constraint_result.used_fallback),
        },
        "rerank_cfg_hash": str(weights.short_hash()),
        "top1_kind": str(hit_rows[0]["kind"]) if hit_rows else "",
        "object_memory_summary": object_memory_summary,
        "place_segment_distribution": place_distribution,
        "interaction_topk": interaction_rows,
        "repo_selection": repo_selection,
        "hits": hit_rows,
    }
