from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    output = ensure_events_v1(_as_output(output_json))
    retriever = Retriever(output_json=output, index=index_prefix, config=dict(retrieval_config or {}))
    plan: QueryPlan = plan_query(str(query))
    candidates = [dict(c) for c in plan.candidates]
    candidate_queries = [str(c["query"]) for c in candidates]
    raw_hits = retriever.retrieve_multi(candidate_queries)

    constraint_result = apply_constraints_detailed(
        hits=raw_hits,
        query_plan=plan,
        cfg=hard_constraints_cfg,
        output=output,
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
            "applied_constraints": list(constraint_result.applied),
            "constraints_relaxed": list(constraint_result.relaxed),
            "filtered_hits_before": int(constraint_result.filtered_before),
            "filtered_hits_after": int(constraint_result.filtered_after),
            "used_fallback": bool(constraint_result.used_fallback),
        },
        "rerank_cfg_hash": str(weights.short_hash()),
        "top1_kind": str(hit_rows[0]["kind"]) if hit_rows else "",
        "place_segment_distribution": place_distribution,
        "interaction_topk": interaction_rows,
        "hits": hit_rows,
    }
