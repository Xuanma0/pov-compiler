from __future__ import annotations

from typing import Any

from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker import Hit, score_hit_components
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.schemas import Output


def explain_scores(
    hits: list[Hit],
    plan: QueryPlan,
    cfg: WeightConfig | dict[str, Any] | str | None,
    context: Output,
    distractors: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    weights = resolve_weight_config(cfg)
    rows: list[dict[str, Any]] = []
    for hit in hits:
        parts = score_hit_components(
            hit=hit,
            plan=plan,
            context=context,
            cfg=weights,
            distractors=distractors,
        )
        rows.append(
            {
                "id": str(hit["id"]),
                "kind": str(hit["kind"]),
                "t0": float(hit["t0"]),
                "t1": float(hit["t1"]),
                "source_query": str(hit.get("source_query", "")),
                "base_score": float(parts["base_score"]),
                "semantic_score": float(parts["semantic_score"]),
                "decision_align_score": float(parts["decision_align_score"]),
                "intent_bonus": float(parts["intent_bonus"]),
                "match_score": float(parts["match_score"]),
                "distractor_penalty": float(parts["distractor_penalty"]),
                "constraint_bonus": float(parts["first_last_bonus"]) + float(parts["scene_penalty"]),
                "first_last_bonus": float(parts["first_last_bonus"]),
                "scene_penalty": float(parts["scene_penalty"]),
                "conf_bonus": float(parts["conf_bonus"]),
                "boundary_bonus": float(parts["boundary_bonus"]),
                "priority_bonus": float(parts["priority_bonus"]),
                "place_match_bonus": float(parts.get("place_match_bonus", 0.0)),
                "object_match_bonus": float(parts.get("object_match_bonus", 0.0)),
                "trigger_match": float(parts["trigger_match"]),
                "action_match": float(parts["action_match"]),
                "constraint_match": float(parts["constraint_match"]),
                "outcome_match": float(parts["outcome_match"]),
                "evidence_quality": float(parts["evidence_quality"]),
                "total_adjustment": float(parts["total_adjustment"]),
                "total": float(parts["total"]),
            }
        )
    rows.sort(key=lambda x: (-float(x["total"]), float(x["t0"]), str(x["kind"]), str(x["id"])))
    return rows
