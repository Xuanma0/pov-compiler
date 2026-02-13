from __future__ import annotations

from typing import Any, TypedDict

from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.schemas import Output


class Hit(TypedDict):
    kind: str
    id: str
    t0: float
    t1: float
    score: float
    source_query: str
    meta: dict[str, Any]


def _center(t0: float, t1: float) -> float:
    return 0.5 * (float(t0) + float(t1))


def _scene_change_reference_time(output: Output) -> float | None:
    times = []
    for tok in output.token_codec.tokens:
        if str(tok.type).upper() == "SCENE_CHANGE":
            times.append(_center(float(tok.t0), float(tok.t1)))
    if not times:
        return None
    return float(min(times))


def _intent_bonus(intent: str, kind: str, cfg: WeightConfig) -> float:
    table: dict[str, dict[str, float]] = {
        "token": {
            "token": float(cfg.bonus_intent_token_on_token),
            "highlight": float(cfg.bonus_intent_token_on_highlight),
            "decision": float(cfg.bonus_intent_token_on_decision),
            "event": float(cfg.bonus_intent_token_on_event),
        },
        "decision": {
            "decision": float(cfg.bonus_intent_decision_on_decision),
            "highlight": float(cfg.bonus_intent_decision_on_highlight),
            "token": float(cfg.bonus_intent_decision_on_token),
            "event": float(cfg.bonus_intent_decision_on_event),
        },
        "anchor": {
            "highlight": float(cfg.bonus_intent_anchor_on_highlight),
            "decision": float(cfg.bonus_intent_anchor_on_decision),
            "token": float(cfg.bonus_intent_anchor_on_token),
            "event": float(cfg.bonus_intent_anchor_on_event),
        },
        "time": {
            "event": float(cfg.bonus_intent_time_on_event),
            "highlight": float(cfg.bonus_intent_time_on_highlight),
            "decision": float(cfg.bonus_intent_time_on_decision),
            "token": float(cfg.bonus_intent_time_on_token),
        },
        "mixed": {
            "highlight": float(cfg.bonus_intent_mixed_on_highlight),
            "decision": float(cfg.bonus_intent_mixed_on_decision),
            "token": float(cfg.bonus_intent_mixed_on_token),
            "event": float(cfg.bonus_intent_mixed_on_event),
        },
    }
    return float(table.get(str(intent), table["mixed"]).get(str(kind), 0.0))


def _meta_bonus_breakdown(meta: dict[str, Any], cfg: WeightConfig) -> tuple[float, float, float]:
    conf_bonus = 0.0
    boundary_bonus = 0.0
    priority_bonus = 0.0
    try:
        conf = float(meta.get("conf", 0.0))
        conf_bonus += max(0.0, min(1.0, conf)) * float(cfg.bonus_conf_scale)
    except Exception:
        pass
    try:
        boundary_conf = float(meta.get("boundary_conf", 0.0))
        boundary_bonus += max(0.0, min(1.0, boundary_conf)) * float(cfg.bonus_boundary_scale)
    except Exception:
        pass
    try:
        priority_max = float(meta.get("priority_max", 0.0))
        priority_bonus += max(0.0, min(float(cfg.bonus_priority_cap), priority_max)) * float(cfg.bonus_priority_scale)
    except Exception:
        pass
    return float(conf_bonus), float(boundary_bonus), float(priority_bonus)


def _match_anchor_constraint(hit: Hit, anchor_type: str, cfg: WeightConfig) -> float:
    kind = str(hit["kind"])
    meta = hit.get("meta", {})
    anchor_type = str(anchor_type).lower()
    if kind == "highlight":
        values = meta.get("anchor_types")
        if isinstance(values, list):
            if anchor_type in {str(x).lower() for x in values}:
                return float(cfg.bonus_anchor_highlight_match)
        if str(meta.get("anchor_type", "")).lower() == anchor_type:
            return float(cfg.bonus_anchor_highlight_match)
        return float(cfg.penalty_anchor_highlight_mismatch)
    if kind == "decision":
        action_type = str(meta.get("action_type", "")).upper()
        if anchor_type == "turn_head" and "TURN_HEAD" in action_type:
            return float(cfg.bonus_anchor_decision_match)
        if anchor_type == "stop_look" and "STOP_LOOK" in action_type:
            return float(cfg.bonus_anchor_decision_match)
        return float(cfg.penalty_anchor_decision_mismatch)
    return 0.0


def _match_token_constraint(hit: Hit, token_type: str, cfg: WeightConfig) -> float:
    kind = str(hit["kind"])
    meta = hit.get("meta", {})
    token_type = str(token_type).upper()
    if kind == "token":
        if str(meta.get("token_type", "")).upper() == token_type:
            return float(cfg.bonus_token_match)
        return float(cfg.penalty_token_mismatch)
    if kind == "highlight":
        values = meta.get("token_types")
        if isinstance(values, list) and token_type in {str(x).upper() for x in values}:
            return float(cfg.bonus_token_highlight_overlap)
    return 0.0


def _match_decision_constraint(hit: Hit, decision_type: str, cfg: WeightConfig) -> float:
    kind = str(hit["kind"])
    meta = hit.get("meta", {})
    decision_type = str(decision_type).upper()
    if kind == "decision":
        if str(meta.get("action_type", "")).upper() == decision_type:
            return float(cfg.bonus_decision_match)
        return float(cfg.penalty_decision_mismatch)
    return 0.0


def score_hit_components(
    *,
    hit: Hit,
    plan: QueryPlan,
    context: Output,
    cfg: WeightConfig | dict[str, Any] | str | None = None,
    distractors: list[tuple[float, float]] | None = None,
    first_last: dict[str, tuple[str, str]] | None = None,
    scene_change_t: float | None = None,
) -> dict[str, Any]:
    weights = resolve_weight_config(cfg)
    constraints = dict(plan.constraints)
    kind = str(hit["kind"])
    t0 = float(hit["t0"])
    t1 = float(hit["t1"])
    meta = dict(hit.get("meta", {}))
    base_score = float(hit.get("score", 0.0))

    intent_bonus = _intent_bonus(plan.intent, kind, weights)
    conf_bonus, boundary_bonus, priority_bonus = _meta_bonus_breakdown(meta, weights)

    match_score = 0.0
    anchor_type = constraints.get("anchor_type")
    if anchor_type:
        match_score += _match_anchor_constraint(hit, str(anchor_type), weights)
    token_type = constraints.get("token_type")
    if token_type:
        match_score += _match_token_constraint(hit, str(token_type), weights)
    decision_type = constraints.get("decision_type")
    if decision_type:
        match_score += _match_decision_constraint(hit, str(decision_type), weights)

    first_last_bonus = 0.0
    which = str(constraints.get("which", "")).lower()
    if which in {"first", "last"} and first_last and kind in first_last:
        first_id, last_id = first_last[kind]
        if which == "first" and str(hit["id"]) == first_id:
            first_last_bonus = float(weights.bonus_first)
        elif which == "last" and str(hit["id"]) == last_id:
            first_last_bonus = float(weights.bonus_last)

    scene_penalty = 0.0
    resolved_scene_t = scene_change_t if scene_change_t is not None else _scene_change_reference_time(context)
    if bool(constraints.get("after_scene_change", False)) and resolved_scene_t is not None and t1 < float(resolved_scene_t):
        # Keep this term soft and bounded; hard filtering is handled in constraints.py.
        scene_penalty = -abs(float(weights.penalty_before_scene_change))

    distractor_penalty = 0.0
    if distractors:
        distractor_centers = [_center(float(d0), float(d1)) for d0, d1 in distractors]
        center = _center(t0, t1)
        if distractor_centers and min(abs(center - dc) for dc in distractor_centers) < float(weights.distractor_near_window_s):
            distractor_penalty = -float(weights.penalty_distractor_near)

    total = (
        base_score
        + intent_bonus
        + conf_bonus
        + boundary_bonus
        + priority_bonus
        + match_score
        + first_last_bonus
        + scene_penalty
        + distractor_penalty
    )
    return {
        "base_score": float(base_score),
        "intent_bonus": float(intent_bonus),
        "match_score": float(match_score),
        "first_last_bonus": float(first_last_bonus),
        "scene_penalty": float(scene_penalty),
        "distractor_penalty": float(distractor_penalty),
        "conf_bonus": float(conf_bonus),
        "boundary_bonus": float(boundary_bonus),
        "priority_bonus": float(priority_bonus),
        "total_adjustment": float(total - base_score),
        "total": float(total),
    }


def rerank(
    hits: list[Hit],
    plan: QueryPlan,
    context: Output,
    cfg: WeightConfig | dict[str, Any] | str | None = None,
    distractors: list[tuple[float, float]] | None = None,
) -> list[Hit]:
    if not hits:
        return []

    weights = resolve_weight_config(cfg)
    constraints = dict(plan.constraints)
    scene_change_t = _scene_change_reference_time(context)

    # Cache first/last per kind for which-constraint.
    grouped: dict[str, list[Hit]] = {}
    for hit in hits:
        grouped.setdefault(str(hit["kind"]), []).append(hit)
    first_last: dict[str, tuple[str, str]] = {}
    for kind, values in grouped.items():
        ordered = sorted(values, key=lambda x: (_center(float(x["t0"]), float(x["t1"])), str(x["id"])))
        first_last[kind] = (str(ordered[0]["id"]), str(ordered[-1]["id"]))

    out: list[Hit] = []
    for hit in hits:
        kind = str(hit["kind"])
        t0 = float(hit["t0"])
        t1 = float(hit["t1"])
        meta = dict(hit.get("meta", {}))
        parts = score_hit_components(
            hit=hit,
            plan=plan,
            context=context,
            cfg=weights,
            distractors=distractors,
            first_last=first_last,
            scene_change_t=scene_change_t,
        )
        final_score = float(parts["total"])
        meta["rerank"] = {
            "base_score": float(parts["base_score"]),
            "adjustments": float(parts["total_adjustment"]),
            "final_score": float(parts["total"]),
            "intent": plan.intent,
            "constraints": constraints,
            "cfg_name": str(weights.name),
            "cfg_hash": str(weights.short_hash()),
            "components": {
                "intent_bonus": float(parts["intent_bonus"]),
                "match_score": float(parts["match_score"]),
                "first_last_bonus": float(parts["first_last_bonus"]),
                "scene_penalty": float(parts["scene_penalty"]),
                "distractor_penalty": float(parts["distractor_penalty"]),
                "conf_bonus": float(parts["conf_bonus"]),
                "boundary_bonus": float(parts["boundary_bonus"]),
                "priority_bonus": float(parts["priority_bonus"]),
            },
        }
        out.append(
            Hit(
                kind=kind,
                id=str(hit["id"]),
                t0=t0,
                t1=t1,
                score=float(final_score),
                source_query=str(hit.get("source_query", "")),
                meta=meta,
            )
        )

    out.sort(key=lambda x: (-float(x["score"]), float(x["t0"]), str(x["kind"]), str(x["id"])))
    return out
