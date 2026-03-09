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


def _token_set(values: Any) -> set[str]:
    if isinstance(values, list):
        return {str(x).upper() for x in values if str(x).strip()}
    if isinstance(values, str) and values.strip():
        return {values.strip().upper()}
    return set()


def _decision_alignment_features(
    *,
    hit: Hit,
    plan: QueryPlan,
    constraints: dict[str, Any],
    constraint_trace: dict[str, Any] | None,
) -> dict[str, float]:
    if str(hit["kind"]) != "decision":
        return {
            "trigger_match": 0.0,
            "action_match": 0.0,
            "constraint_match": 0.0,
            "outcome_match": 0.0,
            "evidence_quality": 0.0,
        }

    meta = dict(hit.get("meta", {}))
    source_query = str(hit.get("source_query", "")).lower()
    action_type = str(meta.get("action_type", "")).upper()
    trigger_anchor_types = {str(x).lower() for x in meta.get("trigger_anchor_types", []) if str(x).strip()}
    if not trigger_anchor_types:
        at = str(meta.get("trigger_anchor_type", "")).lower().strip()
        if at:
            trigger_anchor_types.add(at)

    # a) trigger_match: query anchor hint vs decision trigger.
    trigger_match = 0.0
    anchor_hint = str(constraints.get("anchor_type", "")).lower().strip()
    if anchor_hint:
        trigger_match = 1.0 if anchor_hint in trigger_anchor_types else 0.0
    elif "turn_head" in source_query and "turn_head" in trigger_anchor_types:
        trigger_match = 1.0
    elif "stop_look" in source_query and "stop_look" in trigger_anchor_types:
        trigger_match = 1.0

    # b) action_match: decision_type/anchor hint vs action.
    action_match = 0.0
    decision_hint = str(constraints.get("decision_type", "")).upper().strip()
    if decision_hint:
        action_match = 1.0 if action_type == decision_hint else 0.0
    elif anchor_hint == "turn_head":
        action_match = 1.0 if "TURN_HEAD" in action_type else 0.0
    elif anchor_hint == "stop_look":
        action_match = 1.0 if "STOP_LOOK" in action_type else 0.0

    # c) constraint_match: structural constraints vs decision fields.
    checks: list[float] = []
    decision_constraints = meta.get("decision_constraints", [])
    if not isinstance(decision_constraints, list):
        decision_constraints = []
    if bool(constraints.get("after_scene_change", False)):
        checks.append(1.0 if bool(meta.get("state_scene_change_nearby", False)) else 0.0)
    which = str(constraints.get("which", "")).lower().strip()
    if which in {"first", "last"}:
        rank_key = str(meta.get("decision_order_hint", "")).lower()
        checks.append(1.0 if rank_key == which else 0.0)
    token_hint = str(constraints.get("token_type", "")).upper().strip()
    if token_hint:
        nearby = _token_set(meta.get("state_nearby_tokens", []))
        checks.append(1.0 if token_hint in nearby else 0.0)
    if constraints.get("contact_min", None) is not None:
        try:
            cmin = float(constraints.get("contact_min"))
        except Exception:
            cmin = 0.0
        score_max = 0.0
        for item in decision_constraints:
            if not isinstance(item, dict):
                continue
            try:
                score_max = max(score_max, float(item.get("score", 0.0) or 0.0))
            except Exception:
                continue
        checks.append(1.0 if score_max >= cmin else 0.0)
    constraint_match = float(sum(checks) / len(checks)) if checks else 0.0

    # d) outcome_match: query hint vs outcome.
    outcome_type = str(meta.get("outcome_type", "")).upper().strip()
    outcome_match = 0.0
    if token_hint == "SCENE_CHANGE":
        outcome_match = 1.0 if "SCENE_CHANGED" in outcome_type else 0.0
    elif "resume" in source_query or "moving" in source_query:
        outcome_match = 1.0 if outcome_type in {"RESUME_MOVING", "MOTION_INCREASE"} else 0.0
    elif "stop" in source_query or "still" in source_query:
        outcome_match = 1.0 if outcome_type in {"STOPPED", "STILL_CONTINUE"} else 0.0

    # e) evidence_quality: coverage with distractor/over-filter penalty.
    coverage = meta.get("evidence_coverage", None)
    try:
        evidence_coverage = max(0.0, min(1.0, float(coverage)))
    except Exception:
        token_count = int(meta.get("evidence_token_count", 0))
        evidence_coverage = max(0.0, min(1.0, float(token_count) / 5.0))
    top1_in_distractor = bool(meta.get("top1_in_distractor", False))
    distractor_flag = bool(meta.get("distractor_flag", False))
    over_filtered = bool(meta.get("over_filtered", False))
    if constraint_trace:
        over_filtered = over_filtered or int(constraint_trace.get("filtered_hits_after", 0)) <= 0
        if int(constraint_trace.get("filtered_hits_after", 0)) < int(constraint_trace.get("filtered_hits_before", 0)):
            over_filtered = True
    evidence_quality = evidence_coverage
    if top1_in_distractor or distractor_flag:
        evidence_quality -= 0.6
    if over_filtered:
        evidence_quality -= 0.3
    evidence_quality = max(0.0, min(1.0, evidence_quality))

    return {
        "trigger_match": float(trigger_match),
        "action_match": float(action_match),
        "constraint_match": float(constraint_match),
        "outcome_match": float(outcome_match),
        "evidence_quality": float(evidence_quality),
    }


def score_hit_components(
    *,
    hit: Hit,
    plan: QueryPlan,
    context: Output,
    cfg: WeightConfig | dict[str, Any] | str | None = None,
    distractors: list[tuple[float, float]] | None = None,
    first_last: dict[str, tuple[str, str]] | None = None,
    scene_change_t: float | None = None,
    constraint_trace: dict[str, Any] | None = None,
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

    place_match_bonus = 0.0
    chain_place_mode = str(constraints.get("chain_place_mode", "")).strip().lower()
    chain_place_value = str(constraints.get("chain_place_value", "")).strip()
    if chain_place_mode == "soft" and chain_place_value:
        hit_place = str(meta.get("place_segment_id", "")).strip()
        if hit_place and hit_place == chain_place_value:
            place_match_bonus = float(weights.bonus_token_highlight_overlap)

    object_match_bonus = 0.0
    chain_object_mode = str(constraints.get("chain_object_mode", "")).strip().lower()
    chain_object_value = str(constraints.get("chain_object_value", "")).strip().lower()
    if chain_object_mode == "soft" and chain_object_value:
        hit_object = str(meta.get("interaction_primary_object", meta.get("object_name", ""))).strip().lower()
        if hit_object and (chain_object_value in hit_object or hit_object in chain_object_value):
            object_match_bonus = float(weights.bonus_token_highlight_overlap)
    chain_place_bonus = float(place_match_bonus)
    chain_object_bonus = float(object_match_bonus)
    match_score += float(chain_place_bonus + chain_object_bonus)

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

    semantic_score = (
        base_score
        + intent_bonus
        + match_score
        + first_last_bonus
        + scene_penalty
        + distractor_penalty
        + conf_bonus
        + boundary_bonus
        + priority_bonus
    )
    decision_features = _decision_alignment_features(
        hit=hit,
        plan=plan,
        constraints=constraints,
        constraint_trace=constraint_trace,
    )
    decision_align_score = (
        float(weights.w_trigger) * float(decision_features["trigger_match"])
        + float(weights.w_action) * float(decision_features["action_match"])
        + float(weights.w_constraint) * float(decision_features["constraint_match"])
        + float(weights.w_outcome) * float(decision_features["outcome_match"])
        + float(weights.w_evidence) * float(decision_features["evidence_quality"])
    )
    total = float(weights.w_semantic) * float(semantic_score) + float(decision_align_score)
    return {
        "base_score": float(base_score),
        "semantic_score": float(semantic_score),
        "decision_align_score": float(decision_align_score),
        "intent_bonus": float(intent_bonus),
        "match_score": float(match_score),
        "first_last_bonus": float(first_last_bonus),
        "scene_penalty": float(scene_penalty),
        "distractor_penalty": float(distractor_penalty),
        "conf_bonus": float(conf_bonus),
        "boundary_bonus": float(boundary_bonus),
        "priority_bonus": float(priority_bonus),
        "place_match_bonus": float(place_match_bonus),
        "object_match_bonus": float(object_match_bonus),
        "chain_place_bonus": float(chain_place_bonus),
        "chain_object_bonus": float(chain_object_bonus),
        "trigger_match": float(decision_features["trigger_match"]),
        "action_match": float(decision_features["action_match"]),
        "constraint_match": float(decision_features["constraint_match"]),
        "outcome_match": float(decision_features["outcome_match"]),
        "evidence_quality": float(decision_features["evidence_quality"]),
        "w_semantic": float(weights.w_semantic),
        "total_adjustment": float(total - base_score),
        "total": float(total),
    }


def rerank(
    hits: list[Hit],
    plan: QueryPlan,
    context: Output,
    cfg: WeightConfig | dict[str, Any] | str | None = None,
    distractors: list[tuple[float, float]] | None = None,
    constraint_trace: dict[str, Any] | None = None,
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
            constraint_trace=constraint_trace,
        )
        final_score = float(parts["total"])
        meta["rerank"] = {
            "base_score": float(parts["base_score"]),
            "semantic_score": float(parts["semantic_score"]),
            "decision_align_score": float(parts["decision_align_score"]),
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
                "place_match_bonus": float(parts.get("place_match_bonus", 0.0)),
                "object_match_bonus": float(parts.get("object_match_bonus", 0.0)),
                "chain_place_bonus": float(parts.get("chain_place_bonus", 0.0)),
                "chain_object_bonus": float(parts.get("chain_object_bonus", 0.0)),
                "trigger_match": float(parts["trigger_match"]),
                "action_match": float(parts["action_match"]),
                "constraint_match": float(parts["constraint_match"]),
                "outcome_match": float(parts["outcome_match"]),
                "evidence_quality": float(parts["evidence_quality"]),
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
