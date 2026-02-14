from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pov_compiler.schemas import (
    ConstraintTrace,
    DecisionPoint,
    Evidence,
    Event,
    EventV1,
    KeyClip,
    Output,
    RetrievalHit,
    ScoreBreakdown,
    Token,
)


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


def _duration_from_output(output: Output) -> float:
    duration_s = float(output.meta.get("duration_s", 0.0) or 0.0)
    if duration_s > 0.0:
        return duration_s
    all_t: list[float] = []
    for ev in list(output.events) + list(output.events_v0):
        all_t.append(float(ev.t1))
    for hl in output.highlights:
        all_t.append(float(hl.t1))
    for tok in output.token_codec.tokens:
        all_t.append(float(tok.t1))
    for dp in output.decision_points:
        all_t.append(float(dp.t1))
    return max(all_t) if all_t else 0.0


def _extract_time_and_boundary(output: Output) -> tuple[list[float], list[float]]:
    debug = output.debug if isinstance(output.debug, dict) else {}
    signals = debug.get("signals", {}) if isinstance(debug, dict) else {}
    if not isinstance(signals, dict):
        return [], []
    times_raw = signals.get("time", [])
    score_raw = signals.get("boundary_score", [])
    if not isinstance(times_raw, list) or not isinstance(score_raw, list):
        return [], []
    n = min(len(times_raw), len(score_raw))
    if n <= 0:
        return [], []
    times = [float(x) for x in times_raw[:n]]
    scores = [float(x) for x in score_raw[:n]]
    return times, scores


def build_place_segments_v0(
    output: Output,
    *,
    boundary_thresh: float = 0.65,
    min_segment_s: float = 2.0,
) -> list[dict[str, Any]]:
    duration_s = _duration_from_output(output)
    if duration_s <= 0.0:
        return []

    boundaries: list[tuple[float, float, str]] = []
    for token in output.token_codec.tokens:
        if str(token.type).upper() != "SCENE_CHANGE":
            continue
        t = 0.5 * (float(token.t0) + float(token.t1))
        boundaries.append((max(0.0, min(duration_s, t)), 0.8, "scene_change"))

    times, scores = _extract_time_and_boundary(output)
    for idx, score in enumerate(scores):
        if float(score) < float(boundary_thresh):
            continue
        prev_val = scores[idx - 1] if idx > 0 else score
        next_val = scores[idx + 1] if idx + 1 < len(scores) else score
        if float(score) < float(prev_val) or float(score) < float(next_val):
            continue
        boundaries.append((max(0.0, min(duration_s, float(times[idx]))), min(1.0, max(0.0, float(score))), "visual_change"))

    boundaries.sort(key=lambda x: (float(x[0]), -float(x[1]), str(x[2])))
    dedup: list[tuple[float, float, str]] = []
    for t, conf, reason in boundaries:
        if t <= 0.0 or t >= duration_s:
            continue
        if dedup and abs(float(dedup[-1][0]) - float(t)) < 0.2:
            if float(conf) > float(dedup[-1][1]):
                dedup[-1] = (float(t), float(conf), str(reason))
            continue
        dedup.append((float(t), float(conf), str(reason)))

    segments: list[dict[str, Any]] = []
    prev_t = 0.0
    prev_conf = 0.5
    prev_reason = "heuristic_merge"
    raw: list[tuple[float, float, float, str]] = []
    for t, conf, reason in dedup:
        if t <= prev_t:
            continue
        raw.append((float(prev_t), float(t), float(conf), str(reason)))
        prev_t = float(t)
        prev_conf = float(conf)
        prev_reason = str(reason)
    if prev_t < duration_s:
        raw.append((float(prev_t), float(duration_s), float(prev_conf), str(prev_reason)))
    if not raw:
        raw.append((0.0, float(duration_s), 0.5, "heuristic_merge"))

    merged: list[tuple[float, float, float, str]] = []
    for t0, t1, conf, reason in raw:
        if not merged:
            merged.append((t0, t1, conf, reason))
            continue
        prev0, prev1, prev_conf_val, prev_reason_val = merged[-1]
        if (t1 - t0) < float(min_segment_s):
            merged[-1] = (prev0, t1, min(1.0, max(prev_conf_val, conf)), "heuristic_merge")
            continue
        merged.append((t0, t1, conf, reason))

    out: list[dict[str, Any]] = []
    for idx, (t0, t1, conf, reason) in enumerate(merged, start=1):
        out.append(
            {
                "id": f"place_{idx:04d}",
                "t0": float(t0),
                "t1": float(t1),
                "conf": float(max(0.0, min(1.0, conf))),
                "reason": str(reason) if str(reason) else "heuristic_merge",
            }
        )
    return out


def aggregate_interaction_signature(
    frames: list[dict[str, Any]],
    *,
    t0: float,
    t1: float,
) -> dict[str, Any]:
    t0f = float(t0)
    t1f = float(t1)
    if t1f < t0f:
        t0f, t1f = t1f, t0f
    selected = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            t = float(frame.get("t", -1.0))
        except Exception:
            continue
        if t0f <= t <= t1f:
            selected.append(frame)

    total = len(selected)
    active_flags: list[bool] = []
    active_scores: list[float] = []
    object_counts: dict[str, int] = defaultdict(int)
    for frame in selected:
        contact = frame.get("contact", {})
        if not isinstance(contact, dict):
            contact = {}
        active = contact.get("active")
        if isinstance(active, dict):
            label = str(active.get("label", "")).strip().lower()
            object_id = str(active.get("object_id", "")).strip().lower()
            key = label or object_id or ""
            if key:
                object_counts[key] += 1
            try:
                score = float(active.get("score", contact.get("active_score", 0.0)) or 0.0)
            except Exception:
                score = 0.0
            active_scores.append(max(0.0, min(1.0, score)))
            active_flags.append(True)
        else:
            try:
                score = float(contact.get("active_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if score > 1e-6:
                active_scores.append(max(0.0, min(1.0, score)))
                active_flags.append(True)
            else:
                active_flags.append(False)

    active_frames = int(sum(1 for x in active_flags if x))
    contact_rate = float(active_frames / max(1, total))
    bursts = 0
    prev = False
    for flag in active_flags:
        if flag and not prev:
            bursts += 1
        prev = bool(flag)
    avg_active_score = float(sum(active_scores) / max(1, len(active_scores))) if active_scores else 0.0
    active_object_top1 = ""
    if object_counts:
        active_object_top1 = sorted(object_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]
    interaction_score = float(
        max(
            0.0,
            min(
                1.0,
                (0.55 * contact_rate) + (0.30 * min(1.0, float(bursts) / 3.0)) + (0.15 * avg_active_score),
            ),
        )
    )
    return {
        "active_object_top1": str(active_object_top1),
        "active_object": str(active_object_top1),
        "active_object_counts": dict(sorted(object_counts.items())),
        "contact_burst_count": int(bursts),
        "contact_bursts": int(bursts),
        "contact_rate": float(contact_rate),
        "active_frames": int(active_frames),
        "total_frames": int(total),
        "avg_active_score": float(avg_active_score),
        "interaction_score": float(interaction_score),
    }


def _event_place_info(
    *,
    t0: float,
    t1: float,
    place_segments: list[dict[str, Any]],
) -> tuple[str | None, float | None, str | None]:
    if not place_segments:
        return None, None, None
    best: dict[str, Any] | None = None
    best_overlap = -1.0
    for segment in place_segments:
        s0 = float(segment.get("t0", 0.0))
        s1 = float(segment.get("t1", 0.0))
        inter = max(0.0, min(float(t1), s1) - max(float(t0), s0))
        if inter > best_overlap:
            best = segment
            best_overlap = inter
    if best is None:
        return None, None, None
    return str(best.get("id", "")) or None, float(best.get("conf", 0.0)), str(best.get("reason", "")) or None


def _pick_label(event: Event, *, layer: str) -> str:
    label = str(event.meta.get("label", "")).strip()
    if label:
        return label
    anchor_types = {str(anchor.type) for anchor in event.anchors}
    if "stop_look" in anchor_types and "turn_head" in anchor_types:
        return "interaction-heavy"
    if "stop_look" in anchor_types:
        return "idle"
    if "turn_head" in anchor_types:
        return "navigation"
    return "navigation" if layer == "events" else "event_v0"


def _decision_evidence(decision: DecisionPoint) -> Evidence:
    action = str(decision.action.get("type", ""))
    conf = float(decision.conf)
    return Evidence(
        id=f"evd_{decision.id}",
        type="decision",
        t0=float(decision.t0),
        t1=float(decision.t1),
        conf=conf,
        source={
            "decision_id": decision.id,
            "source_event": decision.source_event,
            "source_highlight": decision.source_highlight,
            "action_type": action,
        },
    )


def _highlight_evidence(hl: KeyClip) -> Evidence:
    anchor_types = hl.meta.get("anchor_types")
    if not isinstance(anchor_types, list):
        anchor_types = [hl.anchor_type]
    return Evidence(
        id=f"evd_{hl.id}",
        type="highlight",
        t0=float(hl.t0),
        t1=float(hl.t1),
        conf=float(hl.conf),
        source={
            "highlight_id": hl.id,
            "source_event": hl.source_event,
            "anchor_type": hl.anchor_type,
            "anchor_types": [str(x) for x in anchor_types],
        },
    )


def _token_evidence(token: Token) -> Evidence:
    return Evidence(
        id=f"evd_{token.id}",
        type="token",
        t0=float(token.t0),
        t1=float(token.t1),
        conf=float(token.conf),
        source={
            "token_id": token.id,
            "token_type": token.type,
            "source_event": token.source_event,
        },
    )


def _anchor_evidence(event_id: str, anchor: Any) -> Evidence:
    t = float(anchor.t)
    return Evidence(
        id=f"evd_{event_id}_{str(anchor.type)}_{int(t * 1000):08d}",
        type="anchor",
        t0=t,
        t1=t,
        conf=float(anchor.conf),
        source={"anchor_type": str(anchor.type), "event_id": event_id, "meta": dict(anchor.meta)},
    )


def _contact_evidence(frame: dict[str, Any], active: dict[str, Any]) -> Evidence:
    t = float(frame.get("t", active.get("t", 0.0)))
    score = float(active.get("score", active.get("active_score", 0.0)))
    object_id = str(active.get("object_id", ""))
    return Evidence(
        id=f"evd_contact_{object_id}_{int(t * 1000):08d}",
        type="contact",
        t0=t,
        t1=t,
        conf=score,
        source={
            "object_id": object_id,
            "object_label": str(active.get("label", "")),
            "hand_id": str(active.get("hand_id", "")),
            "handedness": str(active.get("handedness", "")),
        },
    )


def _with_retrieval_stub(event: EventV1, rerank_cfg_hash: str) -> EventV1:
    if not event.evidence:
        return event
    top1_kind = event.evidence[0].type
    score = float(max(0.0, min(1.0, event.evidence[0].conf)))
    breakdown = ScoreBreakdown(base_score=score, total=score)
    trace = ConstraintTrace(
        source_query="",
        chosen_plan_intent="mixed",
        applied_constraints=[],
        constraints_relaxed=[],
        filtered_hits_before=0,
        filtered_hits_after=0,
        used_fallback=False,
        rerank_cfg_hash=str(rerank_cfg_hash),
        top1_kind=str(top1_kind),
        top1_in_distractor=False,
        score_breakdown=breakdown,
    )
    hit = RetrievalHit(
        kind=str(top1_kind),
        id=str(event.evidence[0].id),
        t0=float(event.evidence[0].t0),
        t1=float(event.evidence[0].t1),
        score=float(score),
        rank=1,
        source_query="",
        chosen_plan_intent="mixed",
        applied_constraints=[],
        score_breakdown=breakdown,
        rerank_cfg_hash=str(rerank_cfg_hash),
        top1_kind=str(top1_kind),
        top1_in_distractor=False,
    )
    event.meta.setdefault("constraint_trace", trace.model_dump() if hasattr(trace, "model_dump") else trace.dict())
    event.meta.setdefault("retrieval_hit", hit.model_dump() if hasattr(hit, "model_dump") else hit.dict())
    for rank, evd in enumerate(event.evidence, start=1):
        evd_score = float(max(0.0, min(1.0, evd.conf)))
        evd_breakdown = ScoreBreakdown(base_score=evd_score, total=evd_score)
        evd.constraint_trace = ConstraintTrace(
            source_query="",
            chosen_plan_intent="mixed",
            applied_constraints=[],
            constraints_relaxed=[],
            filtered_hits_before=0,
            filtered_hits_after=0,
            used_fallback=False,
            rerank_cfg_hash=str(rerank_cfg_hash),
            top1_kind=str(evd.type if rank == 1 else top1_kind),
            top1_in_distractor=False,
            score_breakdown=evd_breakdown,
        )
        evd.retrieval_hit = RetrievalHit(
            kind=str(evd.type),
            id=str(evd.id),
            t0=float(evd.t0),
            t1=float(evd.t1),
            score=evd_score,
            rank=rank,
            source_query="",
            chosen_plan_intent="mixed",
            applied_constraints=[],
            score_breakdown=evd_breakdown,
            rerank_cfg_hash=str(rerank_cfg_hash),
            top1_kind=str(evd.type if rank == 1 else top1_kind),
            top1_in_distractor=False,
        )
    return event


def convert_output_to_events_v1(
    output_json: str | Path | dict[str, Any] | Output,
    *,
    rerank_cfg_hash: str = "",
) -> list[EventV1]:
    output = _as_output(output_json)

    base_events: list[tuple[Event, str]] = []
    if output.events:
        base_events.extend((event, "events") for event in output.events)
    if output.events_v0:
        base_events.extend((event, "events_v0") for event in output.events_v0)
    if not base_events:
        return []

    highlights = list(output.highlights)
    tokens = list(output.token_codec.tokens)
    decisions = list(output.decision_points)
    perception_frames = output.perception.get("frames", []) if isinstance(output.perception, dict) else []
    if not isinstance(perception_frames, list):
        perception_frames = []
    place_segments = build_place_segments_v0(output)

    seen_ids: set[str] = set()
    events_v1: list[EventV1] = []
    for idx, (event, layer) in enumerate(base_events):
        base_id = str(event.id).strip() or f"event_{idx:04d}"
        eid = base_id
        suffix = 1
        while eid in seen_ids:
            suffix += 1
            eid = f"{base_id}__{suffix}"
        seen_ids.add(eid)

        t0 = float(event.t0)
        t1 = float(event.t1)
        ev_evidence: list[Evidence] = []

        for anchor in event.anchors:
            ev_evidence.append(_anchor_evidence(eid, anchor))

        for hl in highlights:
            if _overlap(t0, t1, float(hl.t0), float(hl.t1)):
                ev_evidence.append(_highlight_evidence(hl))

        for token in tokens:
            if _overlap(t0, t1, float(token.t0), float(token.t1)):
                ev_evidence.append(_token_evidence(token))

        for decision in decisions:
            if _overlap(t0, t1, float(decision.t0), float(decision.t1)):
                ev_evidence.append(_decision_evidence(decision))

        for frame in perception_frames:
            if not isinstance(frame, dict):
                continue
            ft = float(frame.get("t", -1.0))
            if ft < t0 or ft > t1:
                continue
            contact = frame.get("contact", {})
            if not isinstance(contact, dict):
                continue
            active = contact.get("active")
            if isinstance(active, dict):
                ev_evidence.append(_contact_evidence(frame, active))

        ev_evidence.sort(key=lambda x: (float(x.t0), float(x.t1), str(x.type), str(x.id)))
        label = _pick_label(event, layer=layer)
        place_segment_id, place_segment_conf, place_segment_reason = _event_place_info(
            t0=t0,
            t1=t1,
            place_segments=place_segments,
        )
        interaction_signature = aggregate_interaction_signature(perception_frames, t0=t0, t1=t1)
        interaction_primary_object = str(interaction_signature.get("active_object_top1", ""))
        interaction_score = float(interaction_signature.get("interaction_score", 0.0))

        hints: set[str] = set()
        anchor_types = {str(anchor.type) for anchor in event.anchors}
        for anchor_type in anchor_types:
            hints.add(f"anchor={anchor_type}")
        for evd in ev_evidence:
            if evd.type == "token":
                token_type = str(evd.source.get("token_type", "")).strip()
                if token_type:
                    hints.add(f"token={token_type}")
            elif evd.type == "decision":
                action_type = str(evd.source.get("action_type", "")).strip()
                if action_type:
                    hints.add(f"decision={action_type}")
        if place_segment_id:
            hints.add(f"place_segment_id={place_segment_id}")
        if interaction_primary_object:
            hints.add(f"interaction_object={interaction_primary_object}")

        duration = max(1e-6, t1 - t0)
        contact_peak = max((float(e.conf) for e in ev_evidence if e.type == "contact"), default=0.0)
        evidence_density = float(len(ev_evidence) / duration)

        ev = EventV1(
            id=eid,
            t0=t0,
            t1=t1,
            label=label,
            source_event_ids=[str(event.id)],
            evidence=ev_evidence,
            retrieval_hints=sorted(hints),
            scores={
                "boundary_conf": float(event.scores.get("boundary_conf", 0.0)),
                "evidence_density": evidence_density,
                "contact_peak": contact_peak,
            },
            place_segment_id=place_segment_id,
            place_segment_conf=place_segment_conf,
            place_segment_reason=place_segment_reason,
            interaction_signature=interaction_signature,
            interaction_primary_object=interaction_primary_object,
            interaction_score=interaction_score,
            meta={
                "layer": layer,
                "source_event_id": str(event.id),
                "anchors_count": len(event.anchors),
                "highlights_count": sum(1 for e in ev_evidence if e.type == "highlight"),
                "tokens_count": sum(1 for e in ev_evidence if e.type == "token"),
                "decisions_count": sum(1 for e in ev_evidence if e.type == "decision"),
                "contacts_count": sum(1 for e in ev_evidence if e.type == "contact"),
                "place_segments_count": len(place_segments),
                "place_segment_id": place_segment_id,
                "interaction_primary_object": interaction_primary_object,
                "interaction_score": interaction_score,
            },
        )
        for evd in ev.evidence:
            evd.place_segment_id = place_segment_id
            evd.place_segment_conf = place_segment_conf
            evd.place_segment_reason = place_segment_reason
            evd.interaction_signature = dict(interaction_signature)
            evd.interaction_primary_object = interaction_primary_object
            evd.interaction_score = interaction_score
        events_v1.append(_with_retrieval_stub(ev, rerank_cfg_hash))

    events_v1.sort(key=lambda x: (float(x.t0), float(x.t1), str(x.id)))
    return events_v1


def ensure_events_v1(
    output_json: str | Path | dict[str, Any] | Output,
    *,
    rerank_cfg_hash: str = "",
) -> Output:
    output = _as_output(output_json)
    if output.events_v1:
        return output
    output.events_v1 = convert_output_to_events_v1(output, rerank_cfg_hash=rerank_cfg_hash)
    return output
