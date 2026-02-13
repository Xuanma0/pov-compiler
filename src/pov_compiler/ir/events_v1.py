from __future__ import annotations

import json
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
            meta={
                "layer": layer,
                "source_event_id": str(event.id),
                "anchors_count": len(event.anchors),
                "highlights_count": sum(1 for e in ev_evidence if e.type == "highlight"),
                "tokens_count": sum(1 for e in ev_evidence if e.type == "token"),
                "decisions_count": sum(1 for e in ev_evidence if e.type == "decision"),
                "contacts_count": sum(1 for e in ev_evidence if e.type == "contact"),
            },
        )
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

