from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pov_compiler.schemas import EventV1, ObjectMemoryItemV0


@dataclass
class _Accum:
    object_name: str
    last_seen_t_ms: int = 0
    last_contact_t_ms: int | None = None
    seen_count: int = 0
    contact_count: int = 0
    contact_score_max: float = 0.0


def _norm_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if not label:
        return ""
    return " ".join(label.split())


def _to_ms(sec: Any) -> int:
    try:
        return int(round(float(sec) * 1000.0))
    except Exception:
        return 0


def _map_place_id(events_v1: list[EventV1], t_ms: int | None) -> str | None:
    if t_ms is None:
        return None
    t_s = float(t_ms) / 1000.0
    for event in events_v1:
        if float(event.t0) <= t_s <= float(event.t1):
            seg = str(event.place_segment_id or "").strip()
            if seg:
                return seg
    # fallback: nearest event center
    if not events_v1:
        return None
    ordered = sorted(
        events_v1,
        key=lambda ev: abs(((float(ev.t0) + float(ev.t1)) * 0.5) - t_s),
    )
    seg = str(ordered[0].place_segment_id or "").strip()
    return seg or None


def _event_ids_for_object(events_v1: list[EventV1], object_name: str, t_ms: int | None) -> list[str]:
    target = _norm_label(object_name)
    if not target:
        return []
    out: list[str] = []
    for event in events_v1:
        obj = _norm_label(event.interaction_primary_object)
        if obj and (target in obj or obj in target):
            out.append(str(event.id))
            continue
        sig = event.interaction_signature if isinstance(event.interaction_signature, dict) else {}
        obj2 = _norm_label(sig.get("active_object_top1", sig.get("active_object", "")))
        if obj2 and (target in obj2 or obj2 in target):
            out.append(str(event.id))
            continue
        if t_ms is not None:
            t_s = float(t_ms) / 1000.0
            if float(event.t0) <= t_s <= float(event.t1):
                out.append(str(event.id))
    return sorted(set(out))


def _score(acc: _Accum) -> float:
    seen_term = min(1.0, float(acc.seen_count) / 10.0)
    contact_term = min(1.0, float(acc.contact_count) / 5.0)
    score = 0.35 * seen_term + 0.45 * contact_term + 0.20 * float(acc.contact_score_max)
    return float(max(0.0, min(1.0, score)))


def build_object_memory_v0(
    *,
    perception: dict[str, Any] | None,
    events_v1: list[EventV1] | None = None,
    contact_threshold: float = 0.6,
) -> list[ObjectMemoryItemV0]:
    payload = perception if isinstance(perception, dict) else {}
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return []

    events = list(events_v1 or [])
    by_object: dict[str, _Accum] = {}

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        t_ms = _to_ms(frame.get("t", 0.0))
        objects = frame.get("objects", [])
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                label = _norm_label(item.get("label", ""))
                if not label:
                    continue
                acc = by_object.setdefault(label, _Accum(object_name=label))
                acc.last_seen_t_ms = max(int(acc.last_seen_t_ms), int(t_ms))
                acc.seen_count += 1

        contact = frame.get("contact", {})
        if not isinstance(contact, dict):
            continue
        active = contact.get("active")
        if not isinstance(active, dict):
            continue
        label = _norm_label(active.get("label", ""))
        if not label:
            continue
        try:
            c_score = float(active.get("score", contact.get("active_score", 0.0)))
        except Exception:
            c_score = 0.0
        if c_score < float(contact_threshold):
            continue
        acc = by_object.setdefault(label, _Accum(object_name=label))
        acc.last_contact_t_ms = max(int(acc.last_contact_t_ms or 0), int(t_ms))
        acc.last_seen_t_ms = max(int(acc.last_seen_t_ms), int(t_ms))
        acc.contact_count += 1
        acc.contact_score_max = max(float(acc.contact_score_max), float(c_score))

    out: list[ObjectMemoryItemV0] = []
    for name in sorted(by_object.keys()):
        acc = by_object[name]
        pivot_ms = int(acc.last_contact_t_ms) if acc.last_contact_t_ms is not None else int(acc.last_seen_t_ms)
        last_place_id = _map_place_id(events, pivot_ms)
        evidence_event_ids = _event_ids_for_object(events, name, pivot_ms)
        out.append(
            ObjectMemoryItemV0(
                object_name=str(name),
                last_seen_t_ms=int(acc.last_seen_t_ms),
                last_contact_t_ms=int(acc.last_contact_t_ms) if acc.last_contact_t_ms is not None else None,
                last_place_id=last_place_id,
                evidence_event_ids=evidence_event_ids,
                score=_score(acc),
                meta={
                    "seen_count": int(acc.seen_count),
                    "contact_count": int(acc.contact_count),
                    "contact_score_max": float(acc.contact_score_max),
                },
            )
        )
    return out

