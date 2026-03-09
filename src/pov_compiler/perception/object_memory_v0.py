from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pov_compiler.schemas import EventV1, ObjectMemoryItemV0


@dataclass
class _Accum:
    object_name: str
    last_seen_t_ms: int = 0
    last_contact_t_ms: int | None = None
    last_tracked_t_ms: int | None = None
    seen_count: int = 0
    contact_count: int = 0
    contact_score_max: float = 0.0
    persistence_frame_count: int = 0
    persistence_score_max: float = 0.0
    mask_area_max: float = 0.0
    persistent_track_ids: set[str] = field(default_factory=set)


_LABEL_ALIASES: dict[str, str] = {
    "cellphone": "cell phone",
    "cell_phone": "cell phone",
    "mobile_phone": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
}


def _norm_label(value: Any, *, enable_alias_merge: bool = True) -> str:
    label = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if not label:
        return ""
    normalized = " ".join(label.split())
    if enable_alias_merge:
        return _LABEL_ALIASES.get(normalized, normalized)
    return normalized


def _to_ms(sec: Any) -> int:
    try:
        return int(round(float(sec) * 1000.0))
    except Exception:
        return 0


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


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


def _score(acc: _Accum, *, logic_variant: str, persistence_min_frames: int) -> float:
    seen_term = min(1.0, float(acc.seen_count) / 10.0)
    contact_term = min(1.0, float(acc.contact_count) / 5.0)
    score = 0.35 * seen_term + 0.45 * contact_term + 0.20 * float(acc.contact_score_max)
    if logic_variant == "persistence_v1":
        persistence_term = min(1.0, float(acc.persistence_frame_count) / float(max(1, persistence_min_frames * 2)))
        persistence_score = max(persistence_term, float(acc.persistence_score_max))
        score = 0.22 * seen_term + 0.33 * contact_term + 0.20 * float(acc.contact_score_max) + 0.25 * persistence_score
        if acc.persistence_frame_count >= max(1, persistence_min_frames) and acc.last_contact_t_ms is None:
            score = max(score, 0.25 + 0.35 * persistence_score)
    return float(max(0.0, min(1.0, score)))


def build_object_memory_v0(
    *,
    perception: dict[str, Any] | None,
    events_v1: list[EventV1] | None = None,
    contact_threshold: float = 0.6,
    logic_variant: str = "current",
    persistence_min_frames: int = 2,
    persistence_score_min: float = 0.5,
    enable_alias_merge: bool = True,
) -> list[ObjectMemoryItemV0]:
    payload = perception if isinstance(perception, dict) else {}
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return []

    events = list(events_v1 or [])
    by_object: dict[str, _Accum] = {}
    resolved_logic = str(logic_variant or "current").strip().lower() or "current"

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        t_ms = _to_ms(frame.get("t", 0.0))
        objects = frame.get("objects", [])
        if isinstance(objects, list):
            for item in objects:
                if not isinstance(item, dict):
                    continue
                label = _norm_label(item.get("label", ""), enable_alias_merge=enable_alias_merge)
                if not label:
                    continue
                acc = by_object.setdefault(label, _Accum(object_name=label))
                acc.last_seen_t_ms = max(int(acc.last_seen_t_ms), int(t_ms))
                acc.seen_count += 1
                if resolved_logic == "persistence_v1":
                    track_id = str(item.get("track_id", "")).strip()
                    persistence_count = int(_to_float(item.get("persistence_count")) or 0)
                    persistence_score = float(_to_float(item.get("persistence_score")) or 0.0)
                    mask_area = float(_to_float(item.get("mask_area")) or 0.0)
                    persistent_flag = bool(item.get("persistent", False))
                    if track_id and (
                        persistent_flag
                        or persistence_count >= max(1, int(persistence_min_frames))
                        or persistence_score >= float(persistence_score_min)
                    ):
                        acc.last_tracked_t_ms = max(int(acc.last_tracked_t_ms or 0), int(t_ms))
                        acc.persistence_frame_count += 1
                        acc.persistence_score_max = max(float(acc.persistence_score_max), float(persistence_score))
                        acc.mask_area_max = max(float(acc.mask_area_max), float(mask_area))
                        acc.persistent_track_ids.add(track_id)

        contact = frame.get("contact", {})
        if not isinstance(contact, dict):
            continue
        active = contact.get("active")
        if not isinstance(active, dict):
            continue
        label = _norm_label(active.get("label", ""), enable_alias_merge=enable_alias_merge)
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
        pivot_ms = (
            int(acc.last_contact_t_ms)
            if acc.last_contact_t_ms is not None
            else int(acc.last_tracked_t_ms or acc.last_seen_t_ms)
        )
        last_place_id = _map_place_id(events, pivot_ms)
        evidence_event_ids = _event_ids_for_object(events, name, pivot_ms)
        persistence_backed = bool(acc.persistence_frame_count >= max(1, int(persistence_min_frames)) or acc.persistent_track_ids)
        out.append(
            ObjectMemoryItemV0(
                object_name=str(name),
                last_seen_t_ms=int(acc.last_seen_t_ms),
                last_contact_t_ms=int(acc.last_contact_t_ms) if acc.last_contact_t_ms is not None else None,
                last_place_id=last_place_id,
                evidence_event_ids=evidence_event_ids,
                score=_score(acc, logic_variant=resolved_logic, persistence_min_frames=int(persistence_min_frames)),
                meta={
                    "seen_count": int(acc.seen_count),
                    "contact_count": int(acc.contact_count),
                    "contact_score_max": float(acc.contact_score_max),
                    "logic_variant": resolved_logic,
                    "last_tracked_t_ms": int(acc.last_tracked_t_ms) if acc.last_tracked_t_ms is not None else None,
                    "persistence_frame_count": int(acc.persistence_frame_count),
                    "persistence_score_max": float(acc.persistence_score_max),
                    "persistent_track_ids": sorted(acc.persistent_track_ids),
                    "persistence_backed": persistence_backed,
                    "mask_area_max": float(acc.mask_area_max),
                },
            )
        )
    return out
