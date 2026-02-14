from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.repository.policy import build_write_policy
from pov_compiler.repository.schema import RepoChunk
from pov_compiler.schemas import Event, EventV1, Output


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(float(a0), float(b0)) <= min(float(a1), float(b1))


def _token_type_counts(output: Output, t0: float, t1: float) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in output.token_codec.tokens:
        if _overlap(token.t0, token.t1, t0, t1):
            counts[token.type] = counts.get(token.type, 0) + 1
    return counts


def _decision_count(output: Output, t0: float, t1: float) -> int:
    return sum(1 for dp in output.decision_points if _overlap(dp.t0, dp.t1, t0, t1))


def _highlight_count(output: Output, t0: float, t1: float) -> int:
    return sum(1 for hl in output.highlights if _overlap(hl.t0, hl.t1, t0, t1))


def _duration(output: Output) -> float:
    duration_s = float(output.meta.get("duration_s", 0.0) or 0.0)
    if duration_s > 0:
        return duration_s
    all_t = [float(e.t1) for e in output.events_v1] + [float(e.t1) for e in output.events] + [float(h.t1) for h in output.highlights]
    return max(all_t) if all_t else 0.0


def _event_text(event: EventV1) -> str:
    boundary = float((event.scores or {}).get("boundary_conf", 0.0))
    contact_peak = float((event.scores or {}).get("contact_peak", 0.0))
    place = str(event.place_segment_id or "")
    interaction = float(event.interaction_score or 0.0)
    evidence_types: dict[str, int] = {}
    for item in list(event.evidence or []):
        et = str(getattr(item, "type", "")).strip() or "unknown"
        evidence_types[et] = evidence_types.get(et, 0) + 1
    evidence_str = ", ".join(f"{k}:{v}" for k, v in sorted(evidence_types.items())) if evidence_types else "none"
    return (
        f"Event [{float(event.t0):.1f}-{float(event.t1):.1f}s] label={str(event.label or 'event')}; "
        f"boundary_conf={boundary:.2f}; contact_peak={contact_peak:.2f}; interaction={interaction:.2f}; "
        f"place={place or 'none'}; evidence={evidence_str}."
    )


def _event_importance(event: EventV1) -> float:
    scores = dict(event.scores or {})
    boundary = float(scores.get("boundary_conf", 0.0))
    contact_peak = float(scores.get("contact_peak", 0.0))
    interaction = float(event.interaction_score or 0.0)
    label = str(event.label or "").lower()
    label_bonus = 0.3 if "interaction" in label else 0.1 if "navigation" in label else 0.0
    decision_density = sum(1 for item in list(event.evidence or []) if str(getattr(item, "type", "")) == "decision")
    value = (
        0.25 * _clamp01(boundary)
        + 0.2 * _clamp01(contact_peak)
        + 0.25 * _clamp01(interaction)
        + 0.1 * min(1.0, decision_density / 3.0)
        + label_bonus
    )
    return _clamp01(value)


def _window_text(output: Output, t0: float, t1: float) -> str:
    token_counts = _token_type_counts(output, t0, t1)
    token_top = sorted(token_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:4]
    token_str = ", ".join(f"{k}:{v}" for k, v in token_top) if token_top else "none"
    return (
        f"Window [{t0:.1f}-{t1:.1f}s]; highlights={_highlight_count(output, t0, t1)}; "
        f"decisions={_decision_count(output, t0, t1)}; tokens={token_str}."
    )


def _window_importance(output: Output, t0: float, t1: float) -> float:
    decision_count = _decision_count(output, t0, t1)
    highlight_count = _highlight_count(output, t0, t1)
    token_counts = _token_type_counts(output, t0, t1)
    attention = sum(v for k, v in token_counts.items() if str(k).startswith("ATTENTION_"))
    scene = int(token_counts.get("SCENE_CHANGE", 0))
    return _clamp01(
        0.3 * min(1.0, highlight_count / 4.0)
        + 0.25 * min(1.0, decision_count / 4.0)
        + 0.2 * min(1.0, attention / 4.0)
        + 0.25 * min(1.0, scene / 2.0)
    )


def _segment_points(output: Output, duration_s: float) -> list[float]:
    points = {0.0, float(max(0.0, duration_s))}
    for event in output.events_v1:
        points.add(float(event.t0))
        points.add(float(event.t1))
    for token in output.token_codec.tokens:
        if str(token.type).upper() == "SCENE_CHANGE":
            points.add(float(token.t0))
            points.add(float(token.t1))
    return sorted(points)


def _to_event_v1(event: Event) -> EventV1:
    label = str(event.meta.get("label", "event")) if isinstance(event.meta, dict) else "event"
    return EventV1(
        id=str(event.id),
        t0=float(event.t0),
        t1=float(event.t1),
        label=label,
        source_event_ids=[str(event.id)],
        evidence=[],
        retrieval_hints=[],
        scores=dict(event.scores or {}),
        meta=dict(event.meta or {}),
    )


def _ms(s: float) -> int:
    return int(round(float(s) * 1000.0))


def _mk_chunk(
    *,
    chunk_id: str,
    level: str,
    t0: float,
    t1: float,
    text: str,
    importance: float,
    source_ids: list[str],
    tags: list[str],
    score_fields: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
) -> RepoChunk:
    return RepoChunk(
        id=chunk_id,
        chunk_id=chunk_id,
        scale=level,
        level=level,
        t0=float(t0),
        t1=float(t1),
        t0_ms=_ms(float(t0)),
        t1_ms=_ms(float(t1)),
        text=str(text),
        importance=float(_clamp01(importance)),
        source_ids=[str(x) for x in source_ids],
        tags=[str(x) for x in tags if str(x)],
        score_fields={str(k): float(v) for k, v in dict(score_fields or {}).items()},
        meta=dict(meta or {}),
    )


def _finalize_chunks(chunks: list[RepoChunk]) -> list[RepoChunk]:
    out = sorted(chunks, key=lambda c: (float(c.t0), float(c.t1), str(c.level), str(c.id)))
    for i, chunk in enumerate(out, start=1):
        if not chunk.id.startswith("repo_"):
            cid = f"repo_{chunk.level}_{i:05d}"
            chunk.id = cid
            chunk.chunk_id = cid
        chunk.meta = dict(chunk.meta or {})
        chunk.meta.setdefault("token_est", max(1, int(round(len(str(chunk.text)) / 4.0))))
    return out


def build_repo_chunks(output: Output, cfg: dict[str, Any] | None = None) -> list[RepoChunk]:
    cfg = dict(cfg or {})
    output = ensure_events_v1(output)
    duration_s = _duration(output)

    scales_cfg = dict(cfg.get("scales", {}))
    enable_event = bool(scales_cfg.get("event", True))
    enable_decision = bool(scales_cfg.get("decision", True))
    enable_place = bool(scales_cfg.get("place", True))
    # backward-compat levels from RepoV0
    enable_window = bool(scales_cfg.get("window", True))
    enable_segment = bool(scales_cfg.get("segment", True))
    window_s = float(cfg.get("window_s", 30.0))
    min_segment_s = float(cfg.get("min_segment_s", 5.0))

    events_source: list[EventV1] = list(output.events_v1) if output.events_v1 else [_to_event_v1(e) for e in output.events]
    chunks: list[RepoChunk] = []

    # L1: event-level.
    if enable_event:
        for event in sorted(events_source, key=lambda e: (float(e.t0), float(e.t1), str(e.id))):
            place_id = str(event.place_segment_id or "")
            tags = ["event", str(event.label or "event").lower()]
            if place_id:
                tags.append(f"place:{place_id}")
            interaction_obj = str(event.interaction_primary_object or "").strip().lower()
            if interaction_obj:
                tags.append(f"obj:{interaction_obj}")
            chunks.append(
                _mk_chunk(
                    chunk_id=f"repo_event_{len(chunks)+1:05d}",
                    level="event",
                    t0=float(event.t0),
                    t1=float(event.t1),
                    text=_event_text(event),
                    importance=_event_importance(event),
                    source_ids=[str(event.id)] + [str(x) for x in (event.source_event_ids or [])],
                    tags=sorted(set(tags)),
                    score_fields={
                        "boundary_conf": float((event.scores or {}).get("boundary_conf", 0.0)),
                        "interaction_score": float(event.interaction_score or 0.0),
                    },
                    meta={"label": str(event.label or ""), "place_segment_id": place_id},
                )
            )

    # L2: decision-level.
    if enable_decision:
        for dp in sorted(output.decision_points, key=lambda d: (float(d.t0), float(d.t1), str(d.id))):
            action_type = str(dp.action.get("type", "")) if isinstance(dp.action, dict) else ""
            outcome_type = str(dp.outcome.get("type", "")) if isinstance(dp.outcome, dict) else ""
            constraints = list(dp.constraints or [])
            tags = ["decision"]
            if action_type:
                tags.append(f"action:{action_type.lower()}")
            if outcome_type:
                tags.append(f"outcome:{outcome_type.lower()}")
            text = (
                f"Decision [{float(dp.t0):.1f}-{float(dp.t1):.1f}s] action={action_type or 'unknown'}; "
                f"outcome={outcome_type or 'unknown'}; constraints={len(constraints)}; conf={float(dp.conf):.2f}."
            )
            importance = _clamp01(
                0.5 * float(dp.conf)
                + 0.25 * min(1.0, len(constraints) / 3.0)
                + 0.25 * (1.0 if dp.source_highlight else 0.0)
            )
            chunks.append(
                _mk_chunk(
                    chunk_id=f"repo_decision_{len(chunks)+1:05d}",
                    level="decision",
                    t0=float(dp.t0),
                    t1=float(dp.t1),
                    text=text,
                    importance=importance,
                    source_ids=[str(dp.id), str(dp.source_event)] + ([str(dp.source_highlight)] if dp.source_highlight else []),
                    tags=sorted(set(tags)),
                    score_fields={"decision_conf": float(dp.conf)},
                    meta={"action_type": action_type, "outcome_type": outcome_type, "source_event": str(dp.source_event)},
                )
            )

    # L3: place + interaction-level.
    if enable_place:
        by_place: dict[str, list[EventV1]] = defaultdict(list)
        for event in events_source:
            pid = str(event.place_segment_id or "")
            if not pid:
                pid = f"place_auto_{int(math.floor(float(event.t0) // max(1.0, window_s))):03d}"
            by_place[pid].append(event)
        for pid in sorted(by_place.keys()):
            group = sorted(by_place[pid], key=lambda e: (float(e.t0), float(e.t1), str(e.id)))
            t0 = min(float(e.t0) for e in group)
            t1 = max(float(e.t1) for e in group)
            interaction_scores = [float(e.interaction_score or 0.0) for e in group]
            interaction_mean = float(sum(interaction_scores) / max(1, len(interaction_scores)))
            objects: dict[str, int] = {}
            for event in group:
                obj = str(event.interaction_primary_object or "").strip().lower()
                if obj:
                    objects[obj] = objects.get(obj, 0) + 1
            primary_obj = ""
            if objects:
                primary_obj = sorted(objects.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            text = (
                f"Place segment [{t0:.1f}-{t1:.1f}s] id={pid}; events={len(group)}; "
                f"interaction_mean={interaction_mean:.2f}; primary_object={primary_obj or 'none'}."
            )
            chunks.append(
                _mk_chunk(
                    chunk_id=f"repo_place_{len(chunks)+1:05d}",
                    level="place",
                    t0=t0,
                    t1=t1,
                    text=text,
                    importance=_clamp01(0.6 * interaction_mean + 0.4 * min(1.0, len(group) / 4.0)),
                    source_ids=[str(e.id) for e in group],
                    tags=["place", f"place:{pid}"] + ([f"obj:{primary_obj}"] if primary_obj else []),
                    score_fields={"interaction_score": interaction_mean},
                    meta={"place_segment_id": pid, "primary_object": primary_obj, "event_count": len(group)},
                )
            )

    # backward-compatible window chunks
    if enable_window and duration_s > 0.0 and window_s > 0.0:
        n = int(math.ceil(duration_s / window_s))
        for i in range(n):
            t0 = float(i * window_s)
            t1 = float(min(duration_s, (i + 1) * window_s))
            if t1 <= t0:
                continue
            source_events = [e.id for e in events_source if _overlap(e.t0, e.t1, t0, t1)]
            tags = ["window"]
            if _decision_count(output, t0, t1) > 0:
                tags.append("decision-rich")
            if _highlight_count(output, t0, t1) > 0:
                tags.append("highlight-rich")
            chunks.append(
                _mk_chunk(
                    chunk_id=f"repo_window_{len(chunks)+1:05d}",
                    level="window",
                    t0=t0,
                    t1=t1,
                    text=_window_text(output, t0, t1),
                    importance=_window_importance(output, t0, t1),
                    source_ids=[str(x) for x in sorted(source_events)],
                    tags=sorted(set(tags)),
                    score_fields={"decision_count": float(_decision_count(output, t0, t1))},
                    meta={"window_s": window_s},
                )
            )

    # backward-compatible segment chunks
    if enable_segment and duration_s > 0.0:
        points = _segment_points(output, duration_s)
        merged: list[tuple[float, float]] = []
        cur0 = points[0] if points else 0.0
        for p in points[1:]:
            t0 = float(cur0)
            t1 = float(p)
            if t1 - t0 < min_segment_s:
                continue
            merged.append((t0, t1))
            cur0 = p
        if cur0 < duration_s:
            if merged and duration_s - merged[-1][1] < min_segment_s:
                last0, _ = merged[-1]
                merged[-1] = (last0, duration_s)
            elif duration_s - cur0 >= min_segment_s:
                merged.append((float(cur0), float(duration_s)))
        for t0, t1 in merged:
            source_events = [e.id for e in events_source if _overlap(e.t0, e.t1, t0, t1)]
            scene_changes = sum(
                1 for tok in output.token_codec.tokens if str(tok.type).upper() == "SCENE_CHANGE" and _overlap(tok.t0, tok.t1, t0, t1)
            )
            text = (
                f"Segment [{t0:.1f}-{t1:.1f}s]; events={len(source_events)}; "
                f"scene_changes={scene_changes}; decisions={_decision_count(output, t0, t1)}."
            )
            importance = _clamp01(
                0.35 * min(1.0, len(source_events) / 4.0)
                + 0.35 * min(1.0, scene_changes / 2.0)
                + 0.3 * min(1.0, _decision_count(output, t0, t1) / 3.0)
            )
            chunks.append(
                _mk_chunk(
                    chunk_id=f"repo_segment_{len(chunks)+1:05d}",
                    level="segment",
                    t0=t0,
                    t1=t1,
                    text=text,
                    importance=importance,
                    source_ids=[str(x) for x in sorted(source_events)],
                    tags=["segment", "scene-boundary" if scene_changes > 0 else "scene-stable"],
                    score_fields={"scene_changes": float(scene_changes)},
                    meta={"scene_changes": int(scene_changes)},
                )
            )

    chunks = _finalize_chunks(chunks)
    write_cfg = cfg.get("write_policy", {})
    if isinstance(write_cfg, str):
        write_cfg = {"name": write_cfg}
    if not isinstance(write_cfg, dict):
        write_cfg = {}
    # Backward-compat: default keeps all chunks.
    write_cfg = {"name": "fixed_interval", "chunk_step_s": 0.0, **write_cfg}
    write_policy = build_write_policy(write_cfg)
    written = write_policy.write(chunks, signals={"output_meta": dict(output.meta or {})}, budget_cfg=dict(cfg.get("budget", {})))
    return _finalize_chunks(written)

