from __future__ import annotations

import math
from typing import Any

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.repository.schema import RepoChunk
from pov_compiler.schemas import Output


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


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


def _event_text(event: Any) -> str:
    label = str(getattr(event, "label", getattr(event, "meta", {}).get("label", ""))).strip() or "event"
    scores = dict(getattr(event, "scores", {}) or {})
    boundary = float(scores.get("boundary_conf", 0.0))
    contact_peak = float(scores.get("contact_peak", 0.0))
    evidence = list(getattr(event, "evidence", []) or [])
    evidence_types: dict[str, int] = {}
    for item in evidence:
        et = str(getattr(item, "type", "")).strip() or "unknown"
        evidence_types[et] = evidence_types.get(et, 0) + 1
    evidence_str = ", ".join(f"{k}:{v}" for k, v in sorted(evidence_types.items())) if evidence_types else "none"
    return (
        f"Event [{float(event.t0):.1f}-{float(event.t1):.1f}s] label={label}; "
        f"boundary_conf={boundary:.2f}; contact_peak={contact_peak:.2f}; evidence={evidence_str}."
    )


def _event_importance(event: Any) -> float:
    scores = dict(getattr(event, "scores", {}) or {})
    boundary = float(scores.get("boundary_conf", 0.0))
    contact_peak = float(scores.get("contact_peak", 0.0))
    evidence = list(getattr(event, "evidence", []) or [])
    decision_density = sum(1 for item in evidence if str(getattr(item, "type", "")) == "decision")
    label = str(getattr(event, "label", "")).lower()
    label_bonus = 0.3 if "interaction" in label else 0.1 if "navigation" in label else 0.0
    value = 0.35 * _clamp01(boundary) + 0.35 * _clamp01(contact_peak) + 0.1 * min(1.0, decision_density / 3.0) + label_bonus
    return _clamp01(value)


def _window_text(output: Output, t0: float, t1: float) -> str:
    token_counts = _token_type_counts(output, t0, t1)
    token_top = sorted(token_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:4]
    token_str = ", ".join(f"{k}:{v}" for k, v in token_top) if token_top else "none"
    decision_count = _decision_count(output, t0, t1)
    highlight_count = _highlight_count(output, t0, t1)
    return (
        f"Window [{t0:.1f}-{t1:.1f}s]; highlights={highlight_count}; decisions={decision_count}; "
        f"tokens={token_str}."
    )


def _window_importance(output: Output, t0: float, t1: float) -> float:
    decision_count = _decision_count(output, t0, t1)
    highlight_count = _highlight_count(output, t0, t1)
    token_counts = _token_type_counts(output, t0, t1)
    attention = sum(v for k, v in token_counts.items() if str(k).startswith("ATTENTION_"))
    scene = int(token_counts.get("SCENE_CHANGE", 0))
    value = 0.3 * min(1.0, highlight_count / 4.0) + 0.25 * min(1.0, decision_count / 4.0) + 0.2 * min(1.0, attention / 4.0) + 0.25 * min(1.0, scene / 2.0)
    return _clamp01(value)


def _segment_points(output: Output, duration_s: float) -> list[float]:
    points = {0.0, float(max(0.0, duration_s))}
    for event in output.events_v1:
        points.add(float(event.t0))
        points.add(float(event.t1))
    for token in output.token_codec.tokens:
        if str(token.type).upper() == "SCENE_CHANGE":
            points.add(float(token.t0))
            points.add(float(token.t1))
    arr = sorted(points)
    return arr


def build_repo_chunks(output: Output, cfg: dict[str, Any] | None = None) -> list[RepoChunk]:
    cfg = dict(cfg or {})
    output = ensure_events_v1(output)
    duration_s = float(output.meta.get("duration_s", 0.0) or 0.0)
    if duration_s <= 0.0:
        all_t = [float(e.t1) for e in output.events_v1] + [float(h.t1) for h in output.highlights]
        duration_s = max(all_t) if all_t else 0.0

    scales_cfg = dict(cfg.get("scales", {}))
    enable_event = bool(scales_cfg.get("event", True))
    enable_window = bool(scales_cfg.get("window", True))
    enable_segment = bool(scales_cfg.get("segment", True))
    window_s = float(cfg.get("window_s", 30.0))
    min_segment_s = float(cfg.get("min_segment_s", 5.0))

    chunks: list[RepoChunk] = []
    idx = 1

    if enable_event:
        for event in sorted(output.events_v1, key=lambda e: (float(e.t0), float(e.t1), str(e.id))):
            evidence_types = sorted({str(getattr(item, "type", "")) for item in getattr(event, "evidence", []) if str(getattr(item, "type", ""))})
            tags = ["event", str(event.label).lower()] + [f"evidence:{x}" for x in evidence_types]
            chunk = RepoChunk(
                id=f"repo_event_{idx:05d}",
                scale="event",
                t0=float(event.t0),
                t1=float(event.t1),
                text=_event_text(event),
                importance=_event_importance(event),
                source_ids=[str(event.id)],
                tags=sorted({t for t in tags if t}),
                meta={
                    "label": str(event.label),
                    "boundary_conf": float(event.scores.get("boundary_conf", 0.0)),
                    "contact_peak": float(event.scores.get("contact_peak", 0.0)),
                },
            )
            chunks.append(chunk)
            idx += 1

    if enable_window and duration_s > 0.0 and window_s > 0.0:
        n = int(math.ceil(duration_s / window_s))
        for i in range(n):
            t0 = float(i * window_s)
            t1 = float(min(duration_s, (i + 1) * window_s))
            if t1 <= t0:
                continue
            source_events = [e.id for e in output.events_v1 if _overlap(e.t0, e.t1, t0, t1)]
            tags = ["window"]
            if _decision_count(output, t0, t1) > 0:
                tags.append("decision-rich")
            if _highlight_count(output, t0, t1) > 0:
                tags.append("highlight-rich")
            chunk = RepoChunk(
                id=f"repo_window_{idx:05d}",
                scale="window",
                t0=t0,
                t1=t1,
                text=_window_text(output, t0, t1),
                importance=_window_importance(output, t0, t1),
                source_ids=[str(x) for x in sorted(source_events)],
                tags=sorted(set(tags)),
                meta={"window_s": window_s},
            )
            chunks.append(chunk)
            idx += 1

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
            source_events = [e.id for e in output.events_v1 if _overlap(e.t0, e.t1, t0, t1)]
            scene_changes = sum(
                1
                for tok in output.token_codec.tokens
                if str(tok.type).upper() == "SCENE_CHANGE" and _overlap(tok.t0, tok.t1, t0, t1)
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
            chunk = RepoChunk(
                id=f"repo_segment_{idx:05d}",
                scale="segment",
                t0=float(t0),
                t1=float(t1),
                text=text,
                importance=importance,
                source_ids=[str(x) for x in sorted(source_events)],
                tags=["segment", "scene-boundary" if scene_changes > 0 else "scene-stable"],
                meta={"scene_changes": int(scene_changes)},
            )
            chunks.append(chunk)
            idx += 1

    chunks.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.scale), str(c.id)))
    return chunks
