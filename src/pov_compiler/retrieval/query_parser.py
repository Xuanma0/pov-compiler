from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedQuery:
    raw: str
    time_range: tuple[float, float] | None = None
    token_types: list[str] = field(default_factory=list)
    anchor_types: list[str] = field(default_factory=list)
    decision_types: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    event_labels: list[str] = field(default_factory=list)
    contact_min: float | None = None
    place: str | None = None
    place_segment_ids: list[str] = field(default_factory=list)
    interaction_min: float | None = None
    interaction_object: str | None = None
    object_name: str | None = None
    lost_object: str | None = None
    object_last_seen: str | None = None
    which: str | None = None
    prefer_contact: bool = False
    text: str | None = None
    top_k: int | None = None
    mode: str | None = None
    budget_overrides: dict[str, Any] = field(default_factory=dict)
    filters_applied: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


_TIME_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_time_range(value: str) -> tuple[float, float]:
    match = _TIME_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid time range: {value}")
    t0 = float(match.group(1))
    t1 = float(match.group(2))
    if t0 > t1:
        t0, t1 = t1, t0
    return t0, t1


def _warn(parsed: ParsedQuery, msg: str) -> None:
    parsed.parse_warnings.append(str(msg))


def parse_query(query: str) -> ParsedQuery:
    parsed = ParsedQuery(raw=str(query))
    if not query or not str(query).strip():
        return parsed

    try:
        parts = shlex.split(query)
    except ValueError:
        parts = str(query).split()

    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        value = value.strip()
        if not value:
            continue

        if key == "time":
            try:
                parsed.time_range = _parse_time_range(value)
                parsed.filters_applied.append("time")
            except Exception:
                _warn(parsed, f"invalid_time={value}")
        elif key == "token":
            parsed.token_types = [x.upper() for x in _parse_csv(value)]
            if parsed.token_types:
                parsed.filters_applied.append("token")
        elif key == "anchor":
            parsed.anchor_types = [x.lower() for x in _parse_csv(value)]
            if parsed.anchor_types:
                parsed.filters_applied.append("anchor")
        elif key == "decision":
            parsed.decision_types = [x.upper() for x in _parse_csv(value)]
            if parsed.decision_types:
                parsed.filters_applied.append("decision")
        elif key == "event":
            parsed.event_ids = _parse_csv(value)
            if parsed.event_ids:
                parsed.filters_applied.append("event")
        elif key in {"event_label", "label"}:
            parsed.event_labels = [x.strip().lower() for x in _parse_csv(value)]
            if parsed.event_labels:
                parsed.filters_applied.append("event_label")
        elif key == "contact_min":
            try:
                parsed.contact_min = float(value)
                parsed.filters_applied.append("contact_min")
            except Exception:
                _warn(parsed, f"invalid_contact_min={value}")
        elif key == "place":
            place = str(value).strip().lower()
            if place in {"first", "last", "any"}:
                parsed.place = place
                parsed.filters_applied.append("place")
            else:
                _warn(parsed, f"invalid_place={value}")
        elif key in {"place_segment_id", "place_segment"}:
            parsed.place_segment_ids = [x.strip() for x in _parse_csv(value) if x.strip()]
            if parsed.place_segment_ids:
                parsed.filters_applied.append("place_segment_id")
        elif key == "interaction_min":
            try:
                parsed.interaction_min = float(value)
                parsed.filters_applied.append("interaction_min")
            except Exception:
                _warn(parsed, f"invalid_interaction_min={value}")
        elif key == "interaction_object":
            parsed.interaction_object = str(value).strip().lower()
            if parsed.interaction_object:
                parsed.filters_applied.append("interaction_object")
        elif key == "which":
            which = str(value).strip().lower()
            if which in {"first", "last"}:
                parsed.which = which
                parsed.filters_applied.append("which")
            else:
                _warn(parsed, f"invalid_which={value}")
        elif key == "lost_object":
            obj = str(value).strip().lower()
            if obj:
                parsed.lost_object = obj
                parsed.object_name = obj
                parsed.interaction_object = obj
                parsed.which = "last"
                parsed.prefer_contact = True
                parsed.filters_applied.append("lost_object")
        elif key == "object_last_seen":
            obj = str(value).strip().lower()
            if obj:
                parsed.object_last_seen = obj
                parsed.object_name = obj
                parsed.interaction_object = obj
                parsed.which = "last"
                parsed.filters_applied.append("object_last_seen")
        elif key == "object":
            obj = str(value).strip().lower()
            if obj:
                parsed.object_name = obj
                parsed.interaction_object = obj
                parsed.filters_applied.append("object")
        elif key == "text":
            parsed.text = value
            parsed.filters_applied.append("text")
        elif key == "top_k":
            try:
                parsed.top_k = int(value)
            except Exception:
                _warn(parsed, f"invalid_top_k={value}")
        elif key == "mode":
            mode = value.lower()
            if mode not in {"timeline", "highlights", "decisions", "full", "repo_only", "events_plus_repo"}:
                _warn(parsed, f"invalid_mode={value}")
            else:
                parsed.mode = mode
        elif key in {"max_tokens", "max_highlights", "max_events", "max_decisions"}:
            try:
                parsed.budget_overrides[key] = int(value)
            except Exception:
                _warn(parsed, f"invalid_{key}={value}")
        elif key == "max_seconds":
            try:
                parsed.budget_overrides[key] = float(value)
            except Exception:
                _warn(parsed, f"invalid_max_seconds={value}")

    return parsed
