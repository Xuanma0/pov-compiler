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
    text: str | None = None
    top_k: int | None = None
    mode: str | None = None
    budget_overrides: dict[str, Any] = field(default_factory=dict)
    filters_applied: list[str] = field(default_factory=list)


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
            parsed.time_range = _parse_time_range(value)
            parsed.filters_applied.append("time")
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
            parsed.contact_min = float(value)
            parsed.filters_applied.append("contact_min")
        elif key == "text":
            parsed.text = value
            parsed.filters_applied.append("text")
        elif key == "top_k":
            parsed.top_k = int(value)
        elif key == "mode":
            mode = value.lower()
            if mode not in {"timeline", "highlights", "decisions", "full"}:
                raise ValueError(f"Invalid mode: {value}")
            parsed.mode = mode
        elif key in {"max_tokens", "max_highlights", "max_events", "max_decisions"}:
            parsed.budget_overrides[key] = int(value)
        elif key == "max_seconds":
            parsed.budget_overrides[key] = float(value)

    return parsed
