from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Budget:
    max_total_s: float | None = None
    max_tokens: int | None = None
    max_decisions: int | None = None

    def normalized(self) -> "Budget":
        total = None if self.max_total_s is None else max(0.0, float(self.max_total_s))
        tokens = None if self.max_tokens is None else max(0, int(self.max_tokens))
        decisions = None if self.max_decisions is None else max(0, int(self.max_decisions))
        return Budget(max_total_s=total, max_tokens=tokens, max_decisions=decisions)


_CONTEXT_NAMES = {"pov.highlight", "pov.token", "pov.decision"}
_BASE_NAMES = {"pov.event"}
_NAME_PRIORITY = {"pov.highlight": 0, "pov.decision": 1, "pov.token": 2}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _payload_span_ms(event: dict[str, Any]) -> tuple[int, int]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        t0 = _as_int(payload.get("t0_ms"), 0)
        t1 = _as_int(payload.get("t1_ms"), t0)
    else:
        t0 = 0
        t1 = 0
    if t1 < t0:
        t1 = t0
    return t0, t1


def _merge_union_duration_s(intervals_ms: list[tuple[int, int]]) -> float:
    if not intervals_ms:
        return 0.0
    points = sorted(intervals_ms)
    merged: list[tuple[int, int]] = []
    cur0, cur1 = points[0]
    for t0, t1 in points[1:]:
        if t0 <= cur1:
            cur1 = max(cur1, t1)
            continue
        merged.append((cur0, cur1))
        cur0, cur1 = t0, t1
    merged.append((cur0, cur1))
    total_ms = sum(max(0, b - a) for a, b in merged)
    return float(total_ms) / 1000.0


def _incremental_union_delta_ms(intervals: list[tuple[int, int]], candidate: tuple[int, int]) -> int:
    c0, c1 = candidate
    if c1 <= c0:
        return 0
    overlap = 0
    for t0, t1 in intervals:
        if t1 <= c0 or c1 <= t0:
            continue
        overlap += max(0, min(c1, t1) - max(c0, t0))
    return max(0, (c1 - c0) - overlap)


def _count_by_name(events: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in events:
        name = str(row.get("name", ""))
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


def apply_budget(
    events: list[dict[str, Any]],
    budget: Budget,
    *,
    keep_base_events: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budget_n = budget.normalized()
    rows = list(events or [])

    before_counts = _count_by_name(rows)
    before_total = len(rows)
    before_context_intervals = [_payload_span_ms(x) for x in rows if str(x.get("name", "")) in _CONTEXT_NAMES]
    before_kept_duration_s = _merge_union_duration_s(before_context_intervals)

    base_events: list[dict[str, Any]] = []
    context_events: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name", ""))
        if name in _CONTEXT_NAMES:
            context_events.append(row)
            continue
        if keep_base_events and (name in _BASE_NAMES or str(row.get("category", "")) == "scenario"):
            base_events.append(row)
            continue
        base_events.append(row)

    context_sorted = sorted(
        context_events,
        key=lambda x: (
            _as_int(x.get("tsMs"), 0),
            _NAME_PRIORITY.get(str(x.get("name", "")), 99),
            _payload_span_ms(x)[0],
            _payload_span_ms(x)[1],
            str(x.get("name", "")),
        ),
    )

    kept_context: list[dict[str, Any]] = []
    kept_intervals: list[tuple[int, int]] = []
    kept_context_ms = 0
    kept_tokens = 0
    kept_decisions = 0

    for row in context_sorted:
        name = str(row.get("name", ""))
        if name == "pov.token" and budget_n.max_tokens is not None and kept_tokens >= int(budget_n.max_tokens):
            continue
        if name == "pov.decision" and budget_n.max_decisions is not None and kept_decisions >= int(budget_n.max_decisions):
            continue

        span = _payload_span_ms(row)
        delta_ms = _incremental_union_delta_ms(kept_intervals, span)
        if budget_n.max_total_s is not None:
            proposed_s = float(kept_context_ms + delta_ms) / 1000.0
            if proposed_s > float(budget_n.max_total_s) + 1e-9:
                continue

        kept_context.append(row)
        kept_intervals.append(span)
        kept_context_ms += int(delta_ms)
        if name == "pov.token":
            kept_tokens += 1
        elif name == "pov.decision":
            kept_decisions += 1

    kept = base_events + kept_context
    kept_sorted = sorted(
        kept,
        key=lambda x: (
            _as_int(x.get("tsMs"), 0),
            _NAME_PRIORITY.get(str(x.get("name", "")), 99),
            _payload_span_ms(x)[0],
            _payload_span_ms(x)[1],
            str(x.get("name", "")),
        ),
    )

    after_counts = _count_by_name(kept_sorted)
    after_total = len(kept_sorted)
    after_context_intervals = [_payload_span_ms(x) for x in kept_sorted if str(x.get("name", "")) in _CONTEXT_NAMES]
    kept_duration_s = _merge_union_duration_s(after_context_intervals)

    stats = {
        "budget_used": {
            "max_total_s": budget_n.max_total_s,
            "max_tokens": budget_n.max_tokens,
            "max_decisions": budget_n.max_decisions,
            "keep_base_events": bool(keep_base_events),
        },
        "before_counts": before_counts,
        "after_counts": after_counts,
        "before_total": int(before_total),
        "after_total": int(after_total),
        "before_kept_duration_s": float(before_kept_duration_s),
        "kept_duration_s": float(kept_duration_s),
        "compression_ratio": float(before_total) / float(max(1, after_total)),
    }
    return kept_sorted, stats

