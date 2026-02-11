from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.eval.metrics import merge_intervals, overlap_duration
from pov_compiler.schemas import DecisionPoint, KeyClip, Output, Token


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(float(a0), float(b0)) <= min(float(a1), float(b1))


def _first_nonempty(values: list[str]) -> str:
    for value in values:
        value = str(value).strip()
        if value:
            return value
    return ""


_TOP_K_PATTERN = re.compile(r"(?:^|\s)top_k=([0-9]+)(?:\s|$)")


@dataclass
class FixedQuery:
    qid: str
    type: str
    query: str
    top_k: int = 6
    time: dict[str, float] | None = None
    relevant: dict[str, list[str]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "FixedQuery":
        relevant: dict[str, list[str]] = {}
        for key in ("highlights", "events", "decisions", "tokens"):
            values = self.relevant.get(key, [])
            if isinstance(values, list):
                uniq = sorted({str(x) for x in values if str(x)})
            else:
                uniq = []
            relevant[key] = uniq
        time_payload: dict[str, float] | None = None
        if isinstance(self.time, dict) and "t0" in self.time and "t1" in self.time:
            t0 = _safe_float(self.time.get("t0", 0.0), 0.0)
            t1 = _safe_float(self.time.get("t1", 0.0), 0.0)
            if t0 > t1:
                t0, t1 = t1, t0
            time_payload = {"t0": t0, "t1": t1}
        top_k = max(1, _safe_int(self.top_k, 6))
        return FixedQuery(
            qid=str(self.qid),
            type=str(self.type),
            query=str(self.query),
            top_k=top_k,
            time=time_payload,
            relevant=relevant,
            meta=dict(self.meta),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.normalized()
        out = {
            "qid": payload.qid,
            "type": payload.type,
            "query": payload.query,
            "top_k": payload.top_k,
            "relevant": payload.relevant,
            "meta": payload.meta,
        }
        if payload.time is not None:
            out["time"] = payload.time
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FixedQuery":
        qid = str(payload.get("qid", ""))
        qtype = str(payload.get("type", "time"))
        query = str(payload.get("query", ""))
        top_k = payload.get("top_k")
        if top_k is None:
            match = _TOP_K_PATTERN.search(query)
            top_k = int(match.group(1)) if match else 6
        relevant = payload.get("relevant", {})
        if not isinstance(relevant, dict):
            relevant = {}
        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        time_payload = payload.get("time")
        if not isinstance(time_payload, dict):
            time_payload = None
        query_obj = cls(
            qid=qid,
            type=qtype,
            query=query,
            top_k=max(1, _safe_int(top_k, 6)),
            time=time_payload,
            relevant={
                "highlights": list(relevant.get("highlights", [])),
                "events": list(relevant.get("events", [])),
                "decisions": list(relevant.get("decisions", [])),
                "tokens": list(relevant.get("tokens", [])),
            },
            meta=dict(meta),
        )
        return query_obj.normalized()


def _collect_relevant_by_window(output: Output, t0: float, t1: float) -> dict[str, list[str]]:
    highlights = [hl.id for hl in output.highlights if _overlap(hl.t0, hl.t1, t0, t1)]
    events = [event.id for event in output.events if _overlap(event.t0, event.t1, t0, t1)]
    decisions = [dp.id for dp in output.decision_points if _overlap(dp.t0, dp.t1, t0, t1)]
    tokens = [token.id for token in output.token_codec.tokens if _overlap(token.t0, token.t1, t0, t1)]
    return {
        "highlights": sorted(set(highlights)),
        "events": sorted(set(events)),
        "decisions": sorted(set(decisions)),
        "tokens": sorted(set(tokens)),
    }


def _highlight_anchor_types(hl: KeyClip) -> set[str]:
    anchor_types = hl.meta.get("anchor_types")
    if isinstance(anchor_types, list):
        return {str(x).lower() for x in anchor_types if str(x)}
    return {str(hl.anchor_type).lower()}


def _decision_anchor_types(dp: DecisionPoint) -> set[str]:
    anchor_types = dp.trigger.get("anchor_types")
    if isinstance(anchor_types, list):
        return {str(x).lower() for x in anchor_types if str(x)}
    anchor_type = dp.trigger.get("anchor_type")
    if anchor_type is None:
        return set()
    return {str(anchor_type).lower()}


def _token_ids_for_decision(dp: DecisionPoint) -> set[str]:
    ids: set[str] = set()
    trigger_ids = dp.trigger.get("token_ids", [])
    if isinstance(trigger_ids, list):
        ids.update(str(x) for x in trigger_ids if str(x))
    evidence_ids = dp.state.get("evidence", {}).get("token_ids", [])
    if isinstance(evidence_ids, list):
        ids.update(str(x) for x in evidence_ids if str(x))
    outcome_ids = dp.outcome.get("evidence", {}).get("token_ids", [])
    if isinstance(outcome_ids, list):
        ids.update(str(x) for x in outcome_ids if str(x))
    return ids


def _query_from_time(t0: float, t1: float, top_k: int) -> str:
    return f"time={t0:.3f}-{t1:.3f} top_k={int(top_k)}"


def _window_overlap_ratio(t0: float, t1: float, merged_highlights: list[tuple[float, float]]) -> float:
    dur = max(1e-9, float(t1 - t0))
    overlap = overlap_duration(float(t0), float(t1), merged_highlights)
    return float(overlap / dur)


def _sample_choices(values: list[str], n: int, rng: np.random.Generator) -> list[str]:
    clean = [str(v) for v in values if str(v)]
    if n <= 0 or not clean:
        return []
    clean = sorted(set(clean))
    if len(clean) <= n:
        return clean
    picks = rng.choice(len(clean), size=n, replace=False)
    return [clean[int(i)] for i in sorted(picks.tolist())]


def generate_fixed_queries(
    output: Output,
    seed: int = 0,
    n_time: int = 10,
    n_anchor: int = 6,
    n_token: int = 10,
    n_decision: int = 10,
    n_hard_time: int = 10,
    time_window_s: float = 8.0,
    default_top_k: int = 6,
    hard_overlap_thresh: float = 0.05,
) -> list[FixedQuery]:
    duration_s = float(output.meta.get("duration_s", 0.0))
    if duration_s <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    window = max(0.5, float(time_window_s))
    top_k = max(1, int(default_top_k))
    merged_highlights = merge_intervals([(float(hl.t0), float(hl.t1)) for hl in output.highlights])

    queries: list[FixedQuery] = []

    def _append(query_type: str, query: str, relevant: dict[str, list[str]], time_payload: dict[str, float] | None, meta: dict[str, Any]) -> None:
        qid = f"q_{len(queries) + 1:06d}"
        if not any(relevant.get(key) for key in ("highlights", "events", "decisions", "tokens")):
            return
        queries.append(
            FixedQuery(
                qid=qid,
                type=query_type,
                query=query,
                top_k=top_k,
                time=time_payload,
                relevant=relevant,
                meta=meta,
            ).normalized()
        )

    # 1) time queries
    attempts = 0
    max_attempts = max(50, n_time * 10)
    while len([q for q in queries if q.type == "time"]) < n_time and attempts < max_attempts:
        attempts += 1
        center = float(rng.uniform(0.0, duration_s))
        t0 = max(0.0, center - 0.5 * window)
        t1 = min(duration_s, center + 0.5 * window)
        relevant = _collect_relevant_by_window(output, t0, t1)
        _append(
            query_type="time",
            query=_query_from_time(t0, t1, top_k=top_k),
            relevant=relevant,
            time_payload={"t0": t0, "t1": t1},
            meta={"window_s": float(t1 - t0)},
        )
    if len([q for q in queries if q.type == "time"]) < n_time:
        centers = np.linspace(0.0, duration_s, num=max(2, n_time))
        for center in centers:
            t0 = max(0.0, float(center) - 0.5 * window)
            t1 = min(duration_s, float(center) + 0.5 * window)
            relevant = _collect_relevant_by_window(output, t0, t1)
            _append(
                query_type="time",
                query=_query_from_time(t0, t1, top_k=top_k),
                relevant=relevant,
                time_payload={"t0": t0, "t1": t1},
                meta={"window_s": float(t1 - t0), "fallback": True},
            )
            if len([q for q in queries if q.type == "time"]) >= n_time:
                break

    # 2) anchor queries
    anchor_pool = sorted(
        {
            anchor_type
            for hl in output.highlights
            for anchor_type in _highlight_anchor_types(hl)
        }.union(
            {
                str(anchor.type).lower()
                for event in output.events
                for anchor in event.anchors
                if str(anchor.type)
            }
        )
    )
    for anchor_type in _sample_choices(anchor_pool, n_anchor, rng):
        relevant_highlights = [hl.id for hl in output.highlights if anchor_type in _highlight_anchor_types(hl)]
        relevant_events = [
            event.id
            for event in output.events
            if anchor_type in {str(anchor.type).lower() for anchor in event.anchors}
        ]
        relevant_decisions = [dp.id for dp in output.decision_points if anchor_type in _decision_anchor_types(dp)]
        relevant_tokens = [
            token.id
            for token in output.token_codec.tokens
            if (
                (anchor_type == "turn_head" and token.type == "ATTENTION_TURN_HEAD")
                or (anchor_type == "stop_look" and token.type == "ATTENTION_STOP_LOOK")
            )
        ]
        _append(
            query_type="anchor",
            query=f"anchor={anchor_type} top_k={top_k}",
            relevant={
                "highlights": sorted(set(relevant_highlights)),
                "events": sorted(set(relevant_events)),
                "decisions": sorted(set(relevant_decisions)),
                "tokens": sorted(set(relevant_tokens)),
            },
            time_payload=None,
            meta={"anchor_type": anchor_type},
        )

    # 3) token queries
    token_types = sorted({str(token.type).upper() for token in output.token_codec.tokens if str(token.type)})
    token_types = [x for x in token_types if x not in {"EVENT_START", "EVENT_END"}]
    for token_type in _sample_choices(token_types, n_token, rng):
        matched_tokens = [token for token in output.token_codec.tokens if str(token.type).upper() == token_type]
        relevant_tokens = [token.id for token in matched_tokens]
        relevant_events = sorted({token.source_event for token in matched_tokens if token.source_event})
        relevant_highlights = [
            hl.id
            for hl in output.highlights
            if any(_overlap(hl.t0, hl.t1, token.t0, token.t1) for token in matched_tokens)
        ]
        relevant_decisions: list[str] = []
        token_ids_set = set(relevant_tokens)
        for dp in output.decision_points:
            if token_ids_set.intersection(_token_ids_for_decision(dp)):
                relevant_decisions.append(dp.id)
            elif any(_overlap(dp.t0, dp.t1, token.t0, token.t1) for token in matched_tokens):
                relevant_decisions.append(dp.id)
        _append(
            query_type="token",
            query=f"token={token_type} top_k={top_k}",
            relevant={
                "highlights": sorted(set(relevant_highlights)),
                "events": sorted(set(relevant_events)),
                "decisions": sorted(set(relevant_decisions)),
                "tokens": sorted(set(relevant_tokens)),
            },
            time_payload=None,
            meta={"token_type": token_type},
        )

    # 4) decision queries
    decision_types = sorted(
        {
            str(dp.action.get("type", "")).upper()
            for dp in output.decision_points
            if str(dp.action.get("type", ""))
        }
    )
    for decision_type in _sample_choices(decision_types, n_decision, rng):
        matched_decisions = [
            dp
            for dp in output.decision_points
            if str(dp.action.get("type", "")).upper() == decision_type
        ]
        relevant_decisions = [dp.id for dp in matched_decisions]
        relevant_highlights = [dp.source_highlight for dp in matched_decisions if dp.source_highlight]
        relevant_events = [dp.source_event for dp in matched_decisions if dp.source_event]
        relevant_tokens: list[str] = []
        for dp in matched_decisions:
            relevant_tokens.extend(list(_token_ids_for_decision(dp)))
            relevant_tokens.extend(
                token.id
                for token in output.token_codec.tokens
                if _overlap(token.t0, token.t1, dp.t0, dp.t1)
            )
        _append(
            query_type="decision",
            query=f"decision={decision_type} top_k={top_k}",
            relevant={
                "highlights": sorted(set(str(x) for x in relevant_highlights if str(x))),
                "events": sorted(set(str(x) for x in relevant_events if str(x))),
                "decisions": sorted(set(relevant_decisions)),
                "tokens": sorted(set(str(x) for x in relevant_tokens if str(x))),
            },
            time_payload=None,
            meta={"decision_type": decision_type},
        )

    # 5) hard_time queries
    hard_candidates: list[tuple[str, str, float, str]] = []
    for token in output.token_codec.tokens:
        token_type = str(token.type).upper()
        if token_type in {"EVENT_START", "EVENT_END", "HIGHLIGHT"}:
            continue
        ratio = _window_overlap_ratio(float(token.t0), float(token.t1), merged_highlights)
        if ratio < hard_overlap_thresh:
            center = 0.5 * (float(token.t0) + float(token.t1))
            hard_candidates.append(("token", token_type, center, token.id))
    for dp in output.decision_points:
        action_type = str(dp.action.get("type", "")).upper()
        if not action_type:
            continue
        ratio = _window_overlap_ratio(float(dp.t0), float(dp.t1), merged_highlights)
        if ratio < hard_overlap_thresh:
            hard_candidates.append(("decision", action_type, float(dp.t), dp.id))

    if hard_candidates:
        idxs = np.arange(len(hard_candidates))
        rng.shuffle(idxs)
        hard_candidates = [hard_candidates[int(i)] for i in idxs.tolist()]

    hard_count = 0
    for source_kind, source_type, center, source_id in hard_candidates:
        if hard_count >= n_hard_time:
            break
        t0 = max(0.0, float(center) - 0.5 * window)
        t1 = min(duration_s, float(center) + 0.5 * window)
        if _window_overlap_ratio(t0, t1, merged_highlights) >= hard_overlap_thresh:
            continue
        relevant = _collect_relevant_by_window(output, t0, t1)
        if source_kind == "token":
            query = f"time={t0:.3f}-{t1:.3f} token={source_type} top_k={top_k}"
            relevant["tokens"] = sorted(
                {
                    token.id
                    for token in output.token_codec.tokens
                    if str(token.type).upper() == source_type and _overlap(token.t0, token.t1, t0, t1)
                }
            )
        else:
            query = f"time={t0:.3f}-{t1:.3f} decision={source_type} top_k={top_k}"
            relevant["decisions"] = sorted(
                {
                    dp.id
                    for dp in output.decision_points
                    if str(dp.action.get("type", "")).upper() == source_type and _overlap(dp.t0, dp.t1, t0, t1)
                }
            )
        # hard_time should focus on non-highlight evidence.
        if not relevant.get("events"):
            continue
        if not relevant.get("tokens") and not relevant.get("decisions"):
            continue
        _append(
            query_type="hard_time",
            query=query,
            relevant=relevant,
            time_payload={"t0": t0, "t1": t1},
            meta={
                "source_kind": source_kind,
                "source_type": source_type,
                "source_id": source_id,
                "highlight_overlap_ratio": _window_overlap_ratio(t0, t1, merged_highlights),
            },
        )
        hard_count += 1

    # fallback hard_time windows if sparse
    attempts = 0
    max_attempts = max(50, n_hard_time * 10)
    while hard_count < n_hard_time and attempts < max_attempts:
        attempts += 1
        center = float(rng.uniform(0.0, duration_s))
        t0 = max(0.0, center - 0.5 * window)
        t1 = min(duration_s, center + 0.5 * window)
        ratio = _window_overlap_ratio(t0, t1, merged_highlights)
        if ratio >= hard_overlap_thresh:
            continue
        in_window_tokens = [
            token
            for token in output.token_codec.tokens
            if _overlap(token.t0, token.t1, t0, t1) and str(token.type).upper() not in {"EVENT_START", "EVENT_END", "HIGHLIGHT"}
        ]
        if not in_window_tokens:
            continue
        token_type = _first_nonempty([str(in_window_tokens[0].type).upper()])
        if not token_type:
            continue
        relevant = _collect_relevant_by_window(output, t0, t1)
        relevant["tokens"] = sorted({token.id for token in in_window_tokens if str(token.type).upper() == token_type})
        if not relevant.get("events"):
            continue
        query = f"time={t0:.3f}-{t1:.3f} token={token_type} top_k={top_k}"
        _append(
            query_type="hard_time",
            query=query,
            relevant=relevant,
            time_payload={"t0": t0, "t1": t1},
            meta={"source_kind": "fallback_token", "source_type": token_type, "highlight_overlap_ratio": ratio},
        )
        hard_count += 1

    return [query.normalized() for query in queries]


def save_queries_jsonl(queries: list[FixedQuery], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(query.to_dict(), ensure_ascii=False) for query in queries]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_queries_jsonl(path: str | Path) -> list[FixedQuery]:
    in_path = Path(path)
    if not in_path.exists():
        return []
    lines = in_path.read_text(encoding="utf-8").splitlines()
    queries: list[FixedQuery] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            queries.append(FixedQuery.from_dict(payload))
    return queries
