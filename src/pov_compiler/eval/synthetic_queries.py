from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pov_compiler.schemas import DecisionPoint, Output, Token


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) <= min(a1, b1)


@dataclass
class SyntheticQuery:
    query: str
    kind: str
    top_k: int
    relevant_highlights: list[str] = field(default_factory=list)
    relevant_decisions: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _query_time_window(
    output: Output,
    t0: float,
    t1: float,
    top_k: int,
) -> SyntheticQuery:
    relevant_highlights = [hl.id for hl in output.highlights if _overlap(hl.t0, hl.t1, t0, t1)]
    relevant_decisions = [dp.id for dp in output.decision_points if _overlap(dp.t0, dp.t1, t0, t1)]
    return SyntheticQuery(
        query=f"time={t0:.3f}-{t1:.3f} top_k={top_k}",
        kind="time",
        top_k=top_k,
        relevant_highlights=relevant_highlights,
        relevant_decisions=relevant_decisions,
        meta={"t0": t0, "t1": t1},
    )


def _highlight_anchor_types(output: Output) -> set[str]:
    types: set[str] = set()
    for hl in output.highlights:
        anchor_types = hl.meta.get("anchor_types")
        if isinstance(anchor_types, list):
            types.update(str(x).lower() for x in anchor_types)
        else:
            types.add(str(hl.anchor_type).lower())
    return types


def _decision_anchor_types(decision: DecisionPoint) -> set[str]:
    anchor_types = decision.trigger.get("anchor_types")
    if isinstance(anchor_types, list):
        return {str(x).lower() for x in anchor_types}
    anchor_type = decision.trigger.get("anchor_type")
    if anchor_type is None:
        return set()
    return {str(anchor_type).lower()}


def _query_anchor_type(output: Output, anchor_type: str, top_k: int) -> SyntheticQuery:
    anchor_type = anchor_type.lower()
    relevant_highlights: list[str] = []
    for hl in output.highlights:
        hl_types = hl.meta.get("anchor_types")
        if isinstance(hl_types, list):
            types = {str(x).lower() for x in hl_types}
        else:
            types = {str(hl.anchor_type).lower()}
        if anchor_type in types:
            relevant_highlights.append(hl.id)

    relevant_decisions = [dp.id for dp in output.decision_points if anchor_type in _decision_anchor_types(dp)]
    return SyntheticQuery(
        query=f"anchor={anchor_type} top_k={top_k}",
        kind="anchor",
        top_k=top_k,
        relevant_highlights=relevant_highlights,
        relevant_decisions=relevant_decisions,
        meta={"anchor_type": anchor_type},
    )


def _query_token_type(output: Output, token_type: str, top_k: int) -> SyntheticQuery:
    token_type = token_type.upper()
    matched_tokens: list[Token] = [token for token in output.token_codec.tokens if token.type == token_type]
    relevant_highlights: list[str] = []
    for hl in output.highlights:
        if any(_overlap(hl.t0, hl.t1, token.t0, token.t1) for token in matched_tokens):
            relevant_highlights.append(hl.id)

    relevant_decisions: list[str] = []
    for dp in output.decision_points:
        trigger_ids = dp.trigger.get("token_ids", [])
        evidence_ids = dp.state.get("evidence", {}).get("token_ids", [])
        decision_token_ids = set()
        if isinstance(trigger_ids, list):
            decision_token_ids.update(str(x) for x in trigger_ids)
        if isinstance(evidence_ids, list):
            decision_token_ids.update(str(x) for x in evidence_ids)
        if any(token.id in decision_token_ids for token in matched_tokens):
            relevant_decisions.append(dp.id)
        elif any(_overlap(dp.t0, dp.t1, token.t0, token.t1) for token in matched_tokens):
            relevant_decisions.append(dp.id)

    return SyntheticQuery(
        query=f"token={token_type} top_k={top_k}",
        kind="token",
        top_k=top_k,
        relevant_highlights=relevant_highlights,
        relevant_decisions=relevant_decisions,
        meta={"token_type": token_type},
    )


def generate_synthetic_queries(
    output: Output,
    num_time_queries: int = 10,
    time_window_s: float = 8.0,
    default_top_k: int = 6,
    max_anchor_queries: int = 4,
    max_token_queries: int = 4,
    seed: int | None = None,
) -> list[SyntheticQuery]:
    queries: list[SyntheticQuery] = []

    duration_s = float(output.meta.get("duration_s", 0.0))
    rng_seed = seed
    if rng_seed is None:
        rng_seed = int(abs(hash(output.video_id)) % (2**32))
    rng = np.random.default_rng(rng_seed)

    if duration_s > 0 and num_time_queries > 0:
        window = max(0.5, float(time_window_s))
        attempts = 0
        max_attempts = max(20, num_time_queries * 8)
        while len([q for q in queries if q.kind == "time"]) < num_time_queries and attempts < max_attempts:
            attempts += 1
            center = float(rng.uniform(0.0, duration_s))
            t0 = max(0.0, center - 0.5 * window)
            t1 = min(duration_s, center + 0.5 * window)
            q = _query_time_window(output, t0, t1, top_k=default_top_k)
            if q.relevant_highlights or q.relevant_decisions:
                queries.append(q)
        # fallback deterministic windows if sparse
        if len([q for q in queries if q.kind == "time"]) < num_time_queries:
            for center in np.linspace(0, duration_s, num=max(2, num_time_queries)):
                t0 = max(0.0, float(center) - 0.5 * window)
                t1 = min(duration_s, float(center) + 0.5 * window)
                q = _query_time_window(output, t0, t1, top_k=default_top_k)
                if q.relevant_highlights or q.relevant_decisions:
                    queries.append(q)
                if len([x for x in queries if x.kind == "time"]) >= num_time_queries:
                    break

    anchor_types = sorted(_highlight_anchor_types(output))
    if anchor_types:
        if len(anchor_types) > max_anchor_queries:
            picks = rng.choice(len(anchor_types), size=max_anchor_queries, replace=False)
            anchor_types = [anchor_types[int(i)] for i in sorted(picks.tolist())]
        for anchor_type in anchor_types:
            q = _query_anchor_type(output, anchor_type=anchor_type, top_k=default_top_k)
            if q.relevant_highlights or q.relevant_decisions:
                queries.append(q)

    token_types = sorted({token.type for token in output.token_codec.tokens})
    preferred = [token_type for token_type in token_types if token_type not in {"EVENT_START", "EVENT_END"}]
    token_types = preferred if preferred else token_types
    if token_types:
        if len(token_types) > max_token_queries:
            picks = rng.choice(len(token_types), size=max_token_queries, replace=False)
            token_types = [token_types[int(i)] for i in sorted(picks.tolist())]
        for token_type in token_types:
            q = _query_token_type(output, token_type=token_type, top_k=default_top_k)
            if q.relevant_highlights or q.relevant_decisions:
                queries.append(q)

    return queries
