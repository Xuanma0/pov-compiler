from __future__ import annotations

from typing import Any

from pov_compiler.repository.schema import RepoChunk
from pov_compiler.retrieval.query_parser import parse_query


def _query_bonus(chunk: RepoChunk, query: str) -> float:
    parsed = parse_query(query)
    tags = {str(x).lower() for x in chunk.tags}
    text = str(chunk.text).lower()
    bonus = 0.0

    for anchor in parsed.anchor_types:
        if str(anchor).lower() in tags or str(anchor).lower() in text:
            bonus += 0.25
    for token in parsed.token_types:
        t = str(token).lower()
        if t in tags or t in text:
            bonus += 0.2
    for decision in parsed.decision_types:
        d = str(decision).lower()
        if d in tags or d in text:
            bonus += 0.25
    if parsed.time_range is not None:
        q0, q1 = parsed.time_range
        if max(float(chunk.t0), float(q0)) <= min(float(chunk.t1), float(q1)):
            bonus += 0.15
    return float(bonus)


def _within_seconds(chunk: RepoChunk, max_seconds: float | None) -> bool:
    if max_seconds is None:
        return True
    return float(chunk.t0) <= float(max_seconds)


def _fits_budget(
    *,
    chosen: list[RepoChunk],
    candidate: RepoChunk,
    max_chunks: int,
    max_chars: int | None,
    max_seconds: float | None,
) -> bool:
    if len(chosen) >= max_chunks:
        return False
    if not _within_seconds(candidate, max_seconds):
        return False
    if max_chars is not None:
        used_chars = sum(len(c.text) for c in chosen)
        if used_chars + len(candidate.text) > max_chars:
            return False
    return True


def select_chunks_for_query(
    repo_chunks: list[RepoChunk],
    query: str,
    budget: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[RepoChunk]:
    budget = dict(budget or {})
    cfg = dict(cfg or {})
    strategy = str(cfg.get("strategy", budget.get("repo_strategy", "importance_greedy"))).lower()
    max_chunks = int(budget.get("max_repo_chunks", cfg.get("max_chunks", 16)))
    max_chars_val = budget.get("max_repo_chars", cfg.get("max_chars", None))
    max_chars = None if max_chars_val in (None, "", "none") else int(max_chars_val)
    max_seconds_val = budget.get("max_seconds", cfg.get("max_seconds", None))
    max_seconds = None if max_seconds_val in (None, "", "none") else float(max_seconds_val)

    chunks = list(repo_chunks)
    if strategy == "recency_greedy":
        ranked = sorted(
            chunks,
            key=lambda c: (float(c.t1), float(c.importance), str(c.id)),
            reverse=True,
        )
    else:
        ranked = sorted(
            chunks,
            key=lambda c: (float(c.importance) + _query_bonus(c, query), float(c.t1), str(c.id)),
            reverse=True,
        )

    chosen: list[RepoChunk] = []
    for chunk in ranked:
        if not _fits_budget(
            chosen=chosen,
            candidate=chunk,
            max_chunks=max_chunks,
            max_chars=max_chars,
            max_seconds=max_seconds,
        ):
            continue
        chosen.append(chunk)

    chosen.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
    return chosen
