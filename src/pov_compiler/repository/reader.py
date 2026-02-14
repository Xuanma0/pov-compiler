from __future__ import annotations

from typing import Any

from pov_compiler.repository.policy import build_read_policy
from pov_compiler.repository.schema import RepoChunk


def select_chunks_for_query(
    repo_chunks: list[RepoChunk],
    query: str,
    budget: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[RepoChunk]:
    budget = dict(budget or {})
    cfg = dict(cfg or {})
    policy_cfg = dict(cfg.get("read_policy", {}))

    # Backward-compatible aliases.
    if not policy_cfg:
        strategy = str(cfg.get("strategy", budget.get("repo_strategy", "importance_greedy"))).lower()
        if strategy in {"recency_greedy", "recency"}:
            max_chunks = int(budget.get("max_repo_chunks", 16))
            max_tokens = int(budget.get("max_tokens", budget.get("max_repo_tokens", 200)))
            max_seconds_raw = budget.get("max_seconds", None)
            max_seconds = None if max_seconds_raw in (None, "", "none") else float(max_seconds_raw)
            ranked = sorted(list(repo_chunks), key=lambda c: (float(c.t1), float(c.importance), str(c.id)), reverse=True)
            selected = []
            used_tokens = 0
            for chunk in ranked:
                if len(selected) >= max_chunks:
                    break
                if max_seconds is not None and float(chunk.t0) > max_seconds:
                    continue
                token_est = int(chunk.meta.get("token_est", max(1, int(round(len(str(chunk.text)) / 4.0)))))
                if used_tokens + token_est > max_tokens:
                    continue
                selected.append(chunk)
                used_tokens += token_est
            return sorted(selected, key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
        policy_cfg = {
            "name": "budgeted_topk",
            "max_chunks": int(budget.get("max_repo_chunks", 16)),
            "max_tokens": int(budget.get("max_tokens", budget.get("max_repo_tokens", 200))),
            "max_seconds": budget.get("max_seconds", None),
        }

    policy = build_read_policy(policy_cfg)
    selected = policy.select(list(repo_chunks), query=query, budget_cfg=budget)
    selected.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
    return selected
