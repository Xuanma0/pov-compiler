from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.repository import build_repo_chunks, deduplicate_chunks, select_chunks_for_query
from pov_compiler.repository.schema import RepoChunk
from pov_compiler.schemas import ContextSchema, DecisionPoint, Event, KeyClip, Output, Token


DEFAULT_BUDGET = {
    "max_events": 8,
    "max_highlights": 10,
    "max_decisions": 12,
    "max_tokens": 200,
    "decisions_min_gap_s": 2.0,
    "max_seconds": None,
    "max_repo_chunks": 16,
    "max_repo_chars": 6000,
    "max_repo_tokens": 200,
    "repo_strategy": "importance_greedy",
    "use_repo": False,
    "repo_read_policy": "budgeted_topk",
}


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _as_output(output_json: str | Path | dict[str, Any] | Output) -> Output:
    if isinstance(output_json, Output):
        return output_json
    if isinstance(output_json, (str, Path)):
        data = json.loads(Path(output_json).read_text(encoding="utf-8"))
    elif isinstance(output_json, dict):
        data = output_json
    else:
        raise TypeError("output_json must be Output, dict, or path")
    if hasattr(Output, "model_validate"):
        return Output.model_validate(data)  # type: ignore[attr-defined]
    return Output.parse_obj(data)


def _count_by_type(tokens: list[Token]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token.type] = counts.get(token.type, 0) + 1
    return counts


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _is_highlight_related(token: Token, highlights: list[dict[str, Any]]) -> bool:
    if token.type == "HIGHLIGHT":
        return True
    for hl in highlights:
        if _overlap(token.t0, token.t1, float(hl["t0"]), float(hl["t1"])):
            return True
    return False


def _is_decision_related(token: Token, decisions: list[dict[str, Any]]) -> bool:
    for decision in decisions:
        if _overlap(token.t0, token.t1, float(decision["t0"]), float(decision["t1"])):
            return True
    return False


def _token_priority(
    token: Token,
    highlights: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    event_ids: set[str],
) -> tuple[int, float]:
    if token.type == "HIGHLIGHT":
        base = 1000
    elif token.type.startswith("ATTENTION_"):
        base = 900
    elif token.type in ("EVENT_START", "EVENT_END"):
        base = 800
    elif token.type == "INTERACTION":
        base = 850
    elif token.type.startswith("MOTION_"):
        base = 400
    elif token.type == "SCENE_CHANGE":
        base = 300
    else:
        base = 200

    if _is_highlight_related(token, highlights):
        base += 80
    if _is_decision_related(token, decisions):
        base += 60
    if token.source_event in event_ids:
        base += 20
    return base, float(token.conf)


def _select_events(
    events: list[Any],
    highlights: list[dict[str, Any]],
    max_events: int,
    preferred_event_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    ordered_events = sorted(events, key=lambda e: (e.t0, e.t1))
    selected_ids: list[str] = []
    for event_id in preferred_event_ids or []:
        if event_id not in selected_ids:
            selected_ids.append(event_id)
    hl_event_ids = [hl["source_event"] for hl in highlights if isinstance(hl.get("source_event"), str)]
    for event_id in hl_event_ids:
        if event_id not in selected_ids:
            selected_ids.append(event_id)
    for event in ordered_events:
        if event.id not in selected_ids:
            selected_ids.append(event.id)

    if max_events > 0:
        selected_ids = selected_ids[:max_events]

    event_map = {event.id: event for event in ordered_events}
    summaries: list[dict[str, Any]] = []
    for event_id in selected_ids:
        event = event_map.get(event_id)
        if event is None:
            continue
        anchor_summary: dict[str, int] = {}
        evidence_summary: dict[str, int] = {}
        if hasattr(event, "anchors"):
            for anchor in event.anchors:
                anchor_summary[anchor.type] = anchor_summary.get(anchor.type, 0) + 1
        else:
            for evidence in getattr(event, "evidence", []):
                et = str(getattr(evidence, "type", ""))
                evidence_summary[et] = evidence_summary.get(et, 0) + 1
                if et == "anchor":
                    anchor_type = str(getattr(evidence, "source", {}).get("anchor_type", ""))
                    if anchor_type:
                        anchor_summary[anchor_type] = anchor_summary.get(anchor_type, 0) + 1

        label = str(getattr(event, "label", getattr(event, "meta", {}).get("label", "")))
        boundary_conf = float(getattr(event, "scores", {}).get("boundary_conf", 0.0))
        summaries.append(
            {
                "id": event.id,
                "t0": float(event.t0),
                "t1": float(event.t1),
                "boundary_conf": boundary_conf,
                "label": label,
                "anchor_summary": anchor_summary,
                "evidence_summary": evidence_summary,
            }
        )
    summaries.sort(key=lambda e: (e["t0"], e["t1"]))
    return summaries


def _select_highlights(highlights_input: list[KeyClip], max_highlights: int) -> list[dict[str, Any]]:
    highlights = sorted(
        highlights_input,
        key=lambda h: (float(h.conf), float(h.t1 - h.t0)),
        reverse=True,
    )
    if max_highlights > 0:
        highlights = highlights[:max_highlights]
    summaries: list[dict[str, Any]] = []
    for hl in highlights:
        anchor_types = hl.meta.get("anchor_types")
        if not isinstance(anchor_types, list):
            anchor_types = [hl.anchor_type]
        summaries.append(
            {
                "id": hl.id,
                "t0": float(hl.t0),
                "t1": float(hl.t1),
                "anchor_types": [str(x) for x in anchor_types],
                "conf": float(hl.conf),
                "source_event": hl.source_event,
            }
        )
    summaries.sort(key=lambda h: (h["t0"], h["t1"]))
    return summaries


def _filter_by_max_seconds(
    events: list[dict[str, Any]],
    highlights: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    tokens: list[Token],
    max_seconds: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[Token]]:
    if max_seconds is None:
        return events, highlights, decisions, tokens
    if max_seconds <= 0:
        return [], [], [], []
    events = [e for e in events if float(e["t0"]) <= max_seconds]
    highlights = [h for h in highlights if float(h["t0"]) <= max_seconds]
    decisions = [d for d in decisions if float(d["t0"]) <= max_seconds]
    tokens = [t for t in tokens if float(t.t0) <= max_seconds]
    return events, highlights, decisions, tokens


def _select_tokens(
    tokens: list[Token],
    mode: str,
    highlights: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    event_ids: set[str],
    max_tokens: int,
) -> list[Token]:
    if max_tokens <= 0:
        return []

    if mode == "timeline":
        candidate = [
            token
            for token in tokens
            if token.type in ("EVENT_START", "EVENT_END", "SCENE_CHANGE")
            or token.type.startswith("ATTENTION_")
        ]
    elif mode == "highlights":
        candidate = [
            token
            for token in tokens
            if _is_highlight_related(token, highlights)
            or (token.type in ("EVENT_START", "EVENT_END") and token.source_event in event_ids)
        ]
    elif mode == "decisions":
        candidate = [
            token
            for token in tokens
            if _is_decision_related(token, decisions)
            or (token.type in ("EVENT_START", "EVENT_END") and token.source_event in event_ids)
        ]
    else:
        candidate = list(tokens)

    if not candidate:
        candidate = list(tokens)

    # Ensure key semantic cues survive hard clipping.
    essential = [t for t in candidate if t.type == "HIGHLIGHT" or t.type.startswith("ATTENTION_")]
    essential = sorted(
        essential,
        key=lambda t: (*_token_priority(t, highlights, decisions, event_ids), float(t.t0)),
        reverse=True,
    )

    selected: list[Token] = []
    seen = set()
    for token in essential:
        key = (token.id, token.t0, token.t1, token.type, token.source_event)
        if key in seen:
            continue
        selected.append(token)
        seen.add(key)
        if len(selected) >= max_tokens:
            selected = selected[:max_tokens]
            selected.sort(key=lambda t: (t.t0, t.t1, t.type))
            return selected

    remaining = sorted(
        candidate,
        key=lambda t: (*_token_priority(t, highlights, decisions, event_ids), float(t.t0)),
        reverse=True,
    )
    for token in remaining:
        key = (token.id, token.t0, token.t1, token.type, token.source_event)
        if key in seen:
            continue
        selected.append(token)
        seen.add(key)
        if len(selected) >= max_tokens:
            break

    selected.sort(key=lambda t: (t.t0, t.t1, t.type))
    return selected


def _normalize_selected_token_ids(selected_tokens: list[str | dict[str, Any] | Token] | None) -> set[str]:
    if not selected_tokens:
        return set()
    ids: set[str] = set()
    for item in selected_tokens:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, Token):
            ids.add(item.id)
        elif isinstance(item, dict):
            token_id = item.get("id")
            if token_id is not None:
                ids.add(str(token_id))
    return ids


def _normalize_selected_decision_ids(selected_decisions: list[str | dict[str, Any] | DecisionPoint] | None) -> set[str]:
    if not selected_decisions:
        return set()
    ids: set[str] = set()
    for item in selected_decisions:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, DecisionPoint):
            ids.add(item.id)
        elif isinstance(item, dict):
            decision_id = item.get("id")
            if decision_id is not None:
                ids.add(str(decision_id))
    return ids


def _decision_summary(decision: DecisionPoint) -> dict[str, Any]:
    constraints_summary = [
        {"type": str(c.get("type", "")), "score": float(c.get("score", 0.0))}
        for c in decision.constraints
    ]
    alternatives_summary = [alt.action_type for alt in decision.alternatives]
    return {
        "id": decision.id,
        "t": float(decision.t),
        "t0": float(decision.t0),
        "t1": float(decision.t1),
        "source_event": decision.source_event,
        "source_highlight": decision.source_highlight,
        "action": {"type": str(decision.action.get("type", "")), "conf": float(decision.action.get("conf", 0.0))},
        "outcome": {"type": str(decision.outcome.get("type", "")), "conf": float(decision.outcome.get("conf", 0.0))},
        "constraints_summary": constraints_summary,
        "alternatives_summary": alternatives_summary,
        "conf": float(decision.conf),
    }


def _select_decisions(
    decisions: list[DecisionPoint],
    max_decisions: int,
    min_gap_s: float,
) -> list[dict[str, Any]]:
    if max_decisions <= 0:
        return []
    ranked = sorted(
        decisions,
        key=lambda d: (1 if d.source_highlight else 0, float(d.conf)),
        reverse=True,
    )
    selected: list[DecisionPoint] = []
    for decision in ranked:
        if min_gap_s > 0 and any(abs(decision.t - picked.t) < min_gap_s for picked in selected):
            continue
        selected.append(decision)
        if len(selected) >= max_decisions:
            break
    if len(selected) < max_decisions:
        selected_ids = {d.id for d in selected}
        for decision in ranked:
            if decision.id in selected_ids:
                continue
            selected.append(decision)
            selected_ids.add(decision.id)
            if len(selected) >= max_decisions:
                break
    selected.sort(key=lambda d: (d.t, d.t0, d.t1))
    return [_decision_summary(d) for d in selected]


def _load_repo_chunks(output: Output, repo_cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    repository = dict(output.repository or {})
    chunks_payload = repository.get("chunks", [])
    chunks: list[dict[str, Any]] = []
    if isinstance(chunks_payload, list):
        for item in chunks_payload:
            if isinstance(item, dict):
                chunks.append(dict(item))
    if chunks:
        return chunks
    cfg = dict(repo_cfg or {})
    dedup_cfg = dict(cfg.get("dedup", {}))
    built = build_repo_chunks(output, cfg=cfg)
    deduped = deduplicate_chunks(built, cfg=dedup_cfg)
    return [_model_dump(chunk) for chunk in deduped]


def summarize_repo_selection(selection_trace: dict[str, Any] | None) -> dict[str, Any]:
    trace = dict(selection_trace or {})
    chunk_ids = trace.get("selected_chunk_ids_time_sorted", trace.get("selected_chunk_ids", []))
    if not isinstance(chunk_ids, list):
        chunk_ids = []
    by_level_raw = trace.get("selected_breakdown_by_level", {})
    by_level: dict[str, int] = {}
    if isinstance(by_level_raw, dict):
        for key, value in by_level_raw.items():
            try:
                by_level[str(key)] = int(value)
            except Exception:
                continue
    dropped_raw = trace.get("dropped_topN", [])
    dropped_count = len(dropped_raw) if isinstance(dropped_raw, list) else 0
    return {
        "policy_name": str(trace.get("policy_name", "")),
        "policy_hash": str(trace.get("policy_hash", "")),
        "selected_chunks": int(len(chunk_ids)),
        "by_level": dict(sorted(by_level.items())),
        "dropped_topN_count": int(dropped_count),
    }


def build_context(
    output_json: str | Path | dict[str, Any] | Output,
    mode: str = "highlights",
    budget: dict[str, Any] | None = None,
    selected_events: list[str] | None = None,
    selected_highlights: list[str] | None = None,
    selected_tokens: list[str | dict[str, Any] | Token] | None = None,
    selected_decisions: list[str | dict[str, Any] | DecisionPoint] | None = None,
    query_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in {"timeline", "highlights", "decisions", "full", "repo_only", "events_plus_repo"}:
        raise ValueError("mode must be one of: timeline, highlights, decisions, full, repo_only, events_plus_repo")

    output = ensure_events_v1(_as_output(output_json))

    merged_budget = dict(DEFAULT_BUDGET)
    if budget:
        merged_budget.update(budget)

    max_events = int(merged_budget.get("max_events", 8))
    max_highlights = int(merged_budget.get("max_highlights", 10))
    max_decisions = int(merged_budget.get("max_decisions", 12))
    max_tokens = int(merged_budget.get("max_tokens", 200))
    decisions_min_gap_s = float(merged_budget.get("decisions_min_gap_s", 2.0))
    raw_max_seconds = merged_budget.get("max_seconds", None)
    max_seconds = None if raw_max_seconds in (None, "", "none") else float(raw_max_seconds)

    events_pool = list(output.events_v1) if output.events_v1 else (list(output.events) + list(output.events_v0))
    highlights_pool = list(output.highlights)
    decisions_pool = list(output.decision_points)
    tokens_pool = list(output.token_codec.tokens)

    if selected_events:
        event_set = set(selected_events)
        events_pool = [event for event in events_pool if event.id in event_set]
    if selected_highlights:
        highlight_set = set(selected_highlights)
        highlights_pool = [hl for hl in highlights_pool if hl.id in highlight_set]
    selected_token_ids = _normalize_selected_token_ids(selected_tokens)
    if selected_token_ids:
        tokens_pool = [token for token in tokens_pool if token.id in selected_token_ids]
    selected_decision_ids = _normalize_selected_decision_ids(selected_decisions)
    if selected_decision_ids:
        decisions_pool = [decision for decision in decisions_pool if decision.id in selected_decision_ids]

    core_mode = "full" if mode == "events_plus_repo" else mode

    if core_mode == "timeline":
        decisions = []
    elif core_mode in {"decisions", "full"} or selected_decision_ids:
        decisions = _select_decisions(
            decisions_pool,
            max_decisions=max_decisions,
            min_gap_s=decisions_min_gap_s,
        )
    else:
        decisions = []

    if core_mode == "timeline":
        highlights = []
    elif core_mode == "decisions":
        decision_highlight_ids = {d["source_highlight"] for d in decisions if d.get("source_highlight")}
        if decision_highlight_ids:
            highlights_source = [hl for hl in highlights_pool if hl.id in decision_highlight_ids]
        else:
            highlights_source = highlights_pool
        highlights = _select_highlights(highlights_source, max_highlights=max_highlights)
    else:
        highlights = _select_highlights(highlights_pool, max_highlights=max_highlights)

    preferred_event_ids = [d["source_event"] for d in decisions if d.get("source_event")]
    events = _select_events(
        events_pool,
        highlights=highlights,
        max_events=max_events,
        preferred_event_ids=preferred_event_ids,
    )

    all_tokens = list(tokens_pool)
    events, highlights, decisions, all_tokens = _filter_by_max_seconds(
        events=events,
        highlights=highlights,
        decisions=decisions,
        tokens=all_tokens,
        max_seconds=max_seconds,
    )

    event_ids = {e["id"] for e in events}
    selected_tokens_final = _select_tokens(
        tokens=all_tokens,
        mode=core_mode,
        highlights=highlights,
        decisions=decisions,
        event_ids=event_ids,
        max_tokens=max_tokens,
    )

    use_repo = bool(merged_budget.get("use_repo", False)) or mode in {"repo_only", "events_plus_repo"}
    repo_cfg = dict((output.repository or {}).get("cfg", {}) if isinstance(output.repository, dict) else {})
    repo_chunks = _load_repo_chunks(output, repo_cfg=repo_cfg) if use_repo else []
    repo_selected: list[dict[str, Any]] = []
    repo_selection_trace: dict[str, Any] = {}
    if use_repo and repo_chunks:
        repo_models = [
            RepoChunk.model_validate(chunk) if hasattr(RepoChunk, "model_validate") else RepoChunk.parse_obj(chunk)
            for chunk in repo_chunks
        ]
        repo_query = str((budget or {}).get("repo_query", ""))
        if not repo_query and query_info and str(query_info.get("query", "")).strip():
            repo_query = str(query_info.get("query", "")).strip()
        repo_query_hints: dict[str, Any] = {}
        if isinstance(query_info, dict):
            hints_raw = query_info.get("query_hints", None)
            if isinstance(hints_raw, dict):
                repo_query_hints.update(dict(hints_raw))
            derived_raw = query_info.get("derived_constraints", None)
            if isinstance(derived_raw, dict):
                repo_query_hints["derived_constraints"] = dict(derived_raw)
            chain_meta_raw = query_info.get("chain_meta", None)
            if isinstance(chain_meta_raw, dict):
                repo_query_hints["chain_meta"] = dict(chain_meta_raw)
        repo_selected_models, repo_selection_trace = select_chunks_for_query(
            repo_models,
            query=repo_query,
            budget={
                "max_repo_chunks": int(merged_budget.get("max_repo_chunks", 16)),
                "max_repo_chars": merged_budget.get("max_repo_chars", 6000),
                "max_repo_tokens": int(merged_budget.get("max_repo_tokens", 200)),
                "max_seconds": max_seconds,
                "repo_strategy": str(merged_budget.get("repo_strategy", "importance_greedy")),
            },
            cfg={
                "strategy": str(merged_budget.get("repo_strategy", "importance_greedy")),
                "read_policy": {
                    "name": str(merged_budget.get("repo_read_policy", merged_budget.get("repo_strategy", "importance_greedy")))
                },
            },
            query_info=query_info,
            query_hints=repo_query_hints if repo_query_hints else None,
            return_trace=True,
        )
        repo_selected = [_model_dump(chunk) for chunk in repo_selected_models]
    repo_trace = {
        "mode": mode,
        "use_repo": use_repo,
        "budget_used": {
            "max_repo_chunks": int(merged_budget.get("max_repo_chunks", 16)),
            "max_repo_chars": merged_budget.get("max_repo_chars", 6000),
            "max_repo_tokens": int(merged_budget.get("max_repo_tokens", 200)),
            "max_seconds": max_seconds,
            "repo_strategy": str(merged_budget.get("repo_strategy", "importance_greedy")),
            "repo_read_policy": str(merged_budget.get("repo_read_policy", merged_budget.get("repo_strategy", "importance_greedy"))),
        },
        "repo_before": len(repo_chunks),
        "repo_after": len(repo_selected),
        "repo_chars_after": int(sum(len(str(c.get("text", ""))) for c in repo_selected)),
        "selection_trace": repo_selection_trace,
    }

    if mode == "repo_only":
        events = []
        highlights = []
        decisions = []
        selected_tokens_final = []

    context = ContextSchema(
        video_id=output.video_id,
        meta=dict(output.meta),
        stats=dict(output.stats),
        mode=mode,
        budget={
            "max_events": max_events,
            "max_highlights": max_highlights,
            "max_decisions": max_decisions,
            "max_tokens": max_tokens,
            "decisions_min_gap_s": decisions_min_gap_s,
            "max_seconds": max_seconds,
            "use_repo": use_repo,
            "max_repo_chunks": int(merged_budget.get("max_repo_chunks", 16)),
            "max_repo_chars": merged_budget.get("max_repo_chars", 6000),
            "max_repo_tokens": int(merged_budget.get("max_repo_tokens", 200)),
            "repo_strategy": str(merged_budget.get("repo_strategy", "importance_greedy")),
            "repo_read_policy": str(merged_budget.get("repo_read_policy", merged_budget.get("repo_strategy", "importance_greedy"))),
        },
        events=events,
        highlights=highlights,
        decision_points=decisions,
        tokens=selected_tokens_final,
        token_stats={
            "before": len(all_tokens),
            "after": len(selected_tokens_final),
            "by_type_before": _count_by_type(all_tokens),
            "by_type_after": _count_by_type(selected_tokens_final),
        },
        repo_chunks=repo_selected,
        repo_trace=repo_trace,
    )
    return _model_dump(context)
