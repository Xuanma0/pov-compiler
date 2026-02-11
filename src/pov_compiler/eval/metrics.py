from __future__ import annotations

from pathlib import Path
from typing import Any

from pov_compiler.eval.synthetic_queries import SyntheticQuery, generate_synthetic_queries
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Output


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned = [(float(min(a, b)), float(max(a, b))) for a, b in intervals if float(max(a, b) - min(a, b)) > 0]
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float]] = [cleaned[0]]
    for t0, t1 in cleaned[1:]:
        p0, p1 = merged[-1]
        if t0 <= p1:
            merged[-1] = (p0, max(p1, t1))
        else:
            merged.append((t0, t1))
    return merged


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, t1 - t0) for t0, t1 in intervals))


def overlap_duration(t0: float, t1: float, merged_intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    for a0, a1 in merged_intervals:
        lo = max(float(t0), float(a0))
        hi = min(float(t1), float(a1))
        if hi > lo:
            total += hi - lo
    return float(total)


def compute_coverage(output: Output) -> dict[str, float]:
    duration_s = float(output.meta.get("duration_s", 0.0))
    merged = merge_intervals([(float(hl.t0), float(hl.t1)) for hl in output.highlights])
    coverage_s = interval_duration(merged)
    coverage_ratio = float(coverage_s / duration_s) if duration_s > 1e-9 else 0.0
    return {
        "highlight_coverage_s": coverage_s,
        "time_coverage": coverage_s,
        "coverage_ratio": coverage_ratio,
    }


def compute_consistency(output: Output) -> dict[str, float]:
    merged = merge_intervals([(float(hl.t0), float(hl.t1)) for hl in output.highlights])

    token_weight_sum = 0.0
    token_overlap_sum = 0.0
    for token in output.token_codec.tokens:
        dur = max(0.001, float(token.t1 - token.t0))
        token_weight_sum += dur
        token_overlap_sum += overlap_duration(float(token.t0), float(token.t1), merged)
    token_ratio = float(token_overlap_sum / token_weight_sum) if token_weight_sum > 1e-9 else 0.0

    decision_weight_sum = 0.0
    decision_overlap_sum = 0.0
    supported = 0
    for dp in output.decision_points:
        dur = max(0.001, float(dp.t1 - dp.t0))
        decision_weight_sum += dur
        decision_overlap_sum += overlap_duration(float(dp.t0), float(dp.t1), merged)
        nearby_tokens = dp.state.get("nearby_tokens", [])
        trigger_token_ids = dp.trigger.get("token_ids", [])
        if (isinstance(nearby_tokens, list) and len(nearby_tokens) > 0) or (
            isinstance(trigger_token_ids, list) and len(trigger_token_ids) > 0
        ):
            supported += 1
    decision_ratio = float(decision_overlap_sum / decision_weight_sum) if decision_weight_sum > 1e-9 else 0.0
    supported_ratio = float(supported / len(output.decision_points)) if output.decision_points else 0.0

    return {
        "token_in_highlight_ratio": token_ratio,
        "decision_in_highlight_ratio": decision_ratio,
        "decision_supported_by_tokens_ratio": supported_ratio,
    }


def _query_rank(ranked: list[str], relevant: set[str]) -> int | None:
    for idx, item in enumerate(ranked, start=1):
        if item in relevant:
            return idx
    return None


def compute_retrieval_proxy(
    output: Output,
    num_time_queries: int = 10,
    time_window_s: float = 8.0,
    default_top_k: int = 6,
    retriever_config: dict[str, Any] | None = None,
    queries: list[SyntheticQuery] | None = None,
) -> dict[str, float]:
    retriever = Retriever(output_json=output, config=retriever_config)
    queries = queries or generate_synthetic_queries(
        output=output,
        num_time_queries=num_time_queries,
        time_window_s=time_window_s,
        default_top_k=default_top_k,
    )

    if not queries:
        return {"hit_at_k": 0.0, "mrr": 0.0, "num_queries": 0.0}

    hits = 0.0
    rr_sum = 0.0
    used = 0

    for q in queries:
        result = retriever.retrieve(q.query)
        ranked_highlights = [str(x) for x in result.get("selected_highlights", [])]
        ranked_decisions = [str(x) for x in result.get("selected_decisions", [])]

        rank: int | None = None
        if q.relevant_highlights:
            rank = _query_rank(ranked_highlights, set(q.relevant_highlights))
        if rank is None and q.relevant_decisions:
            rank = _query_rank(ranked_decisions, set(q.relevant_decisions))
        if q.relevant_highlights or q.relevant_decisions:
            used += 1
            if rank is not None and rank <= int(q.top_k):
                hits += 1.0
                rr_sum += 1.0 / float(rank)

    if used == 0:
        return {"hit_at_k": 0.0, "mrr": 0.0, "num_queries": 0.0}

    return {
        "hit_at_k": float(hits / used),
        "mrr": float(rr_sum / used),
        "num_queries": float(used),
    }


def compute_efficiency(output: Output, index_prefix: str | Path | None = None) -> dict[str, float]:
    index_dim = 0
    if index_prefix is not None:
        meta_path = Path(f"{index_prefix}.index_meta.json")
        if meta_path.exists():
            try:
                import json

                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    index_dim = int(payload.get("dim", 0))
            except Exception:
                index_dim = 0

    return {
        "index_dim": float(index_dim),
        "tokens_total": float(len(output.token_codec.tokens)),
        "decisions_total": float(len(output.decision_points)),
        "highlights_total": float(len(output.highlights)),
    }


def evaluate_output(
    output: Output,
    eval_config: dict[str, Any] | None = None,
    retriever_config: dict[str, Any] | None = None,
    index_prefix: str | Path | None = None,
) -> dict[str, float]:
    cfg = eval_config or {}
    duration_s = float(output.meta.get("duration_s", 0.0))

    coverage = compute_coverage(output)
    consistency = compute_consistency(output)
    retrieval = compute_retrieval_proxy(
        output=output,
        num_time_queries=int(cfg.get("num_time_queries", 10)),
        time_window_s=float(cfg.get("time_window_s", 8.0)),
        default_top_k=int(cfg.get("default_top_k", 6)),
        retriever_config=retriever_config,
    )
    efficiency = compute_efficiency(output, index_prefix=index_prefix)

    kept_duration = float(output.stats.get("kept_duration_s", 0.0))
    if kept_duration <= 0:
        kept_duration = float(coverage["highlight_coverage_s"])
    compression = float(duration_s / kept_duration) if kept_duration > 1e-9 else 0.0

    metrics: dict[str, float] = {
        "duration_s": duration_s,
        "kept_duration_s": kept_duration,
        "compression_ratio": compression,
        **coverage,
        **consistency,
        **retrieval,
        **efficiency,
    }
    return metrics
