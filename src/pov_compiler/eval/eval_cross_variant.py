from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

from pov_compiler.eval.ablation import ALL_VARIANTS, apply_variant
from pov_compiler.eval.budget_sweep import apply_budget
from pov_compiler.eval.fixed_queries import FixedQuery
from pov_compiler.eval.metrics import (
    compute_consistency,
    compute_coverage,
    compute_efficiency,
)
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Output


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


def build_budget_grid(budgets: dict[str, Any], sweep: bool = True) -> list[dict[str, Any]]:
    max_total_s = [float(x) for x in budgets.get("max_total_s", [20, 40, 60])]
    max_tokens = [int(x) for x in budgets.get("max_tokens", [50, 100, 200])]
    max_decisions = [int(x) for x in budgets.get("max_decisions", [4, 8, 12])]
    if not sweep:
        return [
            {
                "max_total_s": float(max(max_total_s) if max_total_s else 60.0),
                "max_tokens": int(max(max_tokens) if max_tokens else 200),
                "max_decisions": int(max(max_decisions) if max_decisions else 12),
            }
        ]

    grid: list[dict[str, Any]] = []
    for b_total, b_token, b_decision in product(max_total_s, max_tokens, max_decisions):
        grid.append(
            {
                "max_total_s": float(b_total),
                "max_tokens": int(b_token),
                "max_decisions": int(b_decision),
            }
        )
    grid.sort(key=lambda x: (x["max_total_s"], x["max_tokens"], x["max_decisions"]))
    return grid


def _rank_of_first(ranked_ids: list[str], relevant_ids: set[str], top_k: int) -> int | None:
    if not relevant_ids:
        return None
    for rank, item_id in enumerate(ranked_ids[: max(1, int(top_k))], start=1):
        if item_id in relevant_ids:
            return rank
    return None


def _score_rank(rank: int | None, has_relevant: bool) -> tuple[float | None, float | None]:
    if not has_relevant:
        return None, None
    if rank is None:
        return 0.0, 0.0
    return 1.0, 1.0 / float(rank)


def _select_primary_targets(query_type: str) -> list[str]:
    qtype = str(query_type).lower()
    if qtype == "decision":
        return ["decision", "highlight", "event", "token"]
    if qtype == "token":
        return ["token", "highlight", "event", "decision"]
    if qtype == "hard_time":
        return ["event", "token", "decision", "highlight"]
    if qtype == "anchor":
        return ["highlight", "event", "decision", "token"]
    return ["highlight", "event", "decision", "token"]


def _base_metrics(
    output: Output,
    index_prefix: str | Path | None = None,
) -> dict[str, float]:
    duration_s = _safe_float(output.meta.get("duration_s", 0.0))
    coverage = compute_coverage(output)
    consistency = compute_consistency(output)
    efficiency = compute_efficiency(output, index_prefix=index_prefix)
    kept_duration = _safe_float(output.stats.get("kept_duration_s", 0.0))
    if kept_duration <= 0:
        kept_duration = _safe_float(coverage.get("highlight_coverage_s", 0.0))
    compression = float(duration_s / kept_duration) if kept_duration > 1e-9 else 0.0
    return {
        "duration_s": duration_s,
        "kept_duration_s": kept_duration,
        "compression_ratio": compression,
        **coverage,
        **consistency,
        **efficiency,
    }


def _mean(values: list[float | None]) -> float:
    valid = [float(x) for x in values if x is not None]
    if not valid:
        return 0.0
    return float(sum(valid) / len(valid))


def _evaluate_single_query(
    query: FixedQuery,
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    top_k = max(1, int(query.top_k))
    relevant = query.relevant

    ranked = {
        "highlight": [str(x) for x in retrieval_result.get("selected_highlights", [])][:top_k],
        "event": [str(x) for x in retrieval_result.get("selected_events", [])][:top_k],
        "decision": [str(x) for x in retrieval_result.get("selected_decisions", [])][:top_k],
        "token": [str(x) for x in retrieval_result.get("selected_tokens", [])][:top_k],
    }
    relevant_sets = {
        "highlight": set(str(x) for x in relevant.get("highlights", [])),
        "event": set(str(x) for x in relevant.get("events", [])),
        "decision": set(str(x) for x in relevant.get("decisions", [])),
        "token": set(str(x) for x in relevant.get("tokens", [])),
    }

    rank_by_target: dict[str, int | None] = {}
    for target in ("highlight", "event", "decision", "token"):
        rank_by_target[target] = _rank_of_first(ranked[target], relevant_sets[target], top_k=top_k)

    hit_highlight, mrr_highlight = _score_rank(rank_by_target["highlight"], bool(relevant_sets["highlight"]))
    hit_event, mrr_event = _score_rank(rank_by_target["event"], bool(relevant_sets["event"]))
    hit_decision, mrr_decision = _score_rank(rank_by_target["decision"], bool(relevant_sets["decision"]))
    hit_token, mrr_token = _score_rank(rank_by_target["token"], bool(relevant_sets["token"]))

    overall_target = ""
    overall_rank: int | None = None
    for candidate in _select_primary_targets(query.type):
        if relevant_sets[candidate]:
            overall_target = candidate
            overall_rank = rank_by_target[candidate]
            break
    hit_overall, mrr_overall = _score_rank(overall_rank, bool(overall_target))

    return {
        "qid": query.qid,
        "query_type": query.type,
        "query": query.query,
        "top_k": top_k,
        "overall_target": overall_target,
        "hit_at_k": hit_overall,
        "mrr": mrr_overall,
        "hit_at_k_highlight": hit_highlight,
        "mrr_highlight": mrr_highlight,
        "hit_at_k_event": hit_event,
        "mrr_event": mrr_event,
        "hit_at_k_decision": hit_decision,
        "mrr_decision": mrr_decision,
        "hit_at_k_token": hit_token,
        "mrr_token": mrr_token,
        "num_rel_highlights": len(relevant_sets["highlight"]),
        "num_rel_events": len(relevant_sets["event"]),
        "num_rel_decisions": len(relevant_sets["decision"]),
        "num_rel_tokens": len(relevant_sets["token"]),
        "reason": str(retrieval_result.get("debug", {}).get("reason", "")),
    }


def _aggregate_query_rows(query_rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "num_queries": float(len(query_rows)),
        "hit_at_k": _mean([row.get("hit_at_k") for row in query_rows]),
        "mrr": _mean([row.get("mrr") for row in query_rows]),
        "hit_at_k_highlight": _mean([row.get("hit_at_k_highlight") for row in query_rows]),
        "mrr_highlight": _mean([row.get("mrr_highlight") for row in query_rows]),
        "hit_at_k_event": _mean([row.get("hit_at_k_event") for row in query_rows]),
        "mrr_event": _mean([row.get("mrr_event") for row in query_rows]),
        "hit_at_k_decision": _mean([row.get("hit_at_k_decision") for row in query_rows]),
        "mrr_decision": _mean([row.get("mrr_decision") for row in query_rows]),
        "hit_at_k_token": _mean([row.get("hit_at_k_token") for row in query_rows]),
        "mrr_token": _mean([row.get("mrr_token") for row in query_rows]),
    }


def evaluate_cross_variant(
    full_output: Output,
    queries: list[FixedQuery],
    variants: list[str] | None = None,
    budgets: dict[str, Any] | None = None,
    sweep: bool = True,
    retriever_config: dict[str, Any] | None = None,
    index_prefix: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    variants = [str(v) for v in (variants or ALL_VARIANTS)]
    budget_grid = build_budget_grid(budgets or {}, sweep=sweep)
    retriever_config = retriever_config or {}

    overall_rows: list[dict[str, Any]] = []
    by_query_type_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []

    for variant in variants:
        variant_output = apply_variant(full_output, variant=variant)
        for budget in budget_grid:
            budgeted_output = apply_budget(variant_output, budget=budget)
            base = _base_metrics(budgeted_output, index_prefix=index_prefix)
            retriever = Retriever(output_json=budgeted_output, config=retriever_config)

            local_rows: list[dict[str, Any]] = []
            for query in queries:
                result = retriever.retrieve(query.query)
                qrow = _evaluate_single_query(query, result)
                qrow.update(
                    {
                        "video_id": full_output.video_id,
                        "variant": variant,
                        "budget_max_total_s": float(budget["max_total_s"]),
                        "budget_max_tokens": int(budget["max_tokens"]),
                        "budget_max_decisions": int(budget["max_decisions"]),
                    }
                )
                local_rows.append(qrow)
                per_query_rows.append(dict(qrow))

            agg = _aggregate_query_rows(local_rows)
            overall_row = {
                "video_id": full_output.video_id,
                "variant": variant,
                "budget_max_total_s": float(budget["max_total_s"]),
                "budget_max_tokens": int(budget["max_tokens"]),
                "budget_max_decisions": int(budget["max_decisions"]),
                **base,
                **agg,
            }
            overall_rows.append(overall_row)

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in local_rows:
                grouped[str(row.get("query_type", ""))].append(row)
            for query_type, qrows in sorted(grouped.items(), key=lambda x: x[0]):
                by_type = _aggregate_query_rows(qrows)
                by_query_type_rows.append(
                    {
                        "video_id": full_output.video_id,
                        "variant": variant,
                        "query_type": query_type,
                        "budget_max_total_s": float(budget["max_total_s"]),
                        "budget_max_tokens": int(budget["max_tokens"]),
                        "budget_max_decisions": int(budget["max_decisions"]),
                        **base,
                        **by_type,
                    }
                )

    return {
        "overall_rows": overall_rows,
        "by_query_type_rows": by_query_type_rows,
        "per_query_rows": per_query_rows,
    }


def _group_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], metric: str) -> dict[tuple[Any, ...], float]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        group_key = tuple(row.get(k) for k in keys)
        groups[group_key].append(_safe_float(row.get(metric, 0.0)))
    return {k: float(sum(v) / len(v)) if v else 0.0 for k, v in groups.items()}


def make_cross_report(
    overall_rows: list[dict[str, Any]],
    by_query_type_rows: list[dict[str, Any]],
    report_path: str | Path,
    overall_csv_path: str | Path,
    by_type_csv_path: str | Path,
) -> None:
    lines: list[str] = []
    variants = sorted({str(r.get("variant", "")) for r in overall_rows if str(r.get("variant", ""))})
    query_types = sorted({str(r.get("query_type", "")) for r in by_query_type_rows if str(r.get("query_type", ""))})
    videos = sorted({str(r.get("video_id", "")) for r in overall_rows if str(r.get("video_id", ""))})

    lines.append("# Cross-Variant Evaluation Report")
    lines.append("")
    lines.append(f"- videos: {len(videos)}")
    lines.append(f"- variants: {', '.join(variants)}")
    lines.append(f"- query_types: {', '.join(query_types)}")
    lines.append(f"- rows_overall: {len(overall_rows)}")
    lines.append(f"- rows_by_query_type: {len(by_query_type_rows)}")
    lines.append(f"- results_overall_csv: `{overall_csv_path}`")
    lines.append(f"- results_by_query_type_csv: `{by_type_csv_path}`")
    lines.append("")

    # Overall summary by variant.
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| variant | hit@k | mrr | coverage_ratio | compression_ratio |")
    lines.append("|---|---:|---:|---:|---:|")
    mean_hit = _group_mean(overall_rows, ("variant",), "hit_at_k")
    mean_mrr = _group_mean(overall_rows, ("variant",), "mrr")
    mean_cov = _group_mean(overall_rows, ("variant",), "coverage_ratio")
    mean_cmp = _group_mean(overall_rows, ("variant",), "compression_ratio")
    for variant in variants:
        lines.append(
            f"| {variant} | {mean_hit.get((variant,), 0.0):.4f} | "
            f"{mean_mrr.get((variant,), 0.0):.4f} | {mean_cov.get((variant,), 0.0):.4f} | "
            f"{mean_cmp.get((variant,), 0.0):.4f} |"
        )
    lines.append("")

    # Per query type summary by variant.
    lines.append("## By Query Type")
    lines.append("")
    lines.append("| query_type | variant | hit@k | mrr | event_hit@k | decision_hit@k | token_hit@k |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    q_hit = _group_mean(by_query_type_rows, ("query_type", "variant"), "hit_at_k")
    q_mrr = _group_mean(by_query_type_rows, ("query_type", "variant"), "mrr")
    q_event = _group_mean(by_query_type_rows, ("query_type", "variant"), "hit_at_k_event")
    q_decision = _group_mean(by_query_type_rows, ("query_type", "variant"), "hit_at_k_decision")
    q_token = _group_mean(by_query_type_rows, ("query_type", "variant"), "hit_at_k_token")
    for query_type in query_types:
        for variant in variants:
            lines.append(
                f"| {query_type} | {variant} | {q_hit.get((query_type, variant), 0.0):.4f} | "
                f"{q_mrr.get((query_type, variant), 0.0):.4f} | {q_event.get((query_type, variant), 0.0):.4f} | "
                f"{q_decision.get((query_type, variant), 0.0):.4f} | {q_token.get((query_type, variant), 0.0):.4f} |"
            )
    lines.append("")

    # Delta-focused findings.
    lines.append("## Key Deltas")
    lines.append("")

    def _delta(query_type: str, left: str, right: str) -> tuple[float, float]:
        left_hit = q_hit.get((query_type, left), 0.0)
        right_hit = q_hit.get((query_type, right), 0.0)
        return left_hit, right_hit - left_hit

    # token
    token_base, token_delta = _delta("token", "highlights_only", "full")
    token_plus, token_plus_delta = _delta("token", "highlights_only", "highlights_plus_tokens")
    lines.append(
        f"- token queries: `full` vs `highlights_only` hit@k {token_base + token_delta:.4f} vs "
        f"{token_base:.4f} (delta {token_delta:+.4f}); "
        f"`highlights_plus_tokens` delta {token_plus_delta:+.4f}."
    )
    # decision
    decision_base, decision_delta = _delta("decision", "highlights_only", "full")
    decision_plus, decision_plus_delta = _delta("decision", "highlights_only", "highlights_plus_decisions")
    lines.append(
        f"- decision queries: `full` vs `highlights_only` hit@k {decision_base + decision_delta:.4f} vs "
        f"{decision_base:.4f} (delta {decision_delta:+.4f}); "
        f"`highlights_plus_decisions` delta {decision_plus_delta:+.4f}."
    )
    # hard_time (event focus)
    hard_full = q_event.get(("hard_time", "full"), 0.0)
    hard_raw = q_event.get(("hard_time", "raw_events_only"), 0.0)
    hard_delta = hard_full - hard_raw
    lines.append(
        f"- hard_time queries (event hit@k): `full` {hard_full:.4f} vs `raw_events_only` {hard_raw:.4f} "
        f"(delta {hard_delta:+.4f})."
    )
    lines.append("")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
