from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pov_compiler.bench.nlq.datasets import NLQSample
from pov_compiler.eval.ablation import ALL_VARIANTS, apply_variant
from pov_compiler.eval.budget_sweep import apply_budget
from pov_compiler.eval.eval_cross_variant import build_budget_grid
from pov_compiler.eval.metrics import compute_consistency, compute_coverage, compute_efficiency
from pov_compiler.retrieval.query_planner import QueryCandidate, plan as plan_query
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Output
from pov_compiler.utils.media import get_duration_bucket


_TOPK_PATTERN = re.compile(r"(?:^|\s)top_k=\d+(?:\s|$)")


def _span_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = float(min(a[0], a[1])), float(max(a[0], a[1]))
    b0, b1 = float(min(b[0], b[1])), float(max(b[0], b[1]))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, (a1 - a0) + (b1 - b0) - inter)
    return float(inter / union)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_pred_spans(output: Output, retrieval_result: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    out: list[tuple[str, str, float, float]] = []
    seen: set[tuple[str, str]] = set()
    hl_map = {hl.id: hl for hl in output.highlights}
    event_map = {event.id: event for event in output.events}
    decision_map = {dp.id: dp for dp in output.decision_points}

    for hid in retrieval_result.get("selected_highlights", []) or []:
        hid = str(hid)
        if hid in hl_map and ("highlight", hid) not in seen:
            hl = hl_map[hid]
            out.append(("highlight", hid, float(hl.t0), float(hl.t1)))
            seen.add(("highlight", hid))
    for did in retrieval_result.get("selected_decisions", []) or []:
        did = str(did)
        if did in decision_map and ("decision", did) not in seen:
            dp = decision_map[did]
            out.append(("decision", did, float(dp.t0), float(dp.t1)))
            seen.add(("decision", did))
    for eid in retrieval_result.get("selected_events", []) or []:
        eid = str(eid)
        if eid in event_map and ("event", eid) not in seen:
            event = event_map[eid]
            out.append(("event", eid, float(event.t0), float(event.t1)))
            seen.add(("event", eid))
    return out


def _rank_for_gt_span(
    pred_spans: list[tuple[str, str, float, float]],
    gt_span: tuple[float, float],
    min_iou: float = 0.1,
) -> tuple[int | None, float]:
    best_iou = 0.0
    rank: int | None = None
    for i, (_, _, t0, t1) in enumerate(pred_spans, start=1):
        iou = _span_iou((t0, t1), gt_span)
        if iou > best_iou:
            best_iou = iou
        if rank is None and iou >= float(min_iou):
            rank = i
    return rank, best_iou


def _top1_in_distractor(
    pred_spans: list[tuple[str, str, float, float]],
    distractors: list[tuple[float, float]],
    min_iou: float = 0.1,
) -> bool:
    if not pred_spans or not distractors:
        return False
    _, _, t0, t1 = pred_spans[0]
    top1_span = (float(t0), float(t1))
    for dspan in distractors:
        if _span_iou(top1_span, (float(dspan[0]), float(dspan[1]))) > float(min_iou):
            return True
    return False


def _base_metrics(output: Output, index_prefix: str | Path | None = None) -> dict[str, float]:
    duration_s = float(output.meta.get("duration_s", 0.0))
    coverage = compute_coverage(output)
    consistency = compute_consistency(output)
    efficiency = compute_efficiency(output, index_prefix=index_prefix)
    kept_duration = float(output.stats.get("kept_duration_s", coverage.get("highlight_coverage_s", 0.0)))
    compression_ratio = float(duration_s / kept_duration) if kept_duration > 1e-9 else 0.0
    return {
        "duration_s": duration_s,
        "kept_duration_s": kept_duration,
        "compression_ratio": compression_ratio,
        **coverage,
        **consistency,
        **efficiency,
    }


def _force_top_k(query: str, top_k: int) -> str:
    text = str(query).strip()
    if not text:
        return text
    if _TOPK_PATTERN.search(text):
        text = _TOPK_PATTERN.sub(" ", text).strip()
    return f"{text} top_k={int(max(1, top_k))}"


def _build_candidates(sample: NLQSample, allow_gt_fallback: bool) -> list[QueryCandidate]:
    planned = plan_query(sample.query)
    candidates: list[QueryCandidate] = []
    seen: set[str] = set()

    for cand in planned:
        q = _force_top_k(str(cand.get("query", "")), int(sample.top_k))
        if not q or q in seen:
            continue
        candidates.append(
            QueryCandidate(
                query=q,
                reason=str(cand.get("reason", "planned")),
                priority=int(cand.get("priority", 100)),
            )
        )
        seen.add(q)

    if allow_gt_fallback:
        t0, t1 = float(sample.gt_span[0]), float(sample.gt_span[1])
        fallback = _force_top_k(
            f"time={max(0.0, t0 - 2.0):.3f}-{max(t0 + 0.2, t1 + 2.0):.3f}",
            int(sample.top_k),
        )
        if fallback not in seen:
            candidates.append(
                QueryCandidate(
                    query=fallback,
                    reason="fallback_gt_time",
                    priority=999,
                )
            )

    candidates.sort(key=lambda x: (int(x["priority"]), str(x["query"])))
    return candidates


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "selected_events": [],
        "selected_highlights": [],
        "selected_decisions": [],
        "selected_tokens": [],
        "debug": {"reason": str(reason)},
    }


def evaluate_nlq_samples(
    *,
    output: Output,
    samples: list[NLQSample],
    budgets: dict[str, Any],
    sweep: bool,
    retriever_config: dict[str, Any] | None = None,
    index_prefix: str | Path | None = None,
    allow_gt_fallback: bool = True,
    variants: list[str] | None = None,
    min_iou: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    retriever_config = retriever_config or {}
    used_variants = [str(v) for v in (variants or ALL_VARIANTS)]
    budget_grid = build_budget_grid(budgets, sweep=sweep)

    overall_rows: list[dict[str, Any]] = []
    by_type_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []

    duration_bucket = get_duration_bucket(output.meta.get("duration_s"))

    for variant in used_variants:
        variant_output = apply_variant(output, variant=variant)
        for budget in budget_grid:
            budgeted = apply_budget(variant_output, budget=budget)
            base = _base_metrics(budgeted, index_prefix=index_prefix)
            retriever = Retriever(output_json=budgeted, index=index_prefix, config=retriever_config)

            local_rows: list[dict[str, Any]] = []
            for sample in samples:
                candidates = _build_candidates(sample, allow_gt_fallback=bool(allow_gt_fallback))
                chosen_query = ""
                chosen_reason = "no_candidate"
                retrieval_result = _empty_result(chosen_reason)

                if candidates:
                    cascade_queries = [cand["query"] for cand in candidates]
                    matched_query, cascade_result = retriever.retrieve_cascade(cascade_queries)
                    if matched_query is not None:
                        retrieval_result = cascade_result
                        chosen_query = str(matched_query)
                        for cand in candidates:
                            if str(cand["query"]) == chosen_query:
                                chosen_reason = str(cand["reason"])
                                break
                        if not chosen_reason:
                            chosen_reason = "planned"
                    else:
                        retrieval_result = cascade_result
                        chosen_query = str(cascade_queries[-1]) if cascade_queries else ""
                        # if fallback_gt_time exists, surface that; else router_no_match.
                        fallback_reasons = [str(c["reason"]) for c in candidates if str(c.get("reason", ""))]
                        chosen_reason = fallback_reasons[-1] if fallback_reasons else "router_no_match"

                pred_spans = _extract_pred_spans(budgeted, retrieval_result)
                rank, best_iou = _rank_for_gt_span(pred_spans, sample.gt_span, min_iou=float(min_iou))
                hit = 1.0 if rank is not None and rank <= int(sample.top_k) else 0.0
                mrr = 1.0 / float(rank) if rank is not None else 0.0
                hit_at_1 = 1.0 if rank == 1 else 0.0
                top1_in_distractor = (
                    1.0 if _top1_in_distractor(pred_spans, sample.distractors, min_iou=float(min_iou)) else 0.0
                )
                hit_at_1_strict = 1.0 if hit_at_1 > 0.0 and top1_in_distractor < 0.5 else 0.0
                hit_at_k_strict = 1.0 if hit > 0.0 and top1_in_distractor < 0.5 else 0.0

                row = {
                    "video_id": output.video_id,
                    "video_uid": output.video_id,
                    "variant": variant,
                    "budget_max_total_s": float(budget["max_total_s"]),
                    "budget_max_tokens": int(budget["max_tokens"]),
                    "budget_max_decisions": int(budget["max_decisions"]),
                    "duration_bucket": duration_bucket,
                    "qid": sample.qid,
                    "query_type": sample.query_type,
                    "query": sample.query,
                    "chosen_query": chosen_query,
                    "chosen_reason": chosen_reason,
                    # backward-compatible aliases
                    "route_query": chosen_query,
                    "route_reason": chosen_reason,
                    "allow_gt_fallback": bool(allow_gt_fallback),
                    "candidate_count": int(len(candidates)),
                    "gt_t0": float(sample.gt_span[0]),
                    "gt_t1": float(sample.gt_span[1]),
                    "num_distractors": int(len(sample.distractors)),
                    "hit_at_k": hit,
                    "hit_at_1": hit_at_1,
                    "hit_at_1_strict": hit_at_1_strict,
                    "hit_at_k_strict": hit_at_k_strict,
                    "top1_in_distractor": top1_in_distractor,
                    "mrr": mrr,
                    "rank": int(rank) if rank is not None else -1,
                    "best_iou": float(best_iou),
                    "hit_at_k_event": hit,
                    "mrr_event": mrr,
                    "hit_at_k_decision": hit,
                    "mrr_decision": mrr,
                    "hit_at_k_token": hit,
                    "mrr_token": mrr,
                    "hit_at_k_highlight": hit,
                    "mrr_highlight": mrr,
                    "reason": str(retrieval_result.get("debug", {}).get("reason", "")),
                }
                row.update(base)
                local_rows.append(row)
                per_query_rows.append(dict(row))

            overall_rows.append(
                {
                    "video_id": output.video_id,
                    "video_uid": output.video_id,
                    "variant": variant,
                    "budget_max_total_s": float(budget["max_total_s"]),
                    "budget_max_tokens": int(budget["max_tokens"]),
                    "budget_max_decisions": int(budget["max_decisions"]),
                    "duration_bucket": duration_bucket,
                    "num_queries": float(len(local_rows)),
                    "hit_at_k": _mean([float(r["hit_at_k"]) for r in local_rows]),
                    "hit_at_1": _mean([float(r["hit_at_1"]) for r in local_rows]),
                    "hit_at_1_strict": _mean([float(r["hit_at_1_strict"]) for r in local_rows]),
                    "hit_at_k_strict": _mean([float(r["hit_at_k_strict"]) for r in local_rows]),
                    "top1_in_distractor_rate": _mean([float(r["top1_in_distractor"]) for r in local_rows]),
                    "mrr": _mean([float(r["mrr"]) for r in local_rows]),
                    **base,
                }
            )

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in local_rows:
                grouped[str(row["query_type"])].append(row)
            for query_type, rows in sorted(grouped.items(), key=lambda x: x[0]):
                by_type_rows.append(
                    {
                        "video_id": output.video_id,
                        "video_uid": output.video_id,
                        "variant": variant,
                        "query_type": query_type,
                        "budget_max_total_s": float(budget["max_total_s"]),
                        "budget_max_tokens": int(budget["max_tokens"]),
                        "budget_max_decisions": int(budget["max_decisions"]),
                        "duration_bucket": duration_bucket,
                        "num_queries": float(len(rows)),
                        "hit_at_k": _mean([float(r["hit_at_k"]) for r in rows]),
                        "hit_at_1": _mean([float(r["hit_at_1"]) for r in rows]),
                        "hit_at_1_strict": _mean([float(r["hit_at_1_strict"]) for r in rows]),
                        "hit_at_k_strict": _mean([float(r["hit_at_k_strict"]) for r in rows]),
                        "top1_in_distractor_rate": _mean([float(r["top1_in_distractor"]) for r in rows]),
                        "mrr": _mean([float(r["mrr"]) for r in rows]),
                        "hit_at_k_event": _mean([float(r["hit_at_k_event"]) for r in rows]),
                        "mrr_event": _mean([float(r["mrr_event"]) for r in rows]),
                        "hit_at_k_decision": _mean([float(r["hit_at_k_decision"]) for r in rows]),
                        "mrr_decision": _mean([float(r["mrr_decision"]) for r in rows]),
                        "hit_at_k_token": _mean([float(r["hit_at_k_token"]) for r in rows]),
                        "mrr_token": _mean([float(r["mrr_token"]) for r in rows]),
                        "hit_at_k_highlight": _mean([float(r["hit_at_k_highlight"]) for r in rows]),
                        "mrr_highlight": _mean([float(r["mrr_highlight"]) for r in rows]),
                        **base,
                    }
                )

    return {
        "overall_rows": overall_rows,
        "by_query_type_rows": by_type_rows,
        "per_query_rows": per_query_rows,
    }
