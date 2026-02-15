from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pov_compiler.bench.nlq.datasets import NLQSample
from pov_compiler.eval.ablation import ALL_VARIANTS, apply_variant
from pov_compiler.eval.budget_sweep import apply_budget
from pov_compiler.eval.eval_cross_variant import build_budget_grid
from pov_compiler.eval.metrics import compute_consistency, compute_coverage, compute_efficiency
from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_parser import parse_query_chain
from pov_compiler.retrieval.query_planner import QueryCandidate, QueryPlan, plan as plan_query
from pov_compiler.retrieval.reranker import Hit, rerank
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Output
from pov_compiler.utils.media import get_duration_bucket


_TOPK_PATTERN = re.compile(r"(?:^|\s)top_k=\d+(?:\s|$)")
_CONSTRAINT_KEYS = (
    "after_scene_change",
    "first_last",
    "type_match",
    "object_match",
    "interaction_object",
    "interaction_min",
    "place_first_last",
    "place_segment_id",
    "chain_time_range",
    "chain_place_match",
    "chain_object_match",
)


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


def _mode(values: list[str], default: str = "none") -> str:
    if not values:
        return default
    counts: dict[str, int] = {}
    for value in values:
        k = str(value)
        counts[k] = counts.get(k, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: (-int(x[1]), str(x[0])))
    return str(ordered[0][0]) if ordered else default


def _kind_rate(rows: list[dict[str, Any]], kind: str) -> float:
    if not rows:
        return 0.0
    k = str(kind)
    return float(sum(1.0 for row in rows if str(row.get("top1_kind", "")) == k) / float(len(rows)))


def _constraint_rate(rows: list[dict[str, Any]], field: str, name: str) -> float:
    if not rows:
        return 0.0
    count = 0.0
    key = str(name)
    for row in rows:
        payload = row.get(field, "[]")
        if isinstance(payload, str):
            try:
                arr = json.loads(payload)
            except Exception:
                arr = []
        elif isinstance(payload, list):
            arr = payload
        else:
            arr = []
        if key in [str(x) for x in arr]:
            count += 1.0
    return float(count / float(len(rows)))


def _constraint_present_flags(plan_constraints: dict[str, Any]) -> dict[str, bool]:
    constraints = dict(plan_constraints)
    out = {
        "after_scene_change": bool(constraints.get("after_scene_change", False)),
        "first_last": str(constraints.get("which", "")).lower() in {"first", "last"},
        "type_match": any(
            bool(str(constraints.get(key, "")).strip()) for key in ("anchor_type", "token_type", "decision_type")
        ),
        "object_match": bool(
            str(
                constraints.get(
                    "object_name",
                    constraints.get("lost_object", constraints.get("object_last_seen", "")),
                )
            ).strip()
        ),
        "interaction_object": bool(str(constraints.get("interaction_object", "")).strip()),
        "interaction_min": constraints.get("interaction_min", None) is not None,
        "place_first_last": str(constraints.get("place", "")).lower() in {"first", "last"},
        "place_segment_id": bool(
            constraints.get("place_segment_id")
            or constraints.get("place_segment_ids")
        ),
        "chain_time_range": (
            str(constraints.get("chain_time_mode", "hard")).strip().lower() != "off"
            and (
                constraints.get("chain_time_min_s", None) is not None
                or constraints.get("chain_time_max_s", None) is not None
            )
        ),
        "chain_place_match": (
            str(constraints.get("chain_place_mode", "soft")).strip().lower() != "off"
            and bool(str(constraints.get("chain_place_value", "")).strip())
        ),
        "chain_object_match": (
            str(constraints.get("chain_object_mode", "soft")).strip().lower() != "off"
            and bool(str(constraints.get("chain_object_value", "")).strip())
        ),
    }
    for key in _CONSTRAINT_KEYS:
        out.setdefault(key, False)
    return out


def _constraint_stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        stats = {"used_fallback_rate": 0.0, "avg_filtered_before": 0.0, "avg_filtered_after": 0.0}
        for key in _CONSTRAINT_KEYS:
            stats[f"present_{key}_rate"] = 0.0
            stats[f"filtered_{key}_rate"] = 0.0
            stats[f"relaxed_{key}_rate"] = 0.0
        return stats

    n = float(len(rows))

    def _mean_bool(field: str) -> float:
        return float(sum(1.0 for row in rows if bool(row.get(field, False))) / n)

    stats: dict[str, float] = {
        "used_fallback_rate": _mean_bool("used_fallback"),
        "avg_filtered_before": float(sum(float(row.get("filtered_hits_before", 0.0)) for row in rows) / n),
        "avg_filtered_after": float(sum(float(row.get("filtered_hits_after", 0.0)) for row in rows) / n),
    }
    for key in _CONSTRAINT_KEYS:
        stats[f"present_{key}_rate"] = _mean_bool(f"present_{key}")
        stats[f"filtered_{key}_rate"] = _mean_bool(f"filtered_{key}")
        stats[f"relaxed_{key}_rate"] = _mean_bool(f"relaxed_{key}")
    return stats


def _extract_pred_spans_from_hits(hits: list[Hit], top_k: int) -> list[tuple[str, str, float, float]]:
    out: list[tuple[str, str, float, float]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits[: max(1, int(top_k))]:
        kind = str(hit["kind"])
        hit_id = str(hit["id"])
        key = (kind, hit_id)
        if key in seen:
            continue
        out.append((kind, hit_id, float(hit["t0"]), float(hit["t1"])))
        seen.add(key)
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


def _build_plan(sample: NLQSample, allow_gt_fallback: bool) -> QueryPlan:
    if " then " in str(sample.query).lower():
        q = _force_top_k(str(sample.query), int(sample.top_k))
        chain = parse_query_chain(q)
        constraints: dict[str, Any] = {}
        if chain is not None and chain.steps:
            step2 = chain.steps[-1].parsed
            if step2.chain_time_mode is not None:
                constraints["chain_time_mode"] = str(step2.chain_time_mode)
            if step2.chain_time_min_s is not None:
                constraints["chain_time_min_s"] = float(step2.chain_time_min_s)
            if step2.chain_time_max_s is not None:
                constraints["chain_time_max_s"] = float(step2.chain_time_max_s)
            if step2.chain_place_mode:
                constraints["chain_place_mode"] = str(step2.chain_place_mode)
            if step2.chain_place_value:
                constraints["chain_place_value"] = str(step2.chain_place_value)
            if step2.chain_object_mode:
                constraints["chain_object_mode"] = str(step2.chain_object_mode)
            if step2.chain_object_value:
                constraints["chain_object_value"] = str(step2.chain_object_value)
        return QueryPlan(
            intent="mixed",
            candidates=[QueryCandidate(query=q, reason="chain_query", priority=0)],
            constraints=constraints,
            debug={"chain_query": True, "candidate_count": 1},
        )

    planned = plan_query(sample.query)
    candidates: list[QueryCandidate] = []
    seen: set[str] = set()

    for cand in planned.candidates:
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
    return QueryPlan(
        intent=planned.intent,
        candidates=candidates,
        constraints=dict(planned.constraints),
        debug=dict(planned.debug),
    )


def evaluate_nlq_samples(
    *,
    output: Output,
    samples: list[NLQSample],
    budgets: dict[str, Any],
    sweep: bool,
    retriever_config: dict[str, Any] | None = None,
    index_prefix: str | Path | None = None,
    rerank_cfg: WeightConfig | dict[str, Any] | str | Path | None = None,
    hard_constraints_cfg: HardConstraintConfig | dict[str, Any] | str | Path | None = None,
    allow_gt_fallback: bool = True,
    variants: list[str] | None = None,
    min_iou: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    retriever_config = retriever_config or {}
    weights = resolve_weight_config(rerank_cfg)
    cfg_name = str(weights.name)
    cfg_hash = str(weights.short_hash())
    if hard_constraints_cfg is None:
        hard_cfg = HardConstraintConfig()
    elif isinstance(hard_constraints_cfg, HardConstraintConfig):
        hard_cfg = hard_constraints_cfg
    elif isinstance(hard_constraints_cfg, dict):
        hard_cfg = HardConstraintConfig.from_dict(hard_constraints_cfg)
    else:
        hard_cfg = HardConstraintConfig.from_yaml(hard_constraints_cfg)
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
                query_plan = _build_plan(sample, allow_gt_fallback=bool(allow_gt_fallback))
                candidates = list(query_plan.candidates)
                candidate_queries = [str(cand["query"]) for cand in candidates]
                merged_hits = retriever.retrieve_multi(candidate_queries)
                cresult = apply_constraints_detailed(
                    merged_hits,
                    query_plan=query_plan,
                    cfg=hard_cfg,
                    output=budgeted,
                )
                reranked_hits = rerank(
                    cresult.hits,
                    plan=query_plan,
                    context=budgeted,
                    cfg=weights,
                    distractors=sample.distractors,
                    constraint_trace={
                        "applied_constraints": list(cresult.applied),
                        "constraints_relaxed": list(cresult.relaxed),
                        "filtered_hits_before": int(cresult.filtered_before),
                        "filtered_hits_after": int(cresult.filtered_after),
                        "used_fallback": bool(cresult.used_fallback),
                    },
                )

                top1_kind = str(reranked_hits[0]["kind"]) if reranked_hits else "none"
                top1_source_query = str(reranked_hits[0]["source_query"]) if reranked_hits else ""
                chosen_query = top1_source_query
                chosen_reason = "no_candidate"
                present_flags = _constraint_present_flags(query_plan.constraints)
                did_filter_any = int(cresult.filtered_after) < int(cresult.filtered_before)
                relaxed_first = str(cresult.relaxed[0]) if cresult.relaxed else ""
                filtered_flags = {
                    key: bool(present_flags.get(key, False) and did_filter_any and key in [str(x) for x in cresult.applied])
                    for key in _CONSTRAINT_KEYS
                }
                relaxed_flags = {
                    key: bool(present_flags.get(key, False) and relaxed_first == key)
                    for key in _CONSTRAINT_KEYS
                }
                if chosen_query:
                    for cand in candidates:
                        if str(cand["query"]) == chosen_query:
                            chosen_reason = str(cand.get("reason", "planned"))
                            break
                    if chosen_reason == "no_candidate":
                        chosen_reason = "planned"
                elif candidates:
                    fallback_reasons = [str(c.get("reason", "")) for c in candidates if str(c.get("reason", ""))]
                    chosen_reason = fallback_reasons[-1] if fallback_reasons else "router_no_match"

                pred_spans = _extract_pred_spans_from_hits(reranked_hits, top_k=int(sample.top_k))
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
                    "chosen_plan_intent": str(query_plan.intent),
                    "applied_constraints": json.dumps(query_plan.constraints, ensure_ascii=False, sort_keys=True),
                    "constraints_applied": json.dumps(cresult.applied, ensure_ascii=False),
                    "constraints_relaxed": json.dumps(cresult.relaxed, ensure_ascii=False),
                    "filtered_hits_before": int(cresult.filtered_before),
                    "filtered_hits_after": int(cresult.filtered_after),
                    "used_fallback": bool(cresult.used_fallback),
                    "chain_backoff_enabled": bool(cresult.chain_backoff_enabled),
                    "chain_backoff_chosen_level": (
                        int(cresult.chain_backoff_chosen_level)
                        if cresult.chain_backoff_chosen_level is not None
                        else -1
                    ),
                    "chain_backoff_exhausted": bool(cresult.chain_backoff_exhausted),
                    "chain_backoff_attempts": json.dumps(list(cresult.chain_backoff_attempts), ensure_ascii=False),
                    "relaxed_first_constraint": relaxed_first,
                    "top1_kind": top1_kind,
                    "top1_source_query": top1_source_query,
                    "rerank_cfg_name": cfg_name,
                    "rerank_cfg_hash": cfg_hash,
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
                    "reason": (
                        f"multi_hits={len(merged_hits)}; filtered={len(cresult.hits)}; "
                        f"reranked={len(reranked_hits)}; fallback={str(bool(cresult.used_fallback)).lower()}"
                    ),
                }
                chain_meta = sample.meta.get("chain_meta", {}) if isinstance(sample.meta, dict) else {}
                if isinstance(chain_meta, dict):
                    row["chain_combo"] = str(chain_meta.get("combo", ""))
                    row["chain_step1_type"] = str(chain_meta.get("step1_type", ""))
                    row["chain_step2_type"] = str(chain_meta.get("step2_type", ""))
                    row["chain_derive"] = str(chain_meta.get("derive", ""))
                for key in _CONSTRAINT_KEYS:
                    row[f"present_{key}"] = bool(present_flags.get(key, False))
                    row[f"filtered_{key}"] = bool(filtered_flags.get(key, False))
                    row[f"relaxed_{key}"] = bool(relaxed_flags.get(key, False))

                top1_meta = dict(reranked_hits[0].get("meta", {})) if reranked_hits else {}
                chain_payload: dict[str, Any] = {}
                for hit_item in reranked_hits:
                    meta_item = hit_item.get("meta", {})
                    if isinstance(meta_item, dict):
                        cmeta = meta_item.get("chain", {})
                        if isinstance(cmeta, dict) and bool(cmeta.get("is_chain", False)):
                            chain_payload = cmeta
                            break
                step1_info = chain_payload.get("step1", {}) if isinstance(chain_payload, dict) else {}
                step2_info = chain_payload.get("step2", {}) if isinstance(chain_payload, dict) else {}
                chain_step1_has_hit = 1.0 if float(step1_info.get("hit_count", 0.0) or 0.0) > 0.0 else 0.0
                chain_step2_has_hit = 1.0 if float(step2_info.get("hit_count", 0.0) or 0.0) > 0.0 else 0.0
                step2_before = float(step2_info.get("filtered_hits_before", 0.0) or 0.0)
                step2_after = float(step2_info.get("filtered_hits_after", 0.0) or 0.0)
                chain_filtered_ratio_step2 = float((step2_before - step2_after) / step2_before) if step2_before > 0.0 else 0.0
                derived_constraints = chain_payload.get("derived_constraints", {}) if isinstance(chain_payload, dict) else {}
                d_time = dict(derived_constraints.get("time", {})) if isinstance(derived_constraints, dict) else {}
                d_place = dict(derived_constraints.get("place", {})) if isinstance(derived_constraints, dict) else {}
                d_object = dict(derived_constraints.get("object", {})) if isinstance(derived_constraints, dict) else {}
                chain_success = (
                    1.0
                    if (
                        chain_step1_has_hit > 0.0
                        and chain_step2_has_hit > 0.0
                        and hit_at_k_strict > 0.0
                        and top1_in_distractor < 0.5
                        and len(pred_spans) > 0
                    )
                    else 0.0
                )
                chain_derived_time_used = 1.0 if bool(d_time.get("enabled", False)) else 0.0
                chain_derived_place_used = 1.0 if bool(d_place.get("enabled", False)) else 0.0
                chain_derived_object_used = 1.0 if bool(d_object.get("enabled", False)) else 0.0
                chain_derived_nonempty = (
                    1.0 if (chain_derived_time_used > 0.0 or chain_derived_place_used > 0.0 or chain_derived_object_used > 0.0) else 0.0
                )
                chain_fail_reason = "other"
                if chain_step1_has_hit <= 0.0:
                    chain_fail_reason = "step1_no_hit"
                elif chain_step2_has_hit <= 0.0:
                    if bool(cresult.chain_backoff_exhausted):
                        chain_fail_reason = "backoff_exhausted"
                    elif step2_before > 0.0 and step2_after <= 0.0:
                        chain_fail_reason = "constraints_over_filtered"
                    else:
                        chain_fail_reason = "step2_no_hit"
                elif top1_in_distractor >= 0.5:
                    chain_fail_reason = "retrieval_distractor"
                elif len(pred_spans) <= 0:
                    chain_fail_reason = "evidence_missing"
                elif chain_success > 0.0:
                    chain_fail_reason = "success"
                elif "budget" in str(row.get("reason", "")).lower():
                    chain_fail_reason = "budget_insufficient"
                gt_place = str(sample.meta.get("gt_place_segment_id", "")).strip()
                pred_place = str(top1_meta.get("place_segment_id", "")).strip()
                row["top1_place_segment_mismatch"] = float(
                    1.0 if gt_place and pred_place and gt_place != pred_place else 0.0
                )
                gt_obj = str(sample.meta.get("interaction_object", sample.meta.get("active_object_top1", ""))).strip().lower()
                pred_obj = str(top1_meta.get("interaction_primary_object", "")).strip().lower()
                row["top1_interaction_object_match"] = float(1.0 if gt_obj and pred_obj and gt_obj == pred_obj else 0.0)
                row["chain_step1_has_hit"] = float(chain_step1_has_hit)
                row["chain_step2_has_hit"] = float(chain_step2_has_hit)
                row["chain_success"] = float(chain_success)
                row["chain_filtered_ratio_step2"] = float(chain_filtered_ratio_step2)
                row["chain_fail_reason"] = str(chain_fail_reason)
                row["chain_derived_time_used"] = float(chain_derived_time_used)
                row["chain_derived_place_used"] = float(chain_derived_place_used)
                row["chain_derived_object_used"] = float(chain_derived_object_used)
                row["chain_derived_nonempty"] = float(chain_derived_nonempty)
                row["chain_backoff_used"] = float(1.0 if bool(cresult.chain_backoff_enabled) else 0.0)
                row["chain_backoff_level"] = float(
                    cresult.chain_backoff_chosen_level
                    if cresult.chain_backoff_chosen_level is not None
                    else (4.0 if bool(cresult.chain_backoff_exhausted) else 0.0)
                )

                row.update(base)
                local_rows.append(row)
                per_query_rows.append(dict(row))

            local_constraint_stats = _constraint_stats_from_rows(local_rows)
            overall_rows.append(
                {
                    "video_id": output.video_id,
                    "video_uid": output.video_id,
                    "variant": variant,
                    "budget_max_total_s": float(budget["max_total_s"]),
                    "budget_max_tokens": int(budget["max_tokens"]),
                    "budget_max_decisions": int(budget["max_decisions"]),
                    "duration_bucket": duration_bucket,
                    "rerank_cfg_name": cfg_name,
                    "rerank_cfg_hash": cfg_hash,
                    "num_queries": float(len(local_rows)),
                    "hit_at_k": _mean([float(r["hit_at_k"]) for r in local_rows]),
                    "hit_at_1": _mean([float(r["hit_at_1"]) for r in local_rows]),
                    "hit_at_1_strict": _mean([float(r["hit_at_1_strict"]) for r in local_rows]),
                    "hit_at_k_strict": _mean([float(r["hit_at_k_strict"]) for r in local_rows]),
                    "top1_in_distractor_rate": _mean([float(r["top1_in_distractor"]) for r in local_rows]),
                    "top1_kind_mode": _mode([str(r.get("top1_kind", "none")) for r in local_rows]),
                    "top1_kind_highlight_rate": _kind_rate(local_rows, "highlight"),
                    "top1_kind_token_rate": _kind_rate(local_rows, "token"),
                    "top1_kind_decision_rate": _kind_rate(local_rows, "decision"),
                    "top1_kind_event_rate": _kind_rate(local_rows, "event"),
                    "top1_place_segment_mismatch_rate": _mean(
                        [float(r.get("top1_place_segment_mismatch", 0.0)) for r in local_rows]
                    ),
                    "top1_interaction_object_match_rate": _mean(
                        [float(r.get("top1_interaction_object_match", 0.0)) for r in local_rows]
                    ),
                    "chain_step1_has_hit_rate": _mean([float(r.get("chain_step1_has_hit", 0.0)) for r in local_rows]),
                    "chain_step2_has_hit_rate": _mean([float(r.get("chain_step2_has_hit", 0.0)) for r in local_rows]),
                    "chain_success_rate": _mean([float(r.get("chain_success", 0.0)) for r in local_rows]),
                    "chain_filtered_ratio_step2": _mean(
                        [float(r.get("chain_filtered_ratio_step2", 0.0)) for r in local_rows]
                    ),
                    "chain_derived_time_used_rate": _mean([float(r.get("chain_derived_time_used", 0.0)) for r in local_rows]),
                    "chain_derived_place_used_rate": _mean([float(r.get("chain_derived_place_used", 0.0)) for r in local_rows]),
                    "chain_derived_object_used_rate": _mean([float(r.get("chain_derived_object_used", 0.0)) for r in local_rows]),
                    "chain_derived_nonempty_rate": _mean([float(r.get("chain_derived_nonempty", 0.0)) for r in local_rows]),
                    "backoff_used_rate": _mean([float(r.get("chain_backoff_used", 0.0)) for r in local_rows]),
                    "backoff_mean_level": _mean([float(r.get("chain_backoff_level", 0.0)) for r in local_rows]),
                    "backoff_exhausted_rate": _mean(
                        [1.0 if bool(r.get("chain_backoff_exhausted", False)) else 0.0 for r in local_rows]
                    ),
                    "chain_fail_step1_no_hit_rate": _mean(
                        [1.0 if str(r.get("chain_fail_reason", "")) == "step1_no_hit" else 0.0 for r in local_rows]
                    ),
                    "chain_fail_step2_no_hit_rate": _mean(
                        [1.0 if str(r.get("chain_fail_reason", "")) == "step2_no_hit" else 0.0 for r in local_rows]
                    ),
                    "chain_fail_constraints_over_filtered_rate": _mean(
                        [1.0 if str(r.get("chain_fail_reason", "")) == "constraints_over_filtered" else 0.0 for r in local_rows]
                    ),
                    "chain_fail_retrieval_distractor_rate": _mean(
                        [1.0 if str(r.get("chain_fail_reason", "")) == "retrieval_distractor" else 0.0 for r in local_rows]
                    ),
                    "chain_fail_evidence_missing_rate": _mean(
                        [1.0 if str(r.get("chain_fail_reason", "")) == "evidence_missing" else 0.0 for r in local_rows]
                    ),
                    "chain_fail_budget_insufficient_rate": _mean(
                        [1.0 if str(r.get("chain_fail_reason", "")) == "budget_insufficient" else 0.0 for r in local_rows]
                    ),
                    "chain_fail_backoff_exhausted_rate": _mean(
                        [1.0 if str(r.get("chain_fail_reason", "")) == "backoff_exhausted" else 0.0 for r in local_rows]
                    ),
                    "mrr": _mean([float(r["mrr"]) for r in local_rows]),
                    **base,
                    **local_constraint_stats,
                    # Backward-compatible aliases.
                    "fallback_rate": float(local_constraint_stats.get("used_fallback_rate", 0.0)),
                    "constraint_apply_rate_after_scene_change": float(
                        local_constraint_stats.get("filtered_after_scene_change_rate", 0.0)
                    ),
                    "constraint_apply_rate_first_last": float(
                        local_constraint_stats.get("filtered_first_last_rate", 0.0)
                    ),
                    "constraint_apply_rate_type_match": float(
                        local_constraint_stats.get("filtered_type_match_rate", 0.0)
                    ),
                    "relax_rate_after_scene_change": float(
                        local_constraint_stats.get("relaxed_after_scene_change_rate", 0.0)
                    ),
                    "relax_rate_first_last": float(local_constraint_stats.get("relaxed_first_last_rate", 0.0)),
                    "relax_rate_type_match": float(local_constraint_stats.get("relaxed_type_match_rate", 0.0)),
                }
            )

            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in local_rows:
                grouped[str(row["query_type"])].append(row)
            for query_type, rows in sorted(grouped.items(), key=lambda x: x[0]):
                type_constraint_stats = _constraint_stats_from_rows(rows)
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
                        "rerank_cfg_name": cfg_name,
                        "rerank_cfg_hash": cfg_hash,
                        "num_queries": float(len(rows)),
                        "hit_at_k": _mean([float(r["hit_at_k"]) for r in rows]),
                        "hit_at_1": _mean([float(r["hit_at_1"]) for r in rows]),
                        "hit_at_1_strict": _mean([float(r["hit_at_1_strict"]) for r in rows]),
                        "hit_at_k_strict": _mean([float(r["hit_at_k_strict"]) for r in rows]),
                        "top1_in_distractor_rate": _mean([float(r["top1_in_distractor"]) for r in rows]),
                        "top1_kind_mode": _mode([str(r.get("top1_kind", "none")) for r in rows]),
                        "top1_kind_highlight_rate": _kind_rate(rows, "highlight"),
                        "top1_kind_token_rate": _kind_rate(rows, "token"),
                        "top1_kind_decision_rate": _kind_rate(rows, "decision"),
                        "top1_kind_event_rate": _kind_rate(rows, "event"),
                        "top1_place_segment_mismatch_rate": _mean(
                            [float(r.get("top1_place_segment_mismatch", 0.0)) for r in rows]
                        ),
                        "top1_interaction_object_match_rate": _mean(
                            [float(r.get("top1_interaction_object_match", 0.0)) for r in rows]
                        ),
                        "chain_step1_has_hit_rate": _mean([float(r.get("chain_step1_has_hit", 0.0)) for r in rows]),
                        "chain_step2_has_hit_rate": _mean([float(r.get("chain_step2_has_hit", 0.0)) for r in rows]),
                        "chain_success_rate": _mean([float(r.get("chain_success", 0.0)) for r in rows]),
                        "chain_filtered_ratio_step2": _mean(
                            [float(r.get("chain_filtered_ratio_step2", 0.0)) for r in rows]
                        ),
                        "chain_derived_time_used_rate": _mean([float(r.get("chain_derived_time_used", 0.0)) for r in rows]),
                        "chain_derived_place_used_rate": _mean([float(r.get("chain_derived_place_used", 0.0)) for r in rows]),
                        "chain_derived_object_used_rate": _mean([float(r.get("chain_derived_object_used", 0.0)) for r in rows]),
                        "chain_derived_nonempty_rate": _mean([float(r.get("chain_derived_nonempty", 0.0)) for r in rows]),
                        "backoff_used_rate": _mean([float(r.get("chain_backoff_used", 0.0)) for r in rows]),
                        "backoff_mean_level": _mean([float(r.get("chain_backoff_level", 0.0)) for r in rows]),
                        "backoff_exhausted_rate": _mean(
                            [1.0 if bool(r.get("chain_backoff_exhausted", False)) else 0.0 for r in rows]
                        ),
                        "chain_fail_step1_no_hit_rate": _mean(
                            [1.0 if str(r.get("chain_fail_reason", "")) == "step1_no_hit" else 0.0 for r in rows]
                        ),
                        "chain_fail_step2_no_hit_rate": _mean(
                            [1.0 if str(r.get("chain_fail_reason", "")) == "step2_no_hit" else 0.0 for r in rows]
                        ),
                        "chain_fail_constraints_over_filtered_rate": _mean(
                            [1.0 if str(r.get("chain_fail_reason", "")) == "constraints_over_filtered" else 0.0 for r in rows]
                        ),
                        "chain_fail_retrieval_distractor_rate": _mean(
                            [1.0 if str(r.get("chain_fail_reason", "")) == "retrieval_distractor" else 0.0 for r in rows]
                        ),
                        "chain_fail_evidence_missing_rate": _mean(
                            [1.0 if str(r.get("chain_fail_reason", "")) == "evidence_missing" else 0.0 for r in rows]
                        ),
                        "chain_fail_budget_insufficient_rate": _mean(
                            [1.0 if str(r.get("chain_fail_reason", "")) == "budget_insufficient" else 0.0 for r in rows]
                        ),
                        "chain_fail_backoff_exhausted_rate": _mean(
                            [1.0 if str(r.get("chain_fail_reason", "")) == "backoff_exhausted" else 0.0 for r in rows]
                        ),
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
                        **type_constraint_stats,
                        # Backward-compatible aliases.
                        "fallback_rate": float(type_constraint_stats.get("used_fallback_rate", 0.0)),
                        "constraint_apply_rate_after_scene_change": float(
                            type_constraint_stats.get("filtered_after_scene_change_rate", 0.0)
                        ),
                        "constraint_apply_rate_first_last": float(
                            type_constraint_stats.get("filtered_first_last_rate", 0.0)
                        ),
                        "constraint_apply_rate_type_match": float(
                            type_constraint_stats.get("filtered_type_match_rate", 0.0)
                        ),
                        "relax_rate_after_scene_change": float(
                            type_constraint_stats.get("relaxed_after_scene_change_rate", 0.0)
                        ),
                        "relax_rate_first_last": float(type_constraint_stats.get("relaxed_first_last_rate", 0.0)),
                        "relax_rate_type_match": float(type_constraint_stats.get("relaxed_type_match_rate", 0.0)),
                    }
                )

    return {
        "overall_rows": overall_rows,
        "by_query_type_rows": by_type_rows,
        "per_query_rows": per_query_rows,
    }
