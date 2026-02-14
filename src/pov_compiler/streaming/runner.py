from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.bench.nlq.datasets import NLQSample, load_hard_pseudo_nlq
from pov_compiler.bench.nlq.safety import classify_failure_reason
from pov_compiler.eval.budget_sweep import apply_budget
from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.memory.vector_index import VectorIndex
from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import QueryCandidate, QueryPlan, plan as plan_query
from pov_compiler.retrieval.reranker import Hit, rerank
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import EventV1, Output
from pov_compiler.streaming.budget_policy import (
    AdaptiveMinBudgetPolicy,
    BudgetSelection,
    BudgetSpec,
    FixedBudgetPolicy,
    RecommendedBudgetPolicy,
    SafetyLatencyInterventionBudgetPolicy,
    SafetyLatencyBudgetPolicy,
)
from pov_compiler.streaming.interventions import (
    InterventionState,
    STOP_ACTIONS,
    apply_intervention_action,
    choose_intervention_action,
    infer_failure_attribution,
    policy_action_order,
)


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


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-12:
        return arr
    return arr / denom


def _event_vec(event: EventV1, duration_s: float, max_evidence: int) -> np.ndarray:
    dur = max(1e-6, float(duration_s))
    t0 = float(event.t0)
    t1 = float(event.t1)
    span = max(1e-6, t1 - t0)
    density = float(event.scores.get("evidence_density", 0.0))
    contact_peak = float(event.scores.get("contact_peak", 0.0))
    boundary_conf = float(event.scores.get("boundary_conf", 0.0))
    ev_ratio = float(min(1.0, len(event.evidence) / max(1, max_evidence)))
    label_hash = (abs(hash(str(event.label))) % 997) / 997.0
    layer_hash = (abs(hash(str(event.meta.get("layer", "")))) % 997) / 997.0
    vec = np.asarray(
        [
            t0 / dur,
            t1 / dur,
            span / dur,
            density,
            contact_peak,
            boundary_conf,
            ev_ratio + label_hash * 0.1,
            layer_hash,
        ],
        dtype=np.float32,
    )
    return _l2_normalize(vec)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    try:
        return float(np.percentile(np.asarray(values, dtype=np.float32), float(q)))
    except Exception:
        values_sorted = sorted(values)
        idx = int((len(values_sorted) - 1) * max(0.0, min(1.0, float(q) / 100.0)))
        return float(values_sorted[idx])


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _slice_output(output: Output, end_t: float) -> Output:
    payload = output.model_dump() if hasattr(output, "model_dump") else output.dict()
    payload["events"] = [e for e in payload.get("events", []) if float(e.get("t0", 0.0)) <= float(end_t)]
    payload["events_v0"] = [e for e in payload.get("events_v0", []) if float(e.get("t0", 0.0)) <= float(end_t)]
    payload["events_v1"] = [e for e in payload.get("events_v1", []) if float(e.get("t0", 0.0)) <= float(end_t)]
    payload["highlights"] = [h for h in payload.get("highlights", []) if float(h.get("t0", 0.0)) <= float(end_t)]
    payload["decision_points"] = [
        d for d in payload.get("decision_points", []) if float(d.get("t0", 0.0)) <= float(end_t)
    ]
    token_codec = dict(payload.get("token_codec", {}))
    token_codec["tokens"] = [t for t in token_codec.get("tokens", []) if float(t.get("t0", 0.0)) <= float(end_t)]
    payload["token_codec"] = token_codec
    payload_meta = dict(payload.get("meta", {}))
    payload_meta["streaming_end_t"] = float(end_t)
    payload["meta"] = payload_meta
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _parse_hard_constraints(cfg: HardConstraintConfig | dict[str, Any] | None) -> HardConstraintConfig:
    if cfg is None:
        return HardConstraintConfig()
    if isinstance(cfg, HardConstraintConfig):
        return cfg
    if isinstance(cfg, dict):
        return HardConstraintConfig.from_dict(cfg)
    return HardConstraintConfig()


def _parse_rerank_cfg(cfg: WeightConfig | dict[str, Any] | str | Path | None) -> WeightConfig:
    return resolve_weight_config(cfg)


def _span_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = float(min(a[0], a[1])), float(max(a[0], a[1]))
    b0, b1 = float(min(b[0], b[1])), float(max(b[0], b[1]))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, (a1 - a0) + (b1 - b0) - inter)
    return float(inter / union)


_TOPK_PATTERN = re.compile(r"(?:^|\s)top_k=\d+(?:\s|$)")


def _force_top_k(query: str, top_k: int) -> str:
    text = str(query).strip()
    if not text:
        return text
    if _TOPK_PATTERN.search(text):
        text = _TOPK_PATTERN.sub(" ", text).strip()
    return f"{text} top_k={int(max(1, top_k))}"


def _build_plan(sample: NLQSample, allow_gt_fallback: bool) -> QueryPlan:
    planned = plan_query(sample.query)
    candidates: list[QueryCandidate] = []
    seen: set[str] = set()
    for cand in planned.candidates:
        q = _force_top_k(str(cand.get("query", "")), int(sample.top_k))
        if not q or q in seen:
            continue
        candidates.append(
            QueryCandidate(query=q, reason=str(cand.get("reason", "planned")), priority=int(cand.get("priority", 100)))
        )
        seen.add(q)

    if allow_gt_fallback:
        t0, t1 = float(sample.gt_span[0]), float(sample.gt_span[1])
        fallback = _force_top_k(f"time={max(0.0, t0 - 2.0):.3f}-{max(t0 + 0.2, t1 + 2.0):.3f}", int(sample.top_k))
        if fallback not in seen:
            candidates.append(QueryCandidate(query=fallback, reason="fallback_gt_time", priority=999))

    candidates.sort(key=lambda x: (int(x["priority"]), str(x["query"])))
    return QueryPlan(
        intent=planned.intent,
        candidates=candidates,
        constraints=dict(planned.constraints),
        debug=dict(planned.debug),
    )


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


def _evaluate_sample_once(
    *,
    sample: NLQSample,
    output: Output,
    index: VectorIndex,
    retrieval_config: dict[str, Any],
    hard_cfg: HardConstraintConfig,
    rerank_cfg: WeightConfig,
    allow_gt_fallback: bool,
) -> dict[str, Any]:
    retriever = Retriever(output_json=output, index=index, config=retrieval_config)
    query_plan = _build_plan(sample, allow_gt_fallback=allow_gt_fallback)
    candidate_queries = [str(cand["query"]) for cand in query_plan.candidates]
    merged_hits = retriever.retrieve_multi(candidate_queries)
    cresult = apply_constraints_detailed(
        merged_hits,
        query_plan=query_plan,
        cfg=hard_cfg,
        output=output,
    )
    reranked_hits = rerank(
        cresult.hits,
        plan=query_plan,
        context=output,
        cfg=rerank_cfg,
        distractors=sample.distractors,
    )
    pred_spans = _extract_pred_spans_from_hits(reranked_hits, top_k=int(sample.top_k))
    rank, best_iou = _rank_for_gt_span(pred_spans, sample.gt_span, min_iou=0.1)
    hit_at_k = 1.0 if rank is not None and rank <= int(sample.top_k) else 0.0
    hit_at_1 = 1.0 if rank == 1 else 0.0
    top1_bad = 1.0 if _top1_in_distractor(pred_spans, sample.distractors, min_iou=0.1) else 0.0
    hit_at_1_strict = 1.0 if hit_at_1 > 0.0 and top1_bad < 0.5 else 0.0
    hit_at_k_strict = 1.0 if hit_at_k > 0.0 and top1_bad < 0.5 else 0.0
    mrr = 1.0 / float(rank) if rank is not None else 0.0
    top1_kind = str(reranked_hits[0]["kind"]) if reranked_hits else "none"
    top1_source_query = str(reranked_hits[0]["source_query"]) if reranked_hits else ""
    return {
        "hit_at_k": float(hit_at_k),
        "hit_at_1": float(hit_at_1),
        "hit_at_1_strict": float(hit_at_1_strict),
        "hit_at_k_strict": float(hit_at_k_strict),
        "top1_in_distractor_rate": float(top1_bad),
        "fp_rate": float(top1_bad),
        "mrr": float(mrr),
        "rank": int(rank) if rank is not None else -1,
        "best_iou": float(best_iou),
        "top1_kind": top1_kind,
        "top1_source_query": top1_source_query,
        "chosen_plan_intent": str(query_plan.intent),
        "constraints_applied": list(cresult.applied),
        "constraints_relaxed": list(cresult.relaxed),
        "used_fallback": bool(cresult.used_fallback),
        "filtered_hits_before": int(cresult.filtered_before),
        "filtered_hits_after": int(cresult.filtered_after),
    }


def _samples_from_queries(output: Output, queries: list[str], top_k: int) -> list[NLQSample]:
    out: list[NLQSample] = []
    highlights = list(output.highlights)
    events = list(output.events_v1) if output.events_v1 else list(output.events)
    for idx, query in enumerate(queries, start=1):
        q = str(query).strip()
        if not q:
            continue
        if highlights:
            h = highlights[(idx - 1) % len(highlights)]
            gt_span = (float(h.t0), float(h.t1))
        elif events:
            e = events[(idx - 1) % len(events)]
            gt_span = (float(e.t0), float(e.t1))
        else:
            gt_span = (0.0, 0.2)
        out.append(
            NLQSample(
                qid=f"user_q_{idx:04d}",
                query=q,
                query_type="user_query",
                gt_span=gt_span,
                top_k=max(1, int(top_k)),
                distractors=[],
                meta={"source": "cli_query"},
            )
        )
    return out


def _derive_safety_for_trial(metrics: dict[str, Any], budget: BudgetSpec) -> tuple[int, str]:
    strict_hit = float(metrics.get("hit_at_k_strict", 0.0) or 0.0) > 0.0
    if strict_hit:
        return 0, ""
    row = {
        "filtered_hits_before": int(metrics.get("filtered_hits_before", 0) or 0),
        "filtered_hits_after": int(metrics.get("filtered_hits_after", 0) or 0),
        "candidate_count": int(metrics.get("filtered_hits_after", 0) or 0),
        "top1_in_distractor": float(metrics.get("top1_in_distractor_rate", 0.0) or 0.0),
        "budget_max_total_s": float(budget.max_total_s),
        "budget_max_tokens": int(budget.max_tokens),
        "budget_max_decisions": int(budget.max_decisions),
    }
    reason = str(classify_failure_reason(row))
    return 1, reason


def _hard_cfg_for_stage(base: HardConstraintConfig, stage: int) -> HardConstraintConfig:
    cfg = HardConstraintConfig.from_dict(base.to_dict())
    level = max(0, int(stage))
    if level >= 1:
        cfg.enable_after_scene_change = False
    if level >= 2:
        cfg.enable_first_last = False
    if level >= 3:
        cfg.enable_type_match = False
    return cfg


def _rerank_cfg_for_profile(base: WeightConfig, profile: str) -> WeightConfig:
    cfg = WeightConfig.from_dict(base.to_dict())
    mode = str(profile or "default").strip().lower()
    if mode == "anti_distractor":
        cfg.name = "anti_distractor"
        cfg.penalty_distractor_near = min(5.0, float(cfg.penalty_distractor_near) * 1.8)
        cfg.distractor_near_window_s = min(30.0, float(cfg.distractor_near_window_s) * 1.5)
        cfg.bonus_conf_scale = max(-5.0, float(cfg.bonus_conf_scale) * 0.8)
        cfg.bonus_boundary_scale = max(-5.0, float(cfg.bonus_boundary_scale) * 0.8)
        return cfg.validate()
    cfg.name = "default"
    return cfg.validate()


def _initial_budget_index(
    *,
    budgets_sorted: list[BudgetSpec],
    cfg: StreamingConfig,
) -> int:
    if bool(cfg.prefer_lower_budget):
        return 0
    if cfg.recommend_dir:
        try:
            rp = RecommendedBudgetPolicy(cfg.recommend_dir)
            rec_key = str(getattr(rp, "_top1_budget_key", "") or "").strip()
            if rec_key:
                for i, b in enumerate(budgets_sorted):
                    if b.key == rec_key:
                        return i
        except Exception:
            pass
    return max(0, int(len(budgets_sorted) // 2))


@dataclass
class StreamingConfig:
    step_s: float = 5.0
    top_k: int = 6
    queries: list[str] | None = None
    retrieval_config: dict[str, Any] | None = None
    # Budgeted online simulation mode.
    budgets: list[BudgetSpec] | None = None
    budget_policy: str = "fixed"
    fixed_budget: str | None = None
    recommend_dir: str | Path | None = None
    policy_gates: dict[str, Any] | None = None
    policy_targets: dict[str, Any] | None = None
    allow_gt_fallback: bool = False
    hard_constraints: HardConstraintConfig | dict[str, Any] | None = None
    rerank_cfg: WeightConfig | dict[str, Any] | str | Path | None = None
    nlq_mode: str = "hard_pseudo_nlq"
    nlq_seed: int = 0
    nlq_n_highlight: int = 10
    nlq_n_token: int = 10
    nlq_n_decision: int = 10
    latency_cap_ms: float = 25.0
    max_trials_per_query: int = 3
    strict_threshold: float = 1.0
    max_top1_in_distractor_rate: float = 0.2
    prefer_lower_budget: bool = True
    escalate_on_reasons: list[str] | None = None
    deescalate_on_latency: bool = True
    stop_on_non_budget_failure: bool = True
    mode: str = "basic"


def _make_step_points(duration_s: float, step_s: float) -> list[float]:
    points: list[float] = []
    t = float(step_s)
    while t < float(duration_s):
        points.append(float(t))
        t += float(step_s)
    if duration_s > 0:
        points.append(float(duration_s))
    if not points:
        points = [0.0]
    return points


def _run_streaming_basic(output: Output, cfg: StreamingConfig) -> dict[str, Any]:
    duration_s = float(output.meta.get("duration_s", 0.0))
    if duration_s <= 0.0 and output.events_v1:
        duration_s = max(float(ev.t1) for ev in output.events_v1)
    duration_s = max(duration_s, 0.0)

    queries = list(cfg.queries or ["anchor=turn_head top_k=6", "token=SCENE_CHANGE top_k=6"])
    if not queries:
        queries = ["anchor=turn_head top_k=6"]

    max_evidence = max((len(ev.evidence) for ev in output.events_v1), default=1)
    event_index = VectorIndex()
    indexed_ids: set[str] = set()
    step_s = max(0.5, float(cfg.step_s))
    step_points = _make_step_points(duration_s, step_s)

    all_latencies_ms: list[float] = []
    all_step_e2e_ms: list[float] = []
    step_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    progressive_rows: list[dict[str, Any]] = []

    prev_t = 0.0
    for step_idx, end_t in enumerate(step_points, start=1):
        step_start = time.perf_counter()
        added_this_step = 0
        for event in output.events_v1:
            if event.id in indexed_ids:
                continue
            if float(event.t1) <= float(end_t):
                event_index.add(
                    item_id=event.id,
                    vec=_event_vec(event, duration_s=max(duration_s, 1.0), max_evidence=max_evidence),
                    meta={
                        "kind": "event_v1",
                        "id": event.id,
                        "t0": float(event.t0),
                        "t1": float(event.t1),
                        "label": str(event.label),
                        "source_event": str(event.meta.get("source_event_id", event.id)),
                    },
                )
                indexed_ids.add(event.id)
                added_this_step += 1

        window_output = _slice_output(output, end_t=end_t)
        retriever = Retriever(output_json=window_output, index=event_index, config=dict(cfg.retrieval_config or {}))

        step_query_latencies: list[float] = []
        step_hits = 0
        for query in queries:
            q = str(query).strip()
            if not q:
                continue
            started = time.perf_counter()
            result = retriever.retrieve(q if "top_k=" in q else f"{q} top_k={int(cfg.top_k)}")
            latency_ms = float((time.perf_counter() - started) * 1000.0)
            step_query_latencies.append(latency_ms)
            all_latencies_ms.append(latency_ms)
            num_hits = (
                len(result.get("selected_events", []))
                + len(result.get("selected_highlights", []))
                + len(result.get("selected_tokens", []))
                + len(result.get("selected_decisions", []))
            )
            step_hits += num_hits
            query_rows.append(
                {
                    "step_idx": int(step_idx),
                    "t0_s": float(prev_t),
                    "t1_s": float(end_t),
                    "query": q,
                    "latency_ms": float(latency_ms),
                    "num_hits": int(num_hits),
                    "selected_events": int(len(result.get("selected_events", []))),
                    "selected_highlights": int(len(result.get("selected_highlights", []))),
                    "selected_tokens": int(len(result.get("selected_tokens", []))),
                    "selected_decisions": int(len(result.get("selected_decisions", []))),
                }
            )

        step_elapsed = max(1e-9, float(time.perf_counter() - step_start))
        step_e2e_ms = float(step_elapsed * 1000.0)
        all_step_e2e_ms.append(step_e2e_ms)
        qps = float(len(step_query_latencies) / step_elapsed) if step_query_latencies else 0.0
        step_rows.append(
            {
                "step_idx": int(step_idx),
                "t0_s": float(prev_t),
                "t1_s": float(end_t),
                "end_t": float(end_t),
                "index_size": int(event_index.size),
                "events_v1_added": int(added_this_step),
                "events_v1_indexed": int(len(indexed_ids)),
                "queries": int(len(step_query_latencies)),
                "hits_total": int(step_hits),
                "retrieval_latency_p50_ms": _percentile(step_query_latencies, 50.0),
                "retrieval_latency_p95_ms": _percentile(step_query_latencies, 95.0),
                "e2e_ms": step_e2e_ms,
                "throughput_qps": float(qps),
                "policy_name": "none",
                "chosen_budget_key_mode": "none",
                "avg_trials_per_query": 1.0 if step_query_latencies else 0.0,
            }
        )
        progressive_rows.append(
            {
                "step_idx": int(step_idx),
                "t0_s": float(prev_t),
                "t1_s": float(end_t),
                "events_v1_count": int(len(window_output.events_v1)),
                "highlights_count": int(len(window_output.highlights)),
                "tokens_count": int(len(window_output.token_codec.tokens)),
                "decisions_count": int(len(window_output.decision_points)),
            }
        )
        prev_t = float(end_t)

    summary = {
        "video_id": str(output.video_id),
        "duration_s": float(duration_s),
        "steps": int(len(step_rows)),
        "queries_total": int(len(query_rows)),
        "events_v1_total": int(len(output.events_v1)),
        "events_v1_indexed": int(len(indexed_ids)),
        "retrieval_latency_p50_ms": _percentile(all_latencies_ms, 50.0),
        "retrieval_latency_p95_ms": _percentile(all_latencies_ms, 95.0),
        "e2e_latency_p50_ms": _percentile(all_step_e2e_ms, 50.0),
        "e2e_latency_p95_ms": _percentile(all_step_e2e_ms, 95.0),
        "throughput_qps_mean": float(statistics.mean([float(r["throughput_qps"]) for r in step_rows]))
        if step_rows
        else 0.0,
        "policy_name": "none",
        "avg_trials_per_query": 1.0 if query_rows else 0.0,
    }
    return {"summary": summary, "step_rows": step_rows, "query_rows": query_rows, "progressive_rows": progressive_rows}


def _run_streaming_budgeted(output: Output, cfg: StreamingConfig) -> dict[str, Any]:
    duration_s = float(output.meta.get("duration_s", 0.0))
    if duration_s <= 0.0 and output.events_v1:
        duration_s = max(float(ev.t1) for ev in output.events_v1)
    duration_s = max(duration_s, 0.0)
    budgets = list(cfg.budgets or [])
    if not budgets:
        return _run_streaming_basic(output, cfg)

    budgets_sorted = sorted(budgets, key=lambda b: (float(b.max_total_s), int(b.max_tokens), int(b.max_decisions)))
    fixed_budget = str(cfg.fixed_budget or budgets_sorted[-1].key)
    policy_name = str(cfg.budget_policy or "fixed").strip().lower()
    if policy_name == "recommend":
        policy = RecommendedBudgetPolicy(cfg.recommend_dir or "")
    elif policy_name == "safety_latency_intervention":
        policy = SafetyLatencyInterventionBudgetPolicy(
            budgets=budgets_sorted,
            latency_cap_ms=float(cfg.latency_cap_ms),
            max_trials_per_query=int(cfg.max_trials_per_query),
            strict_threshold=float(cfg.strict_threshold),
            max_top1_in_distractor_rate=float(cfg.max_top1_in_distractor_rate),
            prefer_lower_budget=bool(cfg.prefer_lower_budget),
        )
    elif policy_name == "safety_latency":
        policy = SafetyLatencyBudgetPolicy(
            budgets=budgets_sorted,
            latency_cap_ms=float(cfg.latency_cap_ms),
            max_trials_per_query=int(cfg.max_trials_per_query),
            prefer_lower_budget=bool(cfg.prefer_lower_budget),
            escalate_on_reasons=list(cfg.escalate_on_reasons or ["budget_insufficient"]),
            deescalate_on_latency=bool(cfg.deescalate_on_latency),
            stop_on_non_budget_failure=bool(cfg.stop_on_non_budget_failure),
        )
    elif policy_name == "adaptive":
        default_gates = {
            "top1_in_distractor_rate": {"op": "<=", "value": 0.20},
            "fp_rate": {"op": "<=", "value": 0.20},
        }
        default_targets = {"hit_at_k_strict": {"op": ">=", "value": 0.60}}
        policy = AdaptiveMinBudgetPolicy(
            budgets_sorted=budgets_sorted,
            gates=dict(cfg.policy_gates or default_gates),
            targets=dict(cfg.policy_targets or default_targets),
            allow_missing=False,
        )
    else:
        policy_name = "fixed"
        policy = FixedBudgetPolicy(fixed_budget)

    hard_cfg = _parse_hard_constraints(cfg.hard_constraints)
    rerank_cfg = _parse_rerank_cfg(cfg.rerank_cfg)

    max_evidence = max((len(ev.evidence) for ev in output.events_v1), default=1)
    event_index = VectorIndex()
    indexed_ids: set[str] = set()
    step_s = max(0.5, float(cfg.step_s))
    step_points = _make_step_points(duration_s, step_s)

    samples_all: list[NLQSample]
    if cfg.queries:
        samples_all = _samples_from_queries(output, list(cfg.queries), top_k=max(1, int(cfg.top_k)))
    elif str(cfg.nlq_mode).strip().lower() == "hard_pseudo_nlq":
        samples_all = load_hard_pseudo_nlq(
            output,
            seed=int(cfg.nlq_seed),
            n_highlight=max(1, int(cfg.nlq_n_highlight)),
            n_token=max(1, int(cfg.nlq_n_token)),
            n_decision=max(1, int(cfg.nlq_n_decision)),
            top_k=max(1, int(cfg.top_k)),
        )
    else:
        # Keep a safe default path for non-hard mode.
        samples_all = load_hard_pseudo_nlq(
            output,
            seed=int(cfg.nlq_seed),
            n_highlight=max(1, int(cfg.nlq_n_highlight)),
            n_token=max(1, int(cfg.nlq_n_token)),
            n_decision=max(1, int(cfg.nlq_n_decision)),
            top_k=max(1, int(cfg.top_k)),
        )

    all_retrieval_lat_ms: list[float] = []
    all_step_e2e_ms: list[float] = []
    all_query_e2e_ms: list[float] = []
    all_trials: list[int] = []
    step_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    progressive_rows: list[dict[str, Any]] = []

    prev_t = 0.0
    for step_idx, end_t in enumerate(step_points, start=1):
        step_start = time.perf_counter()
        added_this_step = 0
        for event in output.events_v1:
            if event.id in indexed_ids:
                continue
            if float(event.t1) <= float(end_t):
                event_index.add(
                    item_id=event.id,
                    vec=_event_vec(event, duration_s=max(duration_s, 1.0), max_evidence=max_evidence),
                    meta={
                        "kind": "event_v1",
                        "id": event.id,
                        "t0": float(event.t0),
                        "t1": float(event.t1),
                        "label": str(event.label),
                        "source_event": str(event.meta.get("source_event_id", event.id)),
                    },
                )
                indexed_ids.add(event.id)
                added_this_step += 1

        window_output = _slice_output(output, end_t=end_t)
        budgeted_cache: dict[str, Output] = {}
        step_query_rows: list[dict[str, Any]] = []

        eligible_samples = [s for s in samples_all if float(s.gt_span[1]) <= float(end_t) + 1e-6]
        for sample in eligible_samples:
            query_started = time.perf_counter()
            trial_cache: dict[str, dict[str, Any]] = {}
            trial_records: list[dict[str, Any]] = []

            def _eval_budget(
                budget: BudgetSpec,
                *,
                constraints_stage: int = 0,
                rerank_profile: str = "default",
                expand_level: int = 0,
                widen_window: bool = False,
            ) -> dict[str, Any]:
                key = f"{budget.key}|c{int(constraints_stage)}|r{str(rerank_profile)}|e{int(expand_level)}|w{int(bool(widen_window))}"
                cached = trial_cache.get(key)
                if cached is not None:
                    return dict(cached)
                if budget.key not in budgeted_cache:
                    budgeted_cache[budget.key] = apply_budget(
                        window_output,
                        budget={
                            "max_total_s": float(budget.max_total_s),
                            "max_tokens": int(budget.max_tokens),
                            "max_decisions": int(budget.max_decisions),
                        },
                    )
                eval_start = time.perf_counter()
                hard_cfg_eval = _hard_cfg_for_stage(hard_cfg, constraints_stage)
                rerank_cfg_eval = _rerank_cfg_for_profile(rerank_cfg, rerank_profile)
                sample_eval = NLQSample(
                    qid=str(sample.qid),
                    query=str(sample.query),
                    query_type=str(sample.query_type),
                    gt_span=(float(sample.gt_span[0]), float(sample.gt_span[1])),
                    top_k=max(1, int(sample.top_k) + max(0, int(expand_level)) * 2),
                    distractors=list(sample.distractors),
                    meta=dict(sample.meta),
                )
                metrics = _evaluate_sample_once(
                    sample=sample_eval,
                    output=budgeted_cache[budget.key],
                    index=event_index,
                    retrieval_config=dict(cfg.retrieval_config or {}),
                    hard_cfg=hard_cfg_eval,
                    rerank_cfg=rerank_cfg_eval,
                    allow_gt_fallback=bool(cfg.allow_gt_fallback) or bool(widen_window),
                )
                eval_ms = float((time.perf_counter() - eval_start) * 1000.0)
                retrieval_ms = float(metrics.get("retrieval_ms", eval_ms))
                metrics["retrieval_ms"] = retrieval_ms
                metrics["latency_retrieval_ms"] = retrieval_ms
                metrics["latency_e2e_ms"] = eval_ms
                safety_is, safety_reason = _derive_safety_for_trial(metrics, budget)
                metrics["safety_is_critical_fn"] = int(safety_is)
                metrics["safety_reason"] = safety_reason
                metrics["strict_hit_at_k"] = float(metrics.get("hit_at_k_strict", 0.0))
                metrics["strict_hit_at_1"] = float(metrics.get("hit_at_1_strict", 0.0))
                metrics["constraints_stage"] = int(constraints_stage)
                metrics["rerank_profile"] = str(rerank_profile)
                metrics["expand_level"] = int(expand_level)
                metrics["widen_evidence_window"] = bool(widen_window)
                metrics["hard_cfg"] = hard_cfg_eval.to_dict()
                metrics["rerank_cfg_name"] = str(rerank_cfg_eval.name)
                metrics["rerank_cfg_hash"] = str(rerank_cfg_eval.short_hash())
                trial_cache[key] = dict(metrics)
                return dict(metrics)

            if policy_name == "safety_latency_intervention":
                state = InterventionState(budget_idx=_initial_budget_index(budgets_sorted=budgets_sorted, cfg=cfg))
                final_metrics: dict[str, Any] = {}
                status = "no_budget_passed"
                reason = "max_trials_reached"
                action_final = "give_up_max_trials"
                action_reason_final = "max_trials_reached"
                chosen_budget = budgets_sorted[state.budget_idx]
                for trial_idx in range(1, max(1, int(cfg.max_trials_per_query)) + 1):
                    budget_before = budgets_sorted[state.budget_idx]
                    config_before = state.to_config()
                    metrics = _eval_budget(
                        budget_before,
                        constraints_stage=int(state.constraints_stage),
                        rerank_profile=str(state.rerank_profile),
                        expand_level=int(state.expand_level),
                        widen_window=bool(state.widen_evidence_window),
                    )
                    strict_hit = float(metrics.get("hit_at_k_strict", 0.0) or 0.0) >= float(cfg.strict_threshold)
                    top1_rate = float(metrics.get("top1_in_distractor_rate", 1.0) or 1.0)
                    latency_now = float(metrics.get("latency_e2e_ms", 0.0) or 0.0)
                    attribution = infer_failure_attribution(metrics, budget_before)
                    action, action_reason = choose_intervention_action(
                        state=state,
                        attribution=attribution,
                        strict_hit=strict_hit,
                        top1_in_distractor_rate=top1_rate,
                        latency_e2e_ms=latency_now,
                        latency_cap_ms=float(cfg.latency_cap_ms),
                        trial_idx=int(trial_idx),
                        max_trials=max(1, int(cfg.max_trials_per_query)),
                        max_top1_in_distractor_rate=float(cfg.max_top1_in_distractor_rate),
                        budgets=budgets_sorted,
                    )
                    updated_state = apply_intervention_action(
                        state=state,
                        action=action,
                        budgets=budgets_sorted,
                    )
                    budget_after = budgets_sorted[updated_state.budget_idx]
                    success = bool(
                        strict_hit
                        and top1_rate <= float(cfg.max_top1_in_distractor_rate)
                        and latency_now <= float(cfg.latency_cap_ms)
                    )
                    trial_records.append(
                        {
                            "trial_index": int(trial_idx),
                            "budget_key": str(budget_before.key),
                            "budget_seconds": float(budget_before.seconds),
                            "budget_before": str(budget_before.key),
                            "budget_after": str(budget_after.key),
                            "config_before": config_before,
                            "config_after": updated_state.to_config(),
                            "action": str(action),
                            "action_reason": str(action_reason),
                            "attribution": str(attribution or ""),
                            "status": "success" if success else "retry",
                            "success": 1 if success else 0,
                            "metrics": dict(metrics),
                        }
                    )
                    final_metrics = dict(metrics)
                    chosen_budget = budget_after if action in {"escalate_budget", "deescalate_budget"} else budget_before
                    action_final = str(action)
                    action_reason_final = str(action_reason)
                    if success:
                        status = "ok"
                        reason = "intervention_success"
                        break
                    if action in {"accept"}:
                        status = "ok"
                        reason = "accepted_without_threshold"
                        break
                    if action in STOP_ACTIONS:
                        status = "no_budget_passed"
                        reason = str(action_reason)
                        break
                    state = updated_state
                selection = BudgetSelection(
                    chosen_budget_key=str(chosen_budget.key),
                    chosen_budget_seconds=float(chosen_budget.seconds),
                    trials_count=len(trial_records),
                    tried_budget_keys=[str(x.get("budget_key", "")) for x in trial_records],
                    status=str(status),
                    reason=str(reason),
                    metrics=final_metrics,
                    action=action_final,
                    action_reason=action_reason_final,
                    trial_records=trial_records,
                )
            else:
                selection = policy.select(
                    budgets=budgets_sorted,
                    evaluate_budget=_eval_budget,
                    query_context={"qid": sample.qid, "query_type": sample.query_type},
                )
            selected_metrics = dict(selection.metrics)
            q_e2e_ms = float((time.perf_counter() - query_started) * 1000.0)
            selected_metrics["query_total_e2e_ms"] = q_e2e_ms

            if selection.trial_records:
                trial_records = [dict(x) for x in selection.trial_records]
            else:
                for i, key in enumerate(selection.tried_budget_keys, start=1):
                    spec = next((b for b in budgets_sorted if b.key == key), None)
                    if spec is None:
                        continue
                    trial_records.append(
                        {
                            "trial_index": int(i),
                            "budget_key": spec.key,
                            "budget_seconds": float(spec.seconds),
                            "action": "continue" if i < len(selection.tried_budget_keys) else (selection.action or "accept"),
                            "action_reason": selection.reason,
                            "status": selection.status,
                            "metrics": dict(trial_cache.get(spec.key, selected_metrics)),
                        }
                    )
            final_trial_idx = int(len(trial_records)) if trial_records else 1

            retrieval_ms = float(selected_metrics.get("retrieval_ms", 0.0))
            all_retrieval_lat_ms.append(retrieval_ms)
            all_query_e2e_ms.append(q_e2e_ms)
            all_trials.append(int(selection.trials_count))
            for trial in trial_records:
                trial_index = int(trial.get("trial_index", 0) or 0)
                trial_metrics = dict(trial.get("metrics", {}))
                bkey = str(trial.get("budget_key", selection.chosen_budget_key))
                bspec = next((b for b in budgets_sorted if b.key == bkey), None)
                bsec = float(trial.get("budget_seconds", bspec.seconds if bspec else selection.chosen_budget_seconds))
                btok = int(bspec.max_tokens if bspec else 0)
                bdec = int(bspec.max_decisions if bspec else 0)
                row = {
                    "step_idx": int(step_idx),
                    "t0_s": float(prev_t),
                    "t1_s": float(end_t),
                    "query_id": str(sample.qid),
                    "query_type": str(sample.query_type),
                    "query_text": str(sample.query),
                    "gt_t0": float(sample.gt_span[0]),
                    "gt_t1": float(sample.gt_span[1]),
                    "policy_name": str(policy_name),
                    "budget_seconds": bsec,
                    "budget_tokens": btok,
                    "budget_decisions": bdec,
                    "budget_key": bkey,
                    "chosen_budget_key": str(selection.chosen_budget_key),
                    "chosen_budget_seconds": float(selection.chosen_budget_seconds),
                    "budget_before": str(trial.get("budget_before", bkey)),
                    "budget_after": str(trial.get("budget_after", bkey)),
                    "trial_index": trial_index,
                    "trial_idx": trial_index,
                    "trial_count_for_query": int(selection.trials_count),
                    "trials_count": int(selection.trials_count),
                    "status": str(selection.status),
                    "reason": str(selection.reason),
                    "tried_budget_keys": json.dumps(selection.tried_budget_keys, ensure_ascii=False),
                    "action": str(trial.get("action", "accept")),
                    "action_reason": str(trial.get("action_reason", selection.reason)),
                    "intervention_action": str(trial.get("action", "accept")),
                    "attribution": str(trial.get("attribution", trial_metrics.get("safety_reason", ""))),
                    "config_before": json.dumps(trial.get("config_before", {}), ensure_ascii=False),
                    "config_after": json.dumps(trial.get("config_after", {}), ensure_ascii=False),
                    "strict_hit_at_k": float(trial_metrics.get("strict_hit_at_k", trial_metrics.get("hit_at_k_strict", 0.0))),
                    "strict_hit_at_1": float(trial_metrics.get("strict_hit_at_1", trial_metrics.get("hit_at_1_strict", 0.0))),
                    "hit_at_k_strict": float(trial_metrics.get("hit_at_k_strict", 0.0)),
                    "hit_at_1_strict": float(trial_metrics.get("hit_at_1_strict", 0.0)),
                    "top1_in_distractor_rate": float(trial_metrics.get("top1_in_distractor_rate", 0.0)),
                    "fp_rate": float(trial_metrics.get("fp_rate", 0.0)),
                    "mrr": float(trial_metrics.get("mrr", 0.0)),
                    "top1_kind": str(trial_metrics.get("top1_kind", "")),
                    "top1_source_query": str(trial_metrics.get("top1_source_query", "")),
                    "chosen_plan_intent": str(trial_metrics.get("chosen_plan_intent", "")),
                    "safety_is_critical_fn": int(trial_metrics.get("safety_is_critical_fn", 0)),
                    "safety_reason": str(trial_metrics.get("safety_reason", "")),
                    "latency_retrieval_ms": float(trial_metrics.get("latency_retrieval_ms", trial_metrics.get("retrieval_ms", 0.0))),
                    "latency_e2e_ms": float(trial_metrics.get("latency_e2e_ms", trial_metrics.get("query_total_e2e_ms", 0.0))),
                    "retrieval_ms": float(trial_metrics.get("retrieval_ms", 0.0)),
                    "e2e_ms": float(trial_metrics.get("latency_e2e_ms", 0.0)),
                    "success": int(
                        trial.get(
                            "success",
                            1
                            if (
                                float(trial_metrics.get("hit_at_k_strict", 0.0)) >= float(cfg.strict_threshold)
                                and float(trial_metrics.get("top1_in_distractor_rate", 1.0))
                                <= float(cfg.max_top1_in_distractor_rate)
                            )
                            else 0,
                        )
                    ),
                    "final_trial": 1 if trial_index == final_trial_idx else 0,
                    "final_trial_idx": int(final_trial_idx),
                    "final_action": str(selection.action),
                    "final_budget": str(selection.chosen_budget_key),
                    "final_success": (
                        int(
                            trial.get(
                                "success",
                                1
                                if (
                                    float(trial_metrics.get("hit_at_k_strict", 0.0)) >= float(cfg.strict_threshold)
                                    and float(trial_metrics.get("top1_in_distractor_rate", 1.0))
                                    <= float(cfg.max_top1_in_distractor_rate)
                                )
                                else 0,
                            )
                        )
                        if trial_index == final_trial_idx
                        else 0
                    ),
                }
                step_query_rows.append(row)
                query_rows.append(dict(row))

        step_elapsed = max(1e-9, float(time.perf_counter() - step_start))
        step_e2e_ms = float(step_elapsed * 1000.0)
        all_step_e2e_ms.append(step_e2e_ms)
        qps = float(len(step_query_rows) / step_elapsed) if step_query_rows else 0.0
        final_rows = [x for x in step_query_rows if int(x.get("final_trial", 0)) == 1]
        metric_rows = final_rows if final_rows else step_query_rows
        step_rows.append(
            {
                "step_idx": int(step_idx),
                "t0_s": float(prev_t),
                "t1_s": float(end_t),
                "end_t": float(end_t),
                "index_size": int(event_index.size),
                "events_v1_added": int(added_this_step),
                "events_v1_indexed": int(len(indexed_ids)),
                "queries": int(len(final_rows)),
                "retrieval_latency_p50_ms": _percentile([float(x["latency_retrieval_ms"]) for x in step_query_rows], 50.0),
                "retrieval_latency_p95_ms": _percentile([float(x["latency_retrieval_ms"]) for x in step_query_rows], 95.0),
                "query_e2e_p50_ms": _percentile([float(x["latency_e2e_ms"]) for x in step_query_rows], 50.0),
                "query_e2e_p95_ms": _percentile([float(x["latency_e2e_ms"]) for x in step_query_rows], 95.0),
                "latency_e2e_p50_ms": _percentile([float(x["latency_e2e_ms"]) for x in step_query_rows], 50.0),
                "latency_e2e_p95_ms": _percentile([float(x["latency_e2e_ms"]) for x in step_query_rows], 95.0),
                "e2e_ms": float(step_e2e_ms),
                "throughput_qps": float(qps),
                "policy_name": str(policy_name),
                "chosen_budget_key_mode": str(policy_name),
                "avg_trials_per_query": _mean([float(x["trial_count_for_query"]) for x in final_rows]),
                "avg_trials_per_query_step": _mean([float(x["trial_count_for_query"]) for x in final_rows]),
                "hit_at_k_strict": _mean([float(x["hit_at_k_strict"]) for x in metric_rows]),
                "hit_at_1_strict": _mean([float(x["hit_at_1_strict"]) for x in metric_rows]),
                "top1_in_distractor_rate": _mean([float(x["top1_in_distractor_rate"]) for x in metric_rows]),
                "fp_rate": _mean([float(x["fp_rate"]) for x in metric_rows]),
                "mrr": _mean([float(x["mrr"]) for x in metric_rows]),
                "safety_critical_fn_rate_step": _mean([float(x["safety_is_critical_fn"]) for x in metric_rows]),
                "safety_budget_insufficient_rate_step": _mean(
                    [1.0 if str(x.get("safety_reason", "")) == "budget_insufficient" else 0.0 for x in metric_rows]
                ),
            }
        )
        progressive_rows.append(
            {
                "step_idx": int(step_idx),
                "t0_s": float(prev_t),
                "t1_s": float(end_t),
                "events_v1_count": int(len(window_output.events_v1)),
                "highlights_count": int(len(window_output.highlights)),
                "tokens_count": int(len(window_output.token_codec.tokens)),
                "decisions_count": int(len(window_output.decision_points)),
            }
        )
        prev_t = float(end_t)

    final_query_rows = [r for r in query_rows if int(r.get("final_trial", 0)) == 1]
    metric_rows = final_query_rows if final_query_rows else query_rows
    action_counts: dict[str, int] = {}
    attribution_counts: dict[str, int] = {}
    for row in query_rows:
        action = str(row.get("action", "")).strip()
        if not action:
            continue
        action_counts[action] = action_counts.get(action, 0) + 1
        attr = str(row.get("attribution", "")).strip()
        if attr:
            attribution_counts[attr] = attribution_counts.get(attr, 0) + 1
    intervention_rows = [r for r in query_rows if str(r.get("action", "")) not in {"", "accept"}]
    intervention_success = _mean([float(r.get("success", 0.0)) for r in intervention_rows]) if intervention_rows else 0.0
    summary = {
        "video_id": str(output.video_id),
        "duration_s": float(duration_s),
        "steps": int(len(step_rows)),
        "queries_total": int(len(final_query_rows)),
        "query_trials_total": int(len(query_rows)),
        "events_v1_total": int(len(output.events_v1)),
        "events_v1_indexed": int(len(indexed_ids)),
        "retrieval_latency_p50_ms": _percentile(all_retrieval_lat_ms, 50.0),
        "retrieval_latency_p95_ms": _percentile(all_retrieval_lat_ms, 95.0),
        "e2e_latency_p50_ms": _percentile(all_step_e2e_ms, 50.0),
        "e2e_latency_p95_ms": _percentile(all_step_e2e_ms, 95.0),
        "query_e2e_p50_ms": _percentile(all_query_e2e_ms, 50.0),
        "query_e2e_p95_ms": _percentile(all_query_e2e_ms, 95.0),
        "throughput_qps_mean": float(statistics.mean([float(r["throughput_qps"]) for r in step_rows]))
        if step_rows
        else 0.0,
        "policy_name": str(policy_name),
        "avg_trials_per_query": _mean([float(x) for x in all_trials]),
        "hit_at_k_strict": _mean([float(r.get("hit_at_k_strict", 0.0)) for r in metric_rows]),
        "hit_at_1_strict": _mean([float(r.get("hit_at_1_strict", 0.0)) for r in metric_rows]),
        "top1_in_distractor_rate": _mean([float(r.get("top1_in_distractor_rate", 0.0)) for r in metric_rows]),
        "fp_rate": _mean([float(r.get("fp_rate", 0.0)) for r in metric_rows]),
        "mrr": _mean([float(r.get("mrr", 0.0)) for r in metric_rows]),
        "safety_critical_fn_rate": _mean([float(r.get("safety_is_critical_fn", 0.0)) for r in metric_rows]),
        "safety_budget_insufficient_rate": _mean(
            [1.0 if str(r.get("safety_reason", "")) == "budget_insufficient" else 0.0 for r in metric_rows]
        ),
        "num_escalate": int(sum(v for k, v in action_counts.items() if str(k).startswith("escalate_"))),
        "num_deescalate": int(sum(v for k, v in action_counts.items() if str(k).startswith("deescalate_"))),
        "num_accept": int(action_counts.get("accept", 0)),
        "num_give_up": int(sum(v for k, v in action_counts.items() if str(k).startswith("give_up"))),
        "intervention_trials_total": int(len(intervention_rows)),
        "intervention_success_rate": float(intervention_success),
        "attribution_counts": attribution_counts,
        "budgets": [b.key for b in budgets_sorted],
        "latency_cap_ms": float(cfg.latency_cap_ms),
        "max_trials_per_query": int(cfg.max_trials_per_query),
        "escalate_on_reasons": list(cfg.escalate_on_reasons or ["budget_insufficient"]),
        "policy_action_counts": action_counts,
        "policy_action_order": policy_action_order() if policy_name == "safety_latency_intervention" else {},
        "strict_threshold": float(cfg.strict_threshold),
        "max_top1_in_distractor_rate": float(cfg.max_top1_in_distractor_rate),
    }
    return {"summary": summary, "step_rows": step_rows, "query_rows": query_rows, "progressive_rows": progressive_rows}


def run_streaming(
    output_json: str | Path | dict[str, Any] | Output,
    *,
    config: StreamingConfig | None = None,
) -> dict[str, Any]:
    cfg = config or StreamingConfig()
    output = ensure_events_v1(_as_output(output_json))
    if cfg.budgets:
        return _run_streaming_budgeted(output, cfg)
    return _run_streaming_basic(output, cfg)
