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
    if str(cfg.nlq_mode).strip().lower() == "hard_pseudo_nlq":
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

            def _eval_budget(budget: BudgetSpec) -> dict[str, Any]:
                key = budget.key
                cached = trial_cache.get(key)
                if cached is not None:
                    return dict(cached)
                if key not in budgeted_cache:
                    budgeted_cache[key] = apply_budget(
                        window_output,
                        budget={
                            "max_total_s": float(budget.max_total_s),
                            "max_tokens": int(budget.max_tokens),
                            "max_decisions": int(budget.max_decisions),
                        },
                    )
                eval_start = time.perf_counter()
                metrics = _evaluate_sample_once(
                    sample=sample,
                    output=budgeted_cache[key],
                    index=event_index,
                    retrieval_config=dict(cfg.retrieval_config or {}),
                    hard_cfg=hard_cfg,
                    rerank_cfg=rerank_cfg,
                    allow_gt_fallback=bool(cfg.allow_gt_fallback),
                )
                retrieval_ms = float((time.perf_counter() - eval_start) * 1000.0)
                metrics["retrieval_ms"] = retrieval_ms
                trial_cache[key] = dict(metrics)
                return dict(metrics)

            selection: BudgetSelection = policy.select(
                budgets=budgets_sorted,
                evaluate_budget=_eval_budget,
                query_context={"qid": sample.qid, "query_type": sample.query_type},
            )
            selected_metrics = dict(selection.metrics)
            q_e2e_ms = float((time.perf_counter() - query_started) * 1000.0)

            retrieval_ms = float(selected_metrics.get("retrieval_ms", 0.0))
            all_retrieval_lat_ms.append(retrieval_ms)
            all_query_e2e_ms.append(q_e2e_ms)
            all_trials.append(int(selection.trials_count))

            row = {
                "step_idx": int(step_idx),
                "t0_s": float(prev_t),
                "t1_s": float(end_t),
                "query_id": str(sample.qid),
                "query_type": str(sample.query_type),
                "query_text": str(sample.query),
                "gt_t0": float(sample.gt_span[0]),
                "gt_t1": float(sample.gt_span[1]),
                "chosen_budget_key": str(selection.chosen_budget_key),
                "chosen_budget_seconds": float(selection.chosen_budget_seconds),
                "trials_count": int(selection.trials_count),
                "status": str(selection.status),
                "reason": str(selection.reason),
                "tried_budget_keys": json.dumps(selection.tried_budget_keys, ensure_ascii=False),
                "hit_at_k_strict": float(selected_metrics.get("hit_at_k_strict", 0.0)),
                "hit_at_1_strict": float(selected_metrics.get("hit_at_1_strict", 0.0)),
                "top1_in_distractor_rate": float(selected_metrics.get("top1_in_distractor_rate", 0.0)),
                "fp_rate": float(selected_metrics.get("fp_rate", 0.0)),
                "mrr": float(selected_metrics.get("mrr", 0.0)),
                "top1_kind": str(selected_metrics.get("top1_kind", "")),
                "top1_source_query": str(selected_metrics.get("top1_source_query", "")),
                "chosen_plan_intent": str(selected_metrics.get("chosen_plan_intent", "")),
                "retrieval_ms": retrieval_ms,
                "e2e_ms": q_e2e_ms,
            }
            step_query_rows.append(row)
            query_rows.append(dict(row))

        step_elapsed = max(1e-9, float(time.perf_counter() - step_start))
        step_e2e_ms = float(step_elapsed * 1000.0)
        all_step_e2e_ms.append(step_e2e_ms)
        qps = float(len(step_query_rows) / step_elapsed) if step_query_rows else 0.0
        step_rows.append(
            {
                "step_idx": int(step_idx),
                "t0_s": float(prev_t),
                "t1_s": float(end_t),
                "end_t": float(end_t),
                "index_size": int(event_index.size),
                "events_v1_added": int(added_this_step),
                "events_v1_indexed": int(len(indexed_ids)),
                "queries": int(len(step_query_rows)),
                "retrieval_latency_p50_ms": _percentile([float(x["retrieval_ms"]) for x in step_query_rows], 50.0),
                "retrieval_latency_p95_ms": _percentile([float(x["retrieval_ms"]) for x in step_query_rows], 95.0),
                "query_e2e_p50_ms": _percentile([float(x["e2e_ms"]) for x in step_query_rows], 50.0),
                "query_e2e_p95_ms": _percentile([float(x["e2e_ms"]) for x in step_query_rows], 95.0),
                "e2e_ms": float(step_e2e_ms),
                "throughput_qps": float(qps),
                "policy_name": str(policy_name),
                "chosen_budget_key_mode": str(policy_name),
                "avg_trials_per_query": _mean([float(x["trials_count"]) for x in step_query_rows]),
                "hit_at_k_strict": _mean([float(x["hit_at_k_strict"]) for x in step_query_rows]),
                "hit_at_1_strict": _mean([float(x["hit_at_1_strict"]) for x in step_query_rows]),
                "top1_in_distractor_rate": _mean([float(x["top1_in_distractor_rate"]) for x in step_query_rows]),
                "fp_rate": _mean([float(x["fp_rate"]) for x in step_query_rows]),
                "mrr": _mean([float(x["mrr"]) for x in step_query_rows]),
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
        "hit_at_k_strict": _mean([float(r.get("hit_at_k_strict", 0.0)) for r in query_rows]),
        "hit_at_1_strict": _mean([float(r.get("hit_at_1_strict", 0.0)) for r in query_rows]),
        "top1_in_distractor_rate": _mean([float(r.get("top1_in_distractor_rate", 0.0)) for r in query_rows]),
        "fp_rate": _mean([float(r.get("fp_rate", 0.0)) for r in query_rows]),
        "mrr": _mean([float(r.get("mrr", 0.0)) for r in query_rows]),
        "budgets": [b.key for b in budgets_sorted],
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
