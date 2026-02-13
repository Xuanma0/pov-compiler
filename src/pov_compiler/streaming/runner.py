from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.memory.vector_index import VectorIndex
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import EventV1, Output


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


@dataclass
class StreamingConfig:
    step_s: float = 5.0
    top_k: int = 6
    queries: list[str] | None = None
    retrieval_config: dict[str, Any] | None = None


def run_streaming(
    output_json: str | Path | dict[str, Any] | Output,
    *,
    config: StreamingConfig | None = None,
) -> dict[str, Any]:
    cfg = config or StreamingConfig()
    output = ensure_events_v1(_as_output(output_json))
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
    step_points: list[float] = []
    t = step_s
    while t < duration_s:
        step_points.append(float(t))
        t += step_s
    if duration_s > 0:
        step_points.append(float(duration_s))
    if not step_points:
        step_points = [0.0]

    all_latencies_ms: list[float] = []
    all_step_e2e_ms: list[float] = []
    step_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    progressive_rows: list[dict[str, Any]] = []

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
        retriever = Retriever(
            output_json=window_output,
            index=event_index,
            config=dict(cfg.retrieval_config or {}),
        )

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
                    "end_t": float(end_t),
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
            }
        )
        progressive_rows.append(
            {
                "step_idx": int(step_idx),
                "end_t": float(end_t),
                "events_v1_count": int(len(window_output.events_v1)),
                "highlights_count": int(len(window_output.highlights)),
                "tokens_count": int(len(window_output.token_codec.tokens)),
                "decisions_count": int(len(window_output.decision_points)),
            }
        )

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
    }
    return {
        "summary": summary,
        "step_rows": step_rows,
        "query_rows": query_rows,
        "progressive_rows": progressive_rows,
    }
