from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.features.embeddings import Embedder
from pov_compiler.io.video_reader import VideoReader
from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.memory.vector_index import VectorIndex
from pov_compiler.schemas import Output, Token


@dataclass
class IndexBuilderConfig:
    sample_fps: float = 4.0
    max_frames_per_segment: int = 24
    use_clip: bool = False


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
    arr = vec.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-12:
        return arr
    return arr / denom


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _token_types_for_range(tokens: list[Token], t0: float, t1: float) -> list[str]:
    types = sorted({token.type for token in tokens if _overlap(token.t0, token.t1, t0, t1)})
    return types


def _anchor_types_for_event_v1(event: Any) -> list[str]:
    out: set[str] = set()
    for evidence in getattr(event, "evidence", []):
        if str(getattr(evidence, "type", "")) == "anchor":
            anchor_type = str(getattr(evidence, "source", {}).get("anchor_type", "")).strip()
            if anchor_type:
                out.add(anchor_type)
        if str(getattr(evidence, "type", "")) == "highlight":
            values = getattr(evidence, "source", {}).get("anchor_types", [])
            if isinstance(values, list):
                for value in values:
                    if str(value).strip():
                        out.add(str(value).strip())
    return sorted(out)


def _nearest_event_id_for_span(output: Output, t0_s: float, t1_s: float) -> str:
    best_id = ""
    best_dist = float("inf")
    center = 0.5 * (float(t0_s) + float(t1_s))
    for ev in list(output.events_v1 or []):
        c = 0.5 * (float(ev.t0) + float(ev.t1))
        dist = abs(c - center)
        if dist < best_dist:
            best_dist = dist
            best_id = str(ev.id)
    if best_id:
        return best_id
    for ev in list(output.events or []):
        c = 0.5 * (float(ev.t0) + float(ev.t1))
        dist = abs(c - center)
        if dist < best_dist:
            best_dist = dist
            best_id = str(ev.id)
    return best_id


def _decision_pool(output: Output) -> tuple[str, list[dict[str, Any]]]:
    model_rows = list(getattr(output, "decisions_model_v1", []) or [])
    pool_model: list[dict[str, Any]] = []
    for i, item in enumerate(model_rows, start=1):
        if not isinstance(item, dict):
            continue
        try:
            t0_ms = int(round(float(item.get("t0_ms", 0))))
            t1_ms = int(round(float(item.get("t1_ms", t0_ms + 1000))))
        except Exception:
            continue
        t0_s = max(0.0, float(t0_ms) / 1000.0)
        t1_s = max(t0_s + 1e-3, float(t1_ms) / 1000.0)
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}
        source_event = str(evidence.get("event_id", "")).strip() or _nearest_event_id_for_span(output, t0_s, t1_s)
        pool_model.append(
            {
                "id": str(item.get("id", "")).strip() or f"model_decision_{i:04d}",
                "t0": t0_s,
                "t1": t1_s,
                "source_event": source_event,
                "action_type": str(item.get("decision_type", item.get("action_type", ""))).strip(),
                "conf": float(item.get("conf", item.get("confidence", 0.5)) or 0.5),
                "source_kind": "decisions_model_v1",
            }
        )
    if pool_model:
        return "decisions_model_v1", pool_model

    pool_heur: list[dict[str, Any]] = []
    for dp in list(output.decision_points or []):
        pool_heur.append(
            {
                "id": str(dp.id),
                "t0": float(dp.t0),
                "t1": float(dp.t1),
                "source_event": str(dp.source_event),
                "action_type": str(dp.action.get("type", "")),
                "conf": float(dp.conf),
                "source_kind": "decision_points",
            }
        )
    return "decision_points", pool_heur


def _sample_segment_vector(
    times: np.ndarray,
    frame_embeds: np.ndarray,
    t0: float,
    t1: float,
    max_frames_per_segment: int,
) -> np.ndarray | None:
    if times.size == 0 or frame_embeds.size == 0:
        return None

    mask = np.where((times >= float(t0)) & (times <= float(t1)))[0]
    if mask.size == 0:
        mid = float((t0 + t1) * 0.5)
        idx = int(np.argmin(np.abs(times - mid)))
        mask = np.asarray([idx], dtype=np.int32)

    if max_frames_per_segment > 0 and mask.size > max_frames_per_segment:
        select = np.linspace(0, mask.size - 1, max_frames_per_segment, dtype=np.int32)
        mask = mask[select]

    vec = np.mean(frame_embeds[mask], axis=0)
    return _l2_normalize(vec)


class IndexBuilder:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.cfg = IndexBuilderConfig(
            sample_fps=float(cfg.get("sample_fps", 4.0)),
            max_frames_per_segment=int(cfg.get("max_frames_per_segment", 24)),
            use_clip=bool(cfg.get("use_clip", False)),
        )

    def build(self, video_path: str | Path, output_json: str | Path | dict[str, Any] | Output) -> tuple[VectorIndex, dict[str, Any]]:
        output = ensure_events_v1(_as_output(output_json))
        reader = VideoReader(video_path)

        sample_fps = float(output.meta.get("sample_fps", self.cfg.sample_fps))
        use_clip = bool(self.cfg.use_clip or str(output.meta.get("embedding_backend", "")) == "clip")
        embedder = Embedder(use_clip=use_clip)

        times: list[float] = []
        frame_embeds: list[np.ndarray] = []
        for t, frame in reader.iter_samples(sample_fps=sample_fps):
            times.append(float(t))
            frame_embeds.append(embedder.embed(frame))

        index = VectorIndex()
        if not frame_embeds:
            stats = {
                "num_event_vecs": 0,
                "num_highlight_vecs": 0,
                "dim": 0,
                "backend": index.backend,
                "embedding_backend": embedder.backend_name,
            }
            return index, stats

        times_arr = np.asarray(times, dtype=np.float32)
        frame_matrix = np.vstack(frame_embeds).astype(np.float32, copy=False)
        tokens = list(output.token_codec.tokens)

        event_v1_count = 0
        event_count = 0
        event_v0_count = 0
        highlight_count = 0
        decision_count = 0
        decision_source_kind, decision_rows = _decision_pool(output)
        for event in output.events_v1:
            vec = _sample_segment_vector(
                times=times_arr,
                frame_embeds=frame_matrix,
                t0=float(event.t0),
                t1=float(event.t1),
                max_frames_per_segment=self.cfg.max_frames_per_segment,
            )
            if vec is None:
                continue
            index.add(
                item_id=event.id,
                vec=vec,
                meta={
                    "kind": "event_v1",
                    "id": event.id,
                    "t0": float(event.t0),
                    "t1": float(event.t1),
                    "source_event": str(event.meta.get("source_event_id", event.id)),
                    "source_event_ids": [str(x) for x in event.source_event_ids],
                    "anchor_types": _anchor_types_for_event_v1(event),
                    "token_types": _token_types_for_range(tokens, float(event.t0), float(event.t1)),
                    "label": str(event.label),
                    "layer": str(event.meta.get("layer", "events_v1")),
                    "embedding_backend": embedder.backend_name,
                },
            )
            event_v1_count += 1

        for event in output.events:
            vec = _sample_segment_vector(
                times=times_arr,
                frame_embeds=frame_matrix,
                t0=float(event.t0),
                t1=float(event.t1),
                max_frames_per_segment=self.cfg.max_frames_per_segment,
            )
            if vec is None:
                continue
            index.add(
                item_id=event.id,
                vec=vec,
                meta={
                    "kind": "event",
                    "id": event.id,
                    "t0": float(event.t0),
                    "t1": float(event.t1),
                    "source_event": event.id,
                    "anchor_types": sorted({anchor.type for anchor in event.anchors}),
                    "token_types": _token_types_for_range(tokens, float(event.t0), float(event.t1)),
                    "embedding_backend": embedder.backend_name,
                },
            )
            event_count += 1

        for event in output.events_v0:
            vec = _sample_segment_vector(
                times=times_arr,
                frame_embeds=frame_matrix,
                t0=float(event.t0),
                t1=float(event.t1),
                max_frames_per_segment=self.cfg.max_frames_per_segment,
            )
            if vec is None:
                continue
            index.add(
                item_id=event.id,
                vec=vec,
                meta={
                    "kind": "event_v0",
                    "id": event.id,
                    "t0": float(event.t0),
                    "t1": float(event.t1),
                    "source_event": event.id,
                    "anchor_types": sorted({anchor.type for anchor in event.anchors}),
                    "token_types": _token_types_for_range(tokens, float(event.t0), float(event.t1)),
                    "label": str(event.meta.get("label", "")),
                    "embedding_backend": embedder.backend_name,
                },
            )
            event_v0_count += 1

        for hl in output.highlights:
            vec = _sample_segment_vector(
                times=times_arr,
                frame_embeds=frame_matrix,
                t0=float(hl.t0),
                t1=float(hl.t1),
                max_frames_per_segment=self.cfg.max_frames_per_segment,
            )
            if vec is None:
                continue
            anchor_types = hl.meta.get("anchor_types")
            if not isinstance(anchor_types, list):
                anchor_types = [hl.anchor_type]
            index.add(
                item_id=hl.id,
                vec=vec,
                meta={
                    "kind": "highlight",
                    "id": hl.id,
                    "t0": float(hl.t0),
                    "t1": float(hl.t1),
                    "source_event": hl.source_event,
                    "anchor_types": [str(x) for x in anchor_types],
                    "token_types": _token_types_for_range(tokens, float(hl.t0), float(hl.t1)),
                    "embedding_backend": embedder.backend_name,
                },
            )
            highlight_count += 1

        for drow in decision_rows:
            vec = _sample_segment_vector(
                times=times_arr,
                frame_embeds=frame_matrix,
                t0=float(drow.get("t0", 0.0)),
                t1=float(drow.get("t1", 0.0)),
                max_frames_per_segment=self.cfg.max_frames_per_segment,
            )
            if vec is None:
                continue
            index.add(
                item_id=str(drow.get("id", "")),
                vec=vec,
                meta={
                    "kind": "decision",
                    "id": str(drow.get("id", "")),
                    "t0": float(drow.get("t0", 0.0)),
                    "t1": float(drow.get("t1", 0.0)),
                    "source_event": str(drow.get("source_event", "")),
                    "action_type": str(drow.get("action_type", "")),
                    "conf": float(drow.get("conf", 0.0) or 0.0),
                    "decision_source_kind": str(drow.get("source_kind", decision_source_kind)),
                    "embedding_backend": embedder.backend_name,
                },
            )
            decision_count += 1

        stats = {
            "num_event_v1_vecs": int(event_v1_count),
            "num_event_vecs": int(event_count),
            "num_event_v0_vecs": int(event_v0_count),
            "num_highlight_vecs": int(highlight_count),
            "num_decision_vecs": int(decision_count),
            "decision_source_kind": str(decision_source_kind),
            "decision_count": int(len(decision_rows)),
            "dim": int(index.dim),
            "backend": index.backend,
            "embedding_backend": embedder.backend_name,
        }
        return index, stats

    def build_and_save(
        self,
        video_path: str | Path,
        output_json: str | Path | dict[str, Any] | Output,
        out_prefix: str | Path,
    ) -> dict[str, Any]:
        index, stats = self.build(video_path=video_path, output_json=output_json)
        npz_path, meta_path = index.save(out_prefix)
        result = dict(stats)
        # Persist a small index-level summary for downstream diagnostics.
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["decision_source_kind"] = str(result.get("decision_source_kind", ""))
                payload["decision_count"] = int(result.get("decision_count", 0))
                payload["num_decision_vecs"] = int(result.get("num_decision_vecs", 0))
                meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        result["index_npz"] = str(npz_path)
        result["index_meta"] = str(meta_path)
        return result
