from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.features.embeddings import Embedder
from pov_compiler.io.video_reader import VideoReader
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
        output = _as_output(output_json)
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

        event_count = 0
        highlight_count = 0
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

        stats = {
            "num_event_vecs": int(event_count),
            "num_highlight_vecs": int(highlight_count),
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
        result["index_npz"] = str(npz_path)
        result["index_meta"] = str(meta_path)
        return result
