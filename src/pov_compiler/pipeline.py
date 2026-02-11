from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.features.embeddings import Embedder, cosine_distance
from pov_compiler.features.motion import compute_motion_energy, to_gray
from pov_compiler.io.video_reader import VideoReader
from pov_compiler.l1_events.anchor_miner import mine_event_anchors
from pov_compiler.l1_events.event_segmenter import fuse_boundary_score, segment_events
from pov_compiler.l2_tokens.token_codec import TokenCodecCompiler
from pov_compiler.l3_decisions.decision_compiler import DecisionCompiler
from pov_compiler.memory.decision_sampling import build_highlights
from pov_compiler.schemas import Output


DEFAULT_CONFIG: dict[str, Any] = {
    "sample_fps": 4.0,
    "segmenter": {
        "thresh": 0.65,
        "min_event_s": 3.0,
    },
    "anchors": {
        "stop_look_quantile": 0.2,
        "stop_look_min_duration_s": 0.5,
        "turn_head_top_k": 2,
        "max_stop_look_per_event": 3,
        "min_gap_s": 2.0,
        "stop_look_min_conf": 0.6,
    },
    "highlights": {
        "enabled": True,
        "pre_s": 2.0,
        "post_s": 2.0,
        "max_total_s": 60.0,
        "merge_gap_s": 0.2,
        "priority_map": {"interaction": 3, "turn_head": 2, "stop_look": 1},
    },
    "tokens": {
        "enabled": True,
        "attention_pre_s": 0.5,
        "attention_post_s": 0.5,
        "motion_min_run_s": 0.8,
        "motion_max_runs_per_event": 4,
        "scene_change_top_k": 3,
        "merge_gap_s": 0.2,
    },
    "decisions": {
        "enabled": True,
        "pre_s": 2.0,
        "post_s": 2.0,
        "merge_iou": 0.7,
        "min_gap_s": 0.0,
        "boundary_thresh": 0.65,
    },
    "context_default": {
        "mode": "highlights",
        "max_events": 8,
        "max_highlights": 10,
        "max_tokens": 200,
        "max_decisions": 12,
    },
    "features": {
        "use_clip": False,
    },
}


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class OfflinePipeline:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = deepcopy(DEFAULT_CONFIG)
        if config:
            _deep_merge(self.config, config)

    def run(self, video_path: str | Path) -> Output:
        path = Path(video_path)
        reader = VideoReader(path)

        sample_fps = float(self.config["sample_fps"])
        embedder = Embedder(use_clip=bool(self.config["features"].get("use_clip", False)))

        times: list[float] = []
        embed_dist: list[float] = []
        motion_energy: list[float] = []

        prev_embed: np.ndarray | None = None
        prev_gray: np.ndarray | None = None

        for t, frame in reader.iter_samples(sample_fps=sample_fps):
            times.append(float(t))

            emb = embedder.embed(frame)
            if prev_embed is None:
                embed_dist.append(0.0)
            else:
                embed_dist.append(float(cosine_distance(prev_embed, emb)))
            prev_embed = emb

            gray = to_gray(frame)
            if prev_gray is None:
                motion_energy.append(0.0)
            else:
                motion_energy.append(float(compute_motion_energy(prev_gray, gray)))
            prev_gray = gray

        duration_s = float(reader.duration_s)
        if duration_s <= 0 and times:
            duration_s = float(times[-1])
        elif times:
            duration_s = max(duration_s, float(times[-1]))

        motion_change, _motion_change_norm, boundary_score = fuse_boundary_score(
            embed_dist=embed_dist, motion_energy=motion_energy
        )

        seg_cfg = self.config["segmenter"]
        events = segment_events(
            times=times,
            boundary_score=boundary_score,
            duration_s=duration_s,
            thresh=float(seg_cfg.get("thresh", 0.65)),
            min_event_s=float(seg_cfg.get("min_event_s", 3.0)),
        )

        anchor_cfg = self.config["anchors"]
        for event in events:
            event.anchors = mine_event_anchors(
                event=event,
                times=times,
                motion_energy=motion_energy,
                embed_dist=embed_dist,
                motion_change=motion_change,
                stop_look_quantile=float(anchor_cfg.get("stop_look_quantile", 0.2)),
                stop_look_min_duration_s=float(anchor_cfg.get("stop_look_min_duration_s", 0.5)),
                turn_head_top_k=int(anchor_cfg.get("turn_head_top_k", 2)),
                max_stop_look_per_event=int(anchor_cfg.get("max_stop_look_per_event", 3)),
                min_gap_s=float(anchor_cfg.get("min_gap_s", 2.0)),
                stop_look_min_conf=float(anchor_cfg.get("stop_look_min_conf", 0.6)),
            )

        highlights: list = []
        stats: dict[str, float | int] = {
            "original_duration_s": float(duration_s),
            "kept_duration_s": 0.0,
            "compression_ratio": 0.0,
            "num_highlights": 0,
        }
        hl_cfg = self.config.get("highlights", {})
        if bool(hl_cfg.get("enabled", True)):
            highlights, stats = build_highlights(
                events=events,
                duration_s=float(duration_s),
                pre_s=float(hl_cfg.get("pre_s", 2.0)),
                post_s=float(hl_cfg.get("post_s", 2.0)),
                max_total_s=float(hl_cfg.get("max_total_s", 60.0)),
                merge_gap_s=float(hl_cfg.get("merge_gap_s", 0.2)),
                priority_map=dict(
                    hl_cfg.get(
                        "priority_map",
                        {"interaction": 3, "turn_head": 2, "stop_look": 1},
                    )
                ),
            )

        output = Output(
            video_id=path.stem,
            meta={
                "fps": float(reader.fps),
                "duration_s": float(duration_s),
                "sample_fps": float(sample_fps),
                "embedding_backend": embedder.backend_name,
            },
            events=events,
            highlights=highlights,
            stats=stats,
            decision_points=[],
            debug={
                "signals": {
                    "time": [float(x) for x in times],
                    "motion_energy": [float(x) for x in motion_energy],
                    "embed_dist": [float(x) for x in embed_dist],
                    "boundary_score": [float(x) for x in boundary_score.tolist()],
                }
            },
        )

        token_cfg = self.config.get("tokens", {})
        if bool(token_cfg.get("enabled", True)):
            compiler = TokenCodecCompiler(config=dict(token_cfg))
            output.token_codec = compiler.compile(output)

        decision_cfg = self.config.get("decisions", {})
        if bool(decision_cfg.get("enabled", True)):
            decision_compiler = DecisionCompiler(config=dict(decision_cfg))
            output.decision_points = decision_compiler.compile(output)
        return output


def output_to_dict(output: Output) -> dict[str, Any]:
    return _model_dump(output)
