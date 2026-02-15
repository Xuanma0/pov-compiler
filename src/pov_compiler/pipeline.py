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
from pov_compiler.l1_events.event_segmentation_v0 import segment_events_v0
from pov_compiler.l2_tokens.token_codec import TokenCodecCompiler
from pov_compiler.l3_decisions.decision_compiler import DecisionCompiler
from pov_compiler.ir.events_v1 import convert_output_to_events_v1
from pov_compiler.memory.decision_sampling import build_highlights
from pov_compiler.perception.object_memory_v0 import build_object_memory_v0
from pov_compiler.perception.runner import run_perception
from pov_compiler.repository import build_repo_chunks, deduplicate_chunks, policy_cfg_hash
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
        "backend": "heuristic",
        "pre_s": 2.0,
        "post_s": 2.0,
        "merge_iou": 0.7,
        "min_gap_s": 0.0,
        "boundary_thresh": 0.65,
        "model_client": {
            "provider": "fake",
            "model": "fake-decision-v1",
            "base_url": None,
            "api_key_env": "",
            "timeout_s": 60,
            "max_tokens": 800,
            "temperature": 0.2,
            "extra_headers": {},
            "extra": {},
        },
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
    "perception": {
        "enabled": False,
        "backend": "stub",
        "fallback_to_stub": True,
        "strict": False,
        "sample_fps": 10.0,
        "max_frames": 300,
        "cache_dir": "data/cache/perception",
        "contact_min_score": 0.25,
        "objects_topk": 10,
        "model_candidates": ["yolo26n.pt", "yolov8n.pt"],
        "hand_task_model_path": "assets/mediapipe/hand_landmarker.task",
    },
    "event_v0": {
        "enabled": True,
        "thresh": 0.45,
        "min_event_s": 3.0,
        "visual_weight": 0.7,
        "contact_weight": 0.3,
        "append_to_events": False,
    },
    "events_v1": {
        "enabled": True,
    },
    "object_memory": {
        "enabled": True,
        "contact_threshold": 0.6,
    },
    "repo": {
        "enable": False,
        "write_policy": {
            "name": "fixed_interval",
            "chunk_step_s": 8.0,
            "keep_levels": ["decision", "place"],
        },
        "read_policy": {
            "name": "budgeted_topk",
            "max_chunks": 16,
            "max_tokens": 200,
            "max_seconds": None,
        },
        "window_s": 30.0,
        "min_segment_s": 5.0,
        "scales": {
            "event": True,
            "decision": True,
            "place": True,
            "window": True,
            "segment": True,
        },
        "dedup": {
            "iou_thresh": 0.6,
            "sim_thresh": 0.9,
            "cross_scale": True,
            "keep_best_importance": True,
        },
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

        perception_cfg = dict(self.config.get("perception", {}))
        perception_payload: dict[str, Any] = {}
        visual_change_v0 = [float(x) for x in embed_dist]
        contact_score_v0 = [0.0 for _ in range(len(visual_change_v0))]
        event_v0_times = [float(x) for x in times]
        if bool(perception_cfg.get("enabled", False)):
            backend_name = str(perception_cfg.get("backend", "real")).strip().lower()
            backend_kwargs = {
                "model_candidates": list(perception_cfg.get("model_candidates", ["yolo26n.pt", "yolov8n.pt"])),
                "hand_task_model_path": perception_cfg.get("hand_task_model_path"),
            }
            try:
                perception_payload = run_perception(
                    video_path=path,
                    sample_fps=float(perception_cfg.get("sample_fps", 10.0)),
                    max_frames=int(perception_cfg.get("max_frames", 300)),
                    backend_name=backend_name,
                    backend_kwargs=backend_kwargs,
                    fallback_to_stub=bool(perception_cfg.get("fallback_to_stub", True)),
                    strict=bool(perception_cfg.get("strict", False)),
                    cache_dir=perception_cfg.get("cache_dir"),
                    contact_min_score=float(perception_cfg.get("contact_min_score", 0.25)),
                    objects_topk=int(perception_cfg.get("objects_topk", 10)),
                )
            except Exception:
                if bool(perception_cfg.get("fallback_to_stub", True)):
                    perception_payload = run_perception(
                        video_path=path,
                        sample_fps=float(perception_cfg.get("sample_fps", 10.0)),
                        max_frames=int(perception_cfg.get("max_frames", 300)),
                        backend_name="stub",
                        backend_kwargs={},
                        fallback_to_stub=False,
                        strict=False,
                        cache_dir=perception_cfg.get("cache_dir"),
                        contact_min_score=float(perception_cfg.get("contact_min_score", 0.25)),
                        objects_topk=int(perception_cfg.get("objects_topk", 10)),
                    )
                else:
                    raise
            signals = perception_payload.get("signals", {}) if isinstance(perception_payload, dict) else {}
            if isinstance(signals, dict):
                t_sig = signals.get("time", [])
                v_sig = signals.get("visual_change", [])
                c_sig = signals.get("contact_score", [])
                if isinstance(t_sig, list) and isinstance(v_sig, list) and isinstance(c_sig, list):
                    m = min(len(t_sig), len(v_sig), len(c_sig))
                    if m > 0:
                        event_v0_times = [float(x) for x in t_sig[:m]]
                        visual_change_v0 = [float(x) for x in v_sig[:m]]
                        contact_score_v0 = [float(x) for x in c_sig[:m]]

        event_v0_cfg = dict(self.config.get("event_v0", {}))
        events_v0: list = []
        boundary_score_v0: list[float] = []
        if bool(event_v0_cfg.get("enabled", True)):
            events_v0, score_v0 = segment_events_v0(
                times=event_v0_times,
                visual_change=visual_change_v0,
                contact_score=contact_score_v0,
                duration_s=float(duration_s),
                thresh=float(event_v0_cfg.get("thresh", 0.45)),
                min_event_s=float(event_v0_cfg.get("min_event_s", 3.0)),
                visual_weight=float(event_v0_cfg.get("visual_weight", 0.7)),
                contact_weight=float(event_v0_cfg.get("contact_weight", 0.3)),
            )
            boundary_score_v0 = [float(x) for x in score_v0.tolist()]
            if bool(event_v0_cfg.get("append_to_events", False)):
                existing_ids = {str(e.id) for e in events}
                for ev in events_v0:
                    if str(ev.id) not in existing_ids:
                        events.append(ev)

        output = Output(
            video_id=path.stem,
            meta={
                "fps": float(reader.fps),
                "duration_s": float(duration_s),
                "sample_fps": float(sample_fps),
                "embedding_backend": embedder.backend_name,
            },
            events=events,
            events_v0=events_v0,
            highlights=highlights,
            stats=stats,
            perception=perception_payload if isinstance(perception_payload, dict) else {},
            decision_points=[],
            debug={
                "signals": {
                    "time": [float(x) for x in times],
                    "motion_energy": [float(x) for x in motion_energy],
                    "embed_dist": [float(x) for x in embed_dist],
                    "boundary_score": [float(x) for x in boundary_score.tolist()],
                    "event_v0_time": [float(x) for x in event_v0_times],
                    "visual_change_v0": [float(x) for x in visual_change_v0],
                    "contact_score_v0": [float(x) for x in contact_score_v0],
                    "boundary_score_v0": [float(x) for x in boundary_score_v0],
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
            backend = str(decision_cfg.get("backend", "heuristic")).strip().lower()
            if backend == "model":
                from pov_compiler.l3_decisions.model_compiler import compile_decisions_with_model
                from pov_compiler.models import ModelClientConfig, make_client

                model_cfg_raw = decision_cfg.get("model_client", {}) if isinstance(decision_cfg, dict) else {}
                if not isinstance(model_cfg_raw, dict):
                    model_cfg_raw = {}
                model_cfg = ModelClientConfig(
                    provider=str(model_cfg_raw.get("provider", "fake")),
                    model=str(model_cfg_raw.get("model", "fake-decision-v1")),
                    base_url=str(model_cfg_raw.get("base_url")) if model_cfg_raw.get("base_url") not in (None, "") else None,
                    api_key_env=str(model_cfg_raw.get("api_key_env", "")),
                    timeout_s=int(model_cfg_raw.get("timeout_s", 60)),
                    max_tokens=int(model_cfg_raw.get("max_tokens", 800)),
                    temperature=float(model_cfg_raw.get("temperature", 0.2)),
                    extra_headers=dict(model_cfg_raw.get("extra_headers", {}))
                    if isinstance(model_cfg_raw.get("extra_headers", {}), dict)
                    else {},
                    extra=dict(model_cfg_raw.get("extra", {})) if isinstance(model_cfg_raw.get("extra", {}), dict) else {},
                )
                model_client = make_client(model_cfg)
                output.decisions_model_v1 = compile_decisions_with_model(output=output, client=model_client, cfg=model_cfg)
            else:
                output.decisions_model_v1 = []

        events_v1_cfg = self.config.get("events_v1", {})
        if bool(events_v1_cfg.get("enabled", True)):
            output.events_v1 = convert_output_to_events_v1(output)

        object_memory_cfg = dict(self.config.get("object_memory", {}))
        if bool(object_memory_cfg.get("enabled", True)):
            output.object_memory_v0 = build_object_memory_v0(
                perception=output.perception,
                events_v1=list(output.events_v1),
                contact_threshold=float(object_memory_cfg.get("contact_threshold", 0.6)),
            )
        else:
            output.object_memory_v0 = []

        repo_cfg = dict(self.config.get("repo", {}))
        if bool(repo_cfg.get("enable", False)):
            raw_chunks = build_repo_chunks(output, cfg=repo_cfg)
            dedup_chunks = deduplicate_chunks(raw_chunks, cfg=dict(repo_cfg.get("dedup", {})))
            by_level: dict[str, int] = {}
            for chunk in dedup_chunks:
                key = str(chunk.level or chunk.scale)
                by_level[key] = by_level.get(key, 0) + 1
            output.repository = {
                "chunks": [_model_dump(chunk) for chunk in dedup_chunks],
                "summary": {
                    "chunks_before_dedup": len(raw_chunks),
                    "chunks_after_dedup": len(dedup_chunks),
                    "by_scale": dict(sorted(by_level.items())),
                },
                "cfg": repo_cfg,
                "cfg_hash": policy_cfg_hash(repo_cfg),
            }
        return output


def output_to_dict(output: Output) -> dict[str, Any]:
    return _model_dump(output)
