from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.pipeline import DEFAULT_CONFIG, OfflinePipeline, output_to_dict
from pov_compiler.context.context_builder import build_context


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        print(f"[warn] pyyaml not installed, ignoring config file: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run POV offline pipeline v0.1")
    parser.add_argument("--video", required=True, help="Path to input mp4 video")
    parser.add_argument("--out", required=True, help="Path to output JSON")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="YAML config path")
    parser.add_argument("--sample-fps", type=float, default=None, help="Override sample fps")
    parser.add_argument("--thresh", type=float, default=None, help="Boundary threshold")
    parser.add_argument("--min-event-s", type=float, default=None, help="Minimum event duration in seconds")
    parser.add_argument("--use-clip", action="store_true", help="Enable CLIP embedding if torch+open_clip exist")
    parser.add_argument("--run-perception", action="store_true", help="Enable Perception v0 (YOLO/MediaPipe or fallback)")
    parser.add_argument(
        "--perception-backend",
        choices=["real", "stub"],
        default=None,
        help="Perception backend (real|stub)",
    )
    parser.set_defaults(perception_fallback_stub=None)
    parser.add_argument(
        "--perception-fallback-stub",
        dest="perception_fallback_stub",
        action="store_true",
        help="Allow fallback from real backend to stub backend",
    )
    parser.add_argument(
        "--no-perception-fallback-stub",
        dest="perception_fallback_stub",
        action="store_false",
        help="Disable fallback to stub backend",
    )
    parser.add_argument(
        "--perception-strict",
        action="store_true",
        help="Strict perception mode: no fallback, fail on missing deps/frame errors",
    )
    parser.add_argument("--perception-fps", type=float, default=None, help="Perception sampling fps")
    parser.add_argument("--perception-max-frames", type=int, default=None, help="Perception frame cap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    file_cfg = load_yaml_config(Path(args.config))
    _deep_merge(config, file_cfg)

    if args.sample_fps is not None:
        config["sample_fps"] = float(args.sample_fps)
    if args.thresh is not None:
        config.setdefault("segmenter", {})["thresh"] = float(args.thresh)
    if args.min_event_s is not None:
        config.setdefault("segmenter", {})["min_event_s"] = float(args.min_event_s)
    if args.use_clip:
        config.setdefault("features", {})["use_clip"] = True
    if args.run_perception:
        config.setdefault("perception", {})["enabled"] = True
    if args.perception_backend is not None:
        config.setdefault("perception", {})["backend"] = str(args.perception_backend)
    if args.perception_fallback_stub is not None:
        config.setdefault("perception", {})["fallback_to_stub"] = bool(args.perception_fallback_stub)
    if args.perception_strict:
        config.setdefault("perception", {})["strict"] = True
        config.setdefault("perception", {})["fallback_to_stub"] = False
    if args.perception_fps is not None:
        config.setdefault("perception", {})["sample_fps"] = float(args.perception_fps)
    if args.perception_max_frames is not None:
        config.setdefault("perception", {})["max_frames"] = int(args.perception_max_frames)
    if config.get("perception", {}).get("enabled", False):
        config.setdefault("perception", {})
        if not config["perception"].get("cache_dir"):
            out_path = Path(args.out)
            config["perception"]["cache_dir"] = str(out_path.parent / "perception_cache")

    pipeline = OfflinePipeline(config=config)
    output = pipeline.run(args.video)
    output_dict = output_to_dict(output)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    print(f"video_id={output.video_id}")
    print(f"events={len(output.events)}")
    for event in output.events:
        dur = event.t1 - event.t0
        print(f"{event.id}: duration={dur:.2f}s anchors={len(event.anchors)}")
    print(f"events_v0={len(output.events_v0)}")
    for event in output.events_v0[:5]:
        dur = event.t1 - event.t0
        label = str(event.meta.get("label", ""))
        print(f"{event.id}: duration={dur:.2f}s label={label}")
    print(f"events_v1={len(output.events_v1)}")
    for event in output.events_v1[:5]:
        dur = event.t1 - event.t0
        print(f"{event.id}: duration={dur:.2f}s label={event.label} evidence={len(event.evidence)}")
    kept = float(output.stats.get("kept_duration_s", 0.0)) if isinstance(output.stats, dict) else 0.0
    ratio = float(output.stats.get("compression_ratio", 0.0)) if isinstance(output.stats, dict) else 0.0
    tokens_total = len(output.token_codec.tokens) if output.token_codec else 0
    decisions_total = len(output.decision_points)
    context_default_cfg = dict(config.get("context_default", {}))
    context_mode = str(context_default_cfg.pop("mode", "highlights"))
    context_json = build_context(output_dict, mode=context_mode, budget=context_default_cfg)
    tokens_context_default = len(context_json.get("tokens", []))
    decisions_context_json = build_context(output_dict, mode="decisions", budget=context_default_cfg)
    decisions_context_default = len(decisions_context_json.get("decision_points", []))
    print(f"kept_duration_s={kept:.2f}")
    print(f"compression_ratio={ratio:.4f}")
    print(f"tokens_total={tokens_total}")
    print(f"tokens_context_default={tokens_context_default}")
    print(f"decision_points_total={decisions_total}")
    print(f"decisions_context_default={decisions_context_default}")
    if isinstance(output.perception, dict) and output.perception:
        p_meta = output.perception.get("meta", {}) if isinstance(output.perception.get("meta", {}), dict) else {}
        p_sum = output.perception.get("summary", {}) if isinstance(output.perception.get("summary", {}), dict) else {}
        print(f"perception_backend={p_meta.get('backend', '')}")
        print(f"perception_frames={p_meta.get('processed_frames', 0)}")
        print(f"contact_events_count={p_sum.get('contact_events_count', 0)}")
        print(f"perception_fallback_used={str(bool(p_sum.get('fallback_used', False))).lower()}")
        if p_sum.get("fallback_reason"):
            print(f"perception_fallback_reason={p_sum.get('fallback_reason')}")
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
