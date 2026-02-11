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

from pov_compiler.memory.index_builder import IndexBuilder


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        return data
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector index for events/highlights")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--json", required=True, help="Pipeline output json path")
    parser.add_argument("--out_prefix", required=True, help="Index output prefix, e.g. data/cache/demo")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Config YAML path")
    parser.add_argument("--sample-fps", type=float, default=None, help="Override sampling fps for indexing")
    parser.add_argument("--max-frames-per-segment", type=int, default=None, help="Override max sampled frames per segment")
    parser.add_argument("--use-clip", action="store_true", help="Use CLIP image embedding if available")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    index_cfg = dict(cfg.get("index", {}))

    if args.sample_fps is not None:
        index_cfg["sample_fps"] = float(args.sample_fps)
    if args.max_frames_per_segment is not None:
        index_cfg["max_frames_per_segment"] = int(args.max_frames_per_segment)
    if args.use_clip:
        index_cfg["use_clip"] = True

    builder = IndexBuilder(config=index_cfg)
    result = builder.build_and_save(
        video_path=Path(args.video),
        output_json=Path(args.json),
        out_prefix=Path(args.out_prefix),
    )

    print(f"num_event_vecs={result.get('num_event_vecs', 0)}")
    print(f"num_highlight_vecs={result.get('num_highlight_vecs', 0)}")
    print(f"dim={result.get('dim', 0)}")
    print(f"backend={result.get('backend', 'numpy')}")
    print(f"embedding_backend={result.get('embedding_backend', 'unknown')}")
    print(f"index_npz={result.get('index_npz')}")
    print(f"index_meta={result.get('index_meta')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
