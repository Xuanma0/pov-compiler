from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import build_place_segments_v0
from pov_compiler.schemas import Output, Token, TokenCodec


def test_build_place_segments_v0_from_scene_and_boundary() -> None:
    output = Output(
        video_id="place_demo",
        meta={"duration_s": 30.0},
        token_codec=TokenCodec(
            tokens=[
                Token(id="tok_sc_1", t0=9.8, t1=10.2, type="SCENE_CHANGE", conf=0.9, source_event="event_0001"),
                Token(id="tok_sc_2", t0=19.8, t1=20.2, type="SCENE_CHANGE", conf=0.9, source_event="event_0002"),
            ]
        ),
        debug={
            "signals": {
                "time": [0, 5, 10, 15, 20, 25, 30],
                "boundary_score": [0.1, 0.2, 0.9, 0.1, 0.85, 0.1, 0.1],
            }
        },
    )

    segments = build_place_segments_v0(output, boundary_thresh=0.65, min_segment_s=2.0)
    assert segments
    assert segments[0]["id"].startswith("place_")
    assert float(segments[0]["t0"]) == 0.0
    assert float(segments[-1]["t1"]) == 30.0
    assert len(segments) >= 3
    assert all(float(item["t1"]) > float(item["t0"]) for item in segments)
