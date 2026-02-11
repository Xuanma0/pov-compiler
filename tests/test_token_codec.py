from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.l2_tokens.token_codec import TokenCodecCompiler
from pov_compiler.schemas import Anchor, Event, KeyClip, Output


def test_token_codec_compile_and_merge() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 10.0, "fps": 30.0},
        stats={},
        events=[
            Event(
                id="event_0001",
                t0=0.0,
                t1=5.0,
                scores={"boundary_conf": 0.8},
                anchors=[
                    Anchor(type="turn_head", t=1.5, conf=0.9, meta={}),
                    Anchor(type="stop_look", t=3.0, conf=0.8, meta={"duration_s": 1.0}),
                ],
            ),
            Event(
                id="event_0002",
                t0=5.0,
                t1=10.0,
                scores={"boundary_conf": 0.7},
                anchors=[Anchor(type="turn_head", t=7.0, conf=0.85, meta={})],
            ),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=4.2,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=3.0,
                conf=0.9,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=4.1,
                t1=6.0,
                source_event="event_0002",
                anchor_type="turn_head",
                anchor_t=5.0,
                conf=0.7,
                meta={"anchor_types": ["turn_head"]},
            ),
        ],
        debug={
            "signals": {
                "time": [float(i) for i in range(10)],
                "motion_energy": [0.1, 0.2, 0.9, 0.95, 0.2, 0.1, 0.85, 0.8, 0.2, 0.1],
                "embed_dist": [0.0, 0.1, 0.4, 0.2, 0.1, 0.6, 0.2, 0.1, 0.5, 0.1],
                "boundary_score": [0.0, 0.2, 0.7, 0.2, 0.1, 0.9, 0.2, 0.1, 0.6, 0.1],
            }
        },
    )

    compiler = TokenCodecCompiler(config={"scene_change_top_k": 2, "motion_max_runs_per_event": 4})
    codec = compiler.compile(output)
    tokens = codec.tokens

    assert codec.version == "0.2"
    assert len(tokens) > 0
    assert all(tokens[i].t0 <= tokens[i + 1].t0 for i in range(len(tokens) - 1))
    assert [t.id for t in tokens] == [f"tok_{i:06d}" for i in range(1, len(tokens) + 1)]

    token_types = {t.type for t in tokens}
    assert "EVENT_START" in token_types
    assert "EVENT_END" in token_types
    assert "ATTENTION_TURN_HEAD" in token_types
    assert "HIGHLIGHT" in token_types

    # Two overlapping highlights should be merged into one HIGHLIGHT token.
    highlight_tokens = [t for t in tokens if t.type == "HIGHLIGHT"]
    assert len(highlight_tokens) == 1
