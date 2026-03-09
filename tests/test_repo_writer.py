from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.repository.writer import build_repo_chunks
from pov_compiler.schemas import Anchor, DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="repo_demo",
        meta={"duration_s": 36.0},
        events=[
            Event(
                id="event_0001",
                t0=0.0,
                t1=12.0,
                scores={"boundary_conf": 0.7},
                anchors=[Anchor(type="turn_head", t=4.0, conf=0.8)],
                meta={"label": "navigation"},
            ),
            Event(
                id="event_0002",
                t0=12.0,
                t1=24.0,
                scores={"boundary_conf": 0.75},
                anchors=[Anchor(type="stop_look", t=18.0, conf=0.7)],
                meta={"label": "interaction-heavy"},
            ),
            Event(
                id="event_0003",
                t0=24.0,
                t1=36.0,
                scores={"boundary_conf": 0.68},
                anchors=[Anchor(type="turn_head", t=28.0, conf=0.8)],
                meta={"label": "idle"},
            ),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=17.5,
                t1=18.5,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=18.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=10.0, t1=10.8, type="SCENE_CHANGE", conf=0.72, source_event="event_0001"),
                Token(id="tok_0002", t0=17.5, t1=18.5, type="ATTENTION_STOP_LOOK", conf=0.8, source_event="event_0002"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=18.0,
                t0=17.5,
                t1=18.5,
                source_event="event_0002",
                source_highlight="hl_0001",
                action={"type": "ATTENTION_STOP_LOOK"},
                conf=0.82,
            )
        ],
    )


def test_repo_writer_multiscale_chunks() -> None:
    output = ensure_events_v1(_make_output())
    chunks = build_repo_chunks(
        output,
        cfg={
            "window_s": 15.0,
            "min_segment_s": 5.0,
            "scales": {"event": True, "window": True, "segment": True},
        },
    )
    assert chunks
    scales = {c.scale for c in chunks}
    assert {"event", "window", "segment"}.issubset(scales)
    for chunk in chunks:
        assert chunk.id
        assert chunk.text
        assert float(chunk.t1) >= float(chunk.t0)
        assert 0.0 <= float(chunk.importance) <= 1.0
