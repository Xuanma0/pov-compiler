from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, KeyClip, Output, Token, TokenCodec
from pov_compiler.streaming.runner import StreamingConfig, run_streaming


def _make_output() -> Output:
    return Output(
        video_id="demo_stream",
        meta={"duration_s": 18.0},
        events=[
            Event(
                id="event_0001",
                t0=0.0,
                t1=9.0,
                scores={"boundary_conf": 0.6},
                anchors=[Anchor(type="turn_head", t=2.0, conf=0.8)],
            ),
            Event(
                id="event_0002",
                t0=9.0,
                t1=18.0,
                scores={"boundary_conf": 0.7},
                anchors=[Anchor(type="stop_look", t=12.0, conf=0.75)],
            ),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=1.5,
                t1=3.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=2.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=11.0,
                t1=13.0,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=12.0,
                conf=0.78,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=1.5, t1=3.0, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=10.0, t1=11.0, type="SCENE_CHANGE", conf=0.7, source_event="event_0002"),
            ],
        ),
    )


def test_streaming_runner_produces_latency_and_progressive_rows() -> None:
    payload = run_streaming(
        _make_output(),
        config=StreamingConfig(
            step_s=6.0,
            top_k=4,
            queries=["anchor=turn_head top_k=4", "token=SCENE_CHANGE top_k=4"],
        ),
    )
    summary = payload["summary"]
    assert summary["steps"] >= 1
    assert summary["queries_total"] >= 2
    assert summary["latency_p50_ms"] >= 0.0
    assert summary["latency_p95_ms"] >= 0.0

    progressive = payload["progressive_rows"]
    assert progressive
    counts = [int(row["events_v1_count"]) for row in progressive]
    assert counts == sorted(counts)

