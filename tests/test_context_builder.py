from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.context.context_builder import build_context
from pov_compiler.schemas import KeyClip, Output, Token, TokenCodec


def test_context_builder_budget_and_priority() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 30.0},
        stats={"original_duration_s": 30.0},
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=10.0,
                t1=14.0,
                source_event="event_0002",
                anchor_type="turn_head",
                anchor_t=12.0,
                conf=0.95,
                meta={"anchor_types": ["turn_head"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(
                    id="tok_000001",
                    t0=10.0,
                    t1=14.0,
                    type="HIGHLIGHT",
                    conf=0.95,
                    source_event="event_0002",
                    source={"highlight_id": "hl_0001"},
                    meta={},
                ),
                Token(
                    id="tok_000002",
                    t0=11.5,
                    t1=12.5,
                    type="ATTENTION_TURN_HEAD",
                    conf=0.9,
                    source_event="event_0002",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_000003",
                    t0=0.0,
                    t1=0.0,
                    type="EVENT_START",
                    conf=0.7,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_000004",
                    t0=5.0,
                    t1=7.0,
                    type="MOTION_MOVING",
                    conf=0.6,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_000005",
                    t0=18.0,
                    t1=19.0,
                    type="SCENE_CHANGE",
                    conf=0.5,
                    source_event="event_0003",
                    source={},
                    meta={},
                ),
            ],
        ),
        events=[
            {
                "id": "event_0001",
                "t0": 0.0,
                "t1": 10.0,
                "scores": {"boundary_conf": 0.6},
                "anchors": [],
            },
            {
                "id": "event_0002",
                "t0": 10.0,
                "t1": 20.0,
                "scores": {"boundary_conf": 0.9},
                "anchors": [],
            },
        ],
    )

    context = build_context(
        output_json=output,
        mode="full",
        budget={"max_events": 8, "max_highlights": 10, "max_tokens": 3},
    )

    assert len(context["tokens"]) <= 3
    token_types = {t["type"] for t in context["tokens"]}
    assert "HIGHLIGHT" in token_types
    assert any(t.startswith("ATTENTION_") for t in token_types)


def test_context_builder_highlight_cap() -> None:
    output = {
        "video_id": "demo",
        "meta": {"duration_s": 20.0},
        "stats": {},
        "events": [],
        "highlights": [
            {
                "id": "hl_0001",
                "t0": 1.0,
                "t1": 2.0,
                "source_event": "event_0001",
                "anchor_type": "turn_head",
                "anchor_t": 1.5,
                "conf": 0.5,
                "meta": {},
            },
            {
                "id": "hl_0002",
                "t0": 3.0,
                "t1": 4.0,
                "source_event": "event_0002",
                "anchor_type": "turn_head",
                "anchor_t": 3.5,
                "conf": 0.9,
                "meta": {},
            },
        ],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
        "decision_points": [],
        "debug": {"signals": {"time": [], "motion_energy": [], "embed_dist": [], "boundary_score": []}},
    }
    context = build_context(
        output_json=output,
        mode="highlights",
        budget={"max_events": 8, "max_highlights": 1, "max_tokens": 50},
    )
    assert len(context["highlights"]) == 1
    assert context["highlights"][0]["id"] == "hl_0002"
