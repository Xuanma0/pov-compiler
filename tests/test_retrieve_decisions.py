from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Alternative, DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def test_retrieve_by_decision_type() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 20.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, scores={}, anchors=[]),
            Event(id="event_0002", t0=10.0, t1=20.0, scores={}, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=4.0,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=3.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=11.0,
                t1=13.0,
                source_event="event_0002",
                anchor_type="turn_head",
                anchor_t=12.0,
                conf=0.9,
                meta={"anchor_types": ["turn_head"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(
                    id="tok_000001",
                    t0=11.0,
                    t1=13.0,
                    type="HIGHLIGHT",
                    conf=0.9,
                    source_event="event_0002",
                    source={},
                    meta={},
                )
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_000001",
                t=3.0,
                t0=2.0,
                t1=4.0,
                source_event="event_0001",
                source_highlight="hl_0001",
                trigger={"anchor_type": "stop_look", "anchor_types": ["stop_look"], "token_ids": []},
                state={"evidence": {"token_ids": []}},
                action={"type": "ATTENTION_STOP_LOOK", "conf": 0.8},
                constraints=[],
                outcome={"type": "STILL_CONTINUE", "conf": 0.7},
                alternatives=[
                    Alternative(action_type="CONTINUE_MOVING", rationale="", expected_outcome="", conf=0.6),
                    Alternative(action_type="SHORTER_PAUSE", rationale="", expected_outcome="", conf=0.6),
                ],
                conf=0.8,
                meta={},
            ),
            DecisionPoint(
                id="dp_000002",
                t=12.0,
                t0=11.0,
                t1=13.0,
                source_event="event_0002",
                source_highlight="hl_0002",
                trigger={"anchor_type": "turn_head", "anchor_types": ["turn_head"], "token_ids": ["tok_000001"]},
                state={"evidence": {"token_ids": ["tok_000001"]}},
                action={"type": "ATTENTION_TURN_HEAD", "conf": 0.9},
                constraints=[],
                outcome={"type": "SCENE_CHANGED", "conf": 0.85},
                alternatives=[
                    Alternative(action_type="LOOK_FORWARD_ONLY", rationale="", expected_outcome="", conf=0.6),
                    Alternative(action_type="TURN_HEAD_OPPOSITE", rationale="", expected_outcome="", conf=0.6),
                ],
                conf=0.9,
                meta={},
            ),
        ],
    )

    retriever = Retriever(output_json=output)
    result = retriever.retrieve("decision=ATTENTION_TURN_HEAD top_k=6")
    assert result["selected_decisions"] == ["dp_000002"]
    assert result["selected_highlights"] == ["hl_0002"]
    assert "event_0002" in result["selected_events"]
