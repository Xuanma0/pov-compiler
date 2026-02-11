from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import load_hard_pseudo_nlq
from pov_compiler.schemas import DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = min(a), max(a)
    b0, b1 = min(b), max(b)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, (a1 - a0) + (b1 - b0) - inter)
    return inter / union


def test_hard_pseudo_nlq_generation_has_no_label_leak_and_has_valid_distractors() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 220.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=110.0, scores={}, anchors=[]),
            Event(id="event_0002", t0=110.0, t1=220.0, scores={}, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=10.0,
                t1=12.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=11.0,
                conf=0.9,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=40.0,
                t1=42.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=41.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0003",
                t0=70.0,
                t1=73.0,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=71.0,
                conf=0.85,
                meta={"anchor_types": ["stop_look"]},
            ),
            KeyClip(
                id="hl_0004",
                t0=100.0,
                t1=103.0,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=101.0,
                conf=0.82,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=20.0, t1=21.0, type="SCENE_CHANGE", conf=0.9, source_event="event_0001"),
                Token(id="tok_0002", t0=150.0, t1=151.0, type="SCENE_CHANGE", conf=0.8, source_event="event_0002"),
                Token(id="tok_0003", t0=30.0, t1=34.0, type="MOTION_MOVING", conf=0.8, source_event="event_0001"),
                Token(id="tok_0004", t0=170.0, t1=174.0, type="MOTION_MOVING", conf=0.7, source_event="event_0002"),
                Token(id="tok_0005", t0=80.0, t1=84.0, type="MOTION_STILL", conf=0.85, source_event="event_0001"),
                Token(id="tok_0006", t0=190.0, t1=194.0, type="MOTION_STILL", conf=0.75, source_event="event_0002"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=16.0,
                t0=15.0,
                t1=17.0,
                source_event="event_0001",
                source_highlight="hl_0001",
                trigger={},
                state={},
                action={"type": "ATTENTION_TURN_HEAD"},
                constraints=[],
                outcome={},
                alternatives=[],
                conf=0.8,
            ),
            DecisionPoint(
                id="dp_0002",
                t=131.0,
                t0=130.0,
                t1=132.0,
                source_event="event_0002",
                source_highlight=None,
                trigger={},
                state={},
                action={"type": "ATTENTION_TURN_HEAD"},
                constraints=[],
                outcome={},
                alternatives=[],
                conf=0.7,
            ),
            DecisionPoint(
                id="dp_0003",
                t=61.0,
                t0=60.0,
                t1=62.0,
                source_event="event_0001",
                source_highlight="hl_0003",
                trigger={},
                state={},
                action={"type": "ATTENTION_STOP_LOOK"},
                constraints=[],
                outcome={},
                alternatives=[],
                conf=0.78,
            ),
            DecisionPoint(
                id="dp_0004",
                t=181.0,
                t0=180.0,
                t1=182.0,
                source_event="event_0002",
                source_highlight=None,
                trigger={},
                state={},
                action={"type": "ATTENTION_STOP_LOOK"},
                constraints=[],
                outcome={},
                alternatives=[],
                conf=0.74,
            ),
        ],
    )

    samples = load_hard_pseudo_nlq(
        output,
        seed=0,
        n_highlight=4,
        n_token=6,
        n_decision=4,
        top_k=6,
    )
    assert samples

    forbidden = ["SCENE_CHANGE", "ATTENTION_", "MOTION_", "TURN_HEAD", "STOP_LOOK", "DECISION="]
    anchor_triggers = ["张望", "回头", "扫视", "looking around", "停下来", "停住", "pause"]
    for sample in samples:
        query_upper = sample.query.upper()
        for token in forbidden:
            assert token not in query_upper
        if sample.query_type == "hard_pseudo_anchor":
            query_lower = sample.query.lower()
            assert any(keyword in query_lower for keyword in anchor_triggers)
        assert 1 <= len(sample.distractors) <= 3
        for dspan in sample.distractors:
            assert _iou(sample.gt_span, dspan) < 0.1
