from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import load_hard_pseudo_nlq
from pov_compiler.schemas import DecisionPoint, Event, EventV1, KeyClip, Output, Token, TokenCodec


def test_hard_pseudo_place_and_interaction_samples_are_generated() -> None:
    output = Output(
        video_id="hard_place_interaction_demo",
        meta={"duration_s": 120.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=40.0),
            Event(id="event_0002", t0=40.0, t1=80.0),
            Event(id="event_0003", t0=80.0, t1=120.0),
        ],
        events_v1=[
            EventV1(
                id="event_0001",
                t0=0.0,
                t1=40.0,
                place_segment_id="place_0001",
                interaction_primary_object="cup",
                interaction_score=0.75,
            ),
            EventV1(
                id="event_0002",
                t0=40.0,
                t1=80.0,
                place_segment_id="place_0001",
                interaction_primary_object="phone",
                interaction_score=0.6,
            ),
            EventV1(
                id="event_0003",
                t0=80.0,
                t1=120.0,
                place_segment_id="place_0002",
                interaction_primary_object="door",
                interaction_score=0.3,
            ),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=10.0,
                t1=12.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=11.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=50.0,
                t1=52.0,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=51.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            tokens=[Token(id="tok_sc_1", t0=39.8, t1=40.2, type="SCENE_CHANGE", conf=0.9, source_event="event_0001")]
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=11.0,
                t0=10.5,
                t1=11.5,
                source_event="event_0001",
                source_highlight="hl_0001",
                action={"type": "ATTENTION_TURN_HEAD"},
            )
        ],
    )

    samples = load_hard_pseudo_nlq(output, seed=0, n_highlight=6, n_token=3, n_decision=2, top_k=6)
    qtypes = {sample.query_type for sample in samples}
    assert "hard_pseudo_place" in qtypes
    assert "hard_pseudo_interaction" in qtypes
    for sample in samples:
        assert float(sample.gt_span[1]) >= float(sample.gt_span[0])
