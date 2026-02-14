from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import load_hard_pseudo_chain
from pov_compiler.schemas import DecisionPoint, Event, EventV1, KeyClip, ObjectMemoryItemV0, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="hard_chain_demo",
        meta={"duration_s": 120.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=40.0),
            Event(id="event_0002", t0=40.0, t1=80.0),
            Event(id="event_0003", t0=80.0, t1=120.0),
        ],
        events_v1=[
            EventV1(
                id="ev1_0001",
                t0=0.0,
                t1=40.0,
                place_segment_id="place_0001",
                interaction_primary_object="door",
                interaction_score=0.75,
            ),
            EventV1(
                id="ev1_0002",
                t0=40.0,
                t1=80.0,
                place_segment_id="place_0001",
                interaction_primary_object="phone",
                interaction_score=0.55,
            ),
            EventV1(
                id="ev1_0003",
                t0=80.0,
                t1=120.0,
                place_segment_id="place_0002",
                interaction_primary_object="door",
                interaction_score=0.30,
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
                conf=0.82,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=52.0,
                t1=54.0,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=53.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=39.8, t1=40.2, type="SCENE_CHANGE", conf=0.9, source_event="event_0001"),
                Token(id="tok_0002", t0=11.0, t1=12.0, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0003", t0=53.0, t1=54.0, type="ATTENTION_STOP_LOOK", conf=0.8, source_event="event_0002"),
            ],
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
                trigger={"anchor_types": ["turn_head"]},
                conf=0.8,
            ),
            DecisionPoint(
                id="dp_0002",
                t=53.0,
                t0=52.5,
                t1=53.5,
                source_event="event_0002",
                source_highlight="hl_0002",
                action={"type": "ATTENTION_STOP_LOOK"},
                trigger={"anchor_types": ["stop_look"]},
                conf=0.78,
            ),
        ],
        object_memory_v0=[
            ObjectMemoryItemV0(
                object_name="door",
                last_seen_t_ms=95000,
                last_contact_t_ms=93000,
                last_place_id="place_0002",
                score=0.8,
            )
        ],
    )


def test_hard_pseudo_chain_samples_generated() -> None:
    samples = load_hard_pseudo_chain(_make_output(), seed=0, n_chain=6, top_k=6)
    assert samples
    assert all(s.query_type == "hard_pseudo_chain" for s in samples)
    assert any(" then " in str(s.query).lower() for s in samples)
    assert all(isinstance(s.meta.get("chain_meta", {}), dict) for s in samples)

