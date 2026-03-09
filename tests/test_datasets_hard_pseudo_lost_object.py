from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import load_hard_pseudo_nlq
from pov_compiler.schemas import Event, EventV1, KeyClip, ObjectMemoryItemV0, Output


def test_hard_pseudo_lost_object_generation() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 20.0},
        events=[Event(id="event_0001", t0=0.0, t1=8.0), Event(id="event_0002", t0=8.0, t1=16.0)],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=1.0,
                t1=3.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=2.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=10.0,
                t1=12.0,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=11.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        events_v1=[
            EventV1(
                id="ev1",
                t0=2.0,
                t1=4.0,
                place_segment_id="place_a",
                interaction_primary_object="door",
                interaction_score=0.7,
            ),
            EventV1(id="ev2", t0=6.0, t1=8.0, place_segment_id="place_b", interaction_primary_object="cup"),
            EventV1(id="ev3", t0=10.0, t1=12.0, place_segment_id="place_c", interaction_primary_object="phone"),
        ],
        object_memory_v0=[
            ObjectMemoryItemV0(
                object_name="door",
                last_seen_t_ms=12500,
                last_contact_t_ms=12000,
                last_place_id="place_c",
                score=0.9,
            )
        ],
    )

    samples = load_hard_pseudo_nlq(output, seed=0, n_highlight=8, n_token=4, n_decision=4, top_k=6)
    lost = [s for s in samples if s.query_type == "hard_pseudo_lost_object"]
    assert lost, "expected hard_pseudo_lost_object samples"
    sample = lost[0]
    assert "door" in sample.query.lower()
    assert ("lost_object=door" in sample.query.lower()) or ("object_last_seen=door" in sample.query.lower())
    assert sample.gt_span[0] <= 12.0 <= sample.gt_span[1]
    assert sample.distractors
