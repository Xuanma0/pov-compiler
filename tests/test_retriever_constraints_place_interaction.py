from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import EventV1, Output


def _make_output() -> Output:
    return Output(
        video_id="retriever_place_interaction_demo",
        meta={"duration_s": 20.0},
        events_v1=[
            EventV1(
                id="event_0001",
                t0=0.0,
                t1=5.0,
                label="navigation",
                place_segment_id="place_a",
                interaction_primary_object="cup",
                interaction_score=0.2,
            ),
            EventV1(
                id="event_0002",
                t0=5.0,
                t1=10.0,
                label="interaction-heavy",
                place_segment_id="place_a",
                interaction_primary_object="cup",
                interaction_score=0.8,
            ),
            EventV1(
                id="event_0003",
                t0=10.0,
                t1=15.0,
                label="interaction-heavy",
                place_segment_id="place_b",
                interaction_primary_object="phone",
                interaction_score=0.9,
            ),
        ],
    )


def test_retriever_place_segment_and_first_last_constraint() -> None:
    retriever = Retriever(output_json=_make_output(), index=None, config={"default_top_k": 8})
    result = retriever.retrieve("place_segment_id=place_a place=last top_k=8")
    assert result["selected_events"] == ["event_0002"]


def test_retriever_interaction_constraints() -> None:
    retriever = Retriever(output_json=_make_output(), index=None, config={"default_top_k": 8})
    result = retriever.retrieve("interaction_object=phone interaction_min=0.5 top_k=8")
    assert "event_0003" in result["selected_events"]
    assert "event_0001" not in result["selected_events"]
