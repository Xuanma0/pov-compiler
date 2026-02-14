from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker import Hit


def test_place_interaction_constraints_are_applied_and_filter() -> None:
    hits: list[Hit] = [
        Hit(
            kind="event",
            id="event_0001",
            t0=0.0,
            t1=4.0,
            score=0.9,
            source_query="q",
            meta={"place_segment_id": "place_a", "interaction_score": 0.2, "interaction_primary_object": "door"},
        ),
        Hit(
            kind="event",
            id="event_0002",
            t0=4.0,
            t1=8.0,
            score=0.8,
            source_query="q",
            meta={"place_segment_id": "place_a", "interaction_score": 0.8, "interaction_primary_object": "door"},
        ),
        Hit(
            kind="event",
            id="event_0003",
            t0=8.0,
            t1=12.0,
            score=0.7,
            source_query="q",
            meta={"place_segment_id": "place_b", "interaction_score": 0.9, "interaction_primary_object": "cup"},
        ),
    ]
    plan = QueryPlan(
        intent="mixed",
        candidates=[],
        constraints={"place": "first", "interaction_min": 0.3, "interaction_object": "door"},
        debug={},
    )
    cfg = HardConstraintConfig(enable_place=True, enable_interaction=True, relax_on_empty=False)
    result = apply_constraints_detailed(hits, query_plan=plan, cfg=cfg, output=None)

    assert result.applied
    assert "interaction_object" in result.applied
    assert "interaction_min" in result.applied
    assert "place_first_last" in result.applied
    assert result.filtered_after < result.filtered_before
    assert len(result.hits) == 1
    assert str(result.hits[0]["id"]) == "event_0002"
