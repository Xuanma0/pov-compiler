from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import QueryPlan


def test_constraints_object_match_and_last() -> None:
    hits = [
        {
            "kind": "event",
            "id": "ev_old_door",
            "t0": 2.0,
            "t1": 3.0,
            "score": 0.8,
            "source_query": "lost_object=door top_k=6",
            "meta": {"interaction_primary_object": "door"},
        },
        {
            "kind": "event",
            "id": "ev_latest_door",
            "t0": 8.0,
            "t1": 9.0,
            "score": 0.7,
            "source_query": "lost_object=door top_k=6",
            "meta": {"interaction_primary_object": "door"},
        },
        {
            "kind": "event",
            "id": "ev_phone",
            "t0": 10.0,
            "t1": 11.0,
            "score": 0.9,
            "source_query": "lost_object=door top_k=6",
            "meta": {"interaction_primary_object": "phone"},
        },
    ]
    plan = QueryPlan(
        intent="mixed",
        constraints={"object_name": "door", "which": "last"},
        candidates=[],
        debug={},
    )
    cfg = HardConstraintConfig(
        enable_after_scene_change=False,
        enable_type_match=False,
        enable_interaction=False,
        enable_object_match=True,
        enable_place=False,
        enable_first_last=True,
        relax_on_empty=False,
    )
    result = apply_constraints_detailed(hits=hits, query_plan=plan, cfg=cfg, output=None)
    assert "object_match" in result.applied
    assert "first_last" in result.applied
    assert result.filtered_before == 3
    assert result.filtered_after == 1
    assert result.hits[0]["id"] == "ev_latest_door"
