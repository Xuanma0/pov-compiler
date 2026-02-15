from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import QueryCandidate, QueryPlan


def _hits() -> list[dict]:
    return [
        {
            "kind": "event",
            "id": "ev_a",
            "t0": 1.0,
            "t1": 2.0,
            "score": 0.4,
            "source_query": "q",
            "meta": {"interaction_primary_object": "door"},
        },
        {
            "kind": "event",
            "id": "ev_b",
            "t0": 2.5,
            "t1": 3.5,
            "score": 0.4,
            "source_query": "q",
            "meta": {"interaction_primary_object": "cup"},
        },
        {
            "kind": "event",
            "id": "ev_c",
            "t0": 4.0,
            "t1": 5.0,
            "score": 0.4,
            "source_query": "q",
            "meta": {},
        },
    ]


def test_chain_object_match_hard_filters_and_traces_details() -> None:
    plan = QueryPlan(
        intent="mixed",
        candidates=[QueryCandidate(query="token=SCENE_CHANGE top_k=6", reason="test", priority=0)],
        constraints={
            "chain_object_mode": "hard",
            "chain_object_value": "door",
            "chain_time_mode": "off",
        },
    )
    result = apply_constraints_detailed(_hits(), plan, cfg=HardConstraintConfig(), output=None)
    assert result.filtered_before == 3
    assert result.filtered_after == 1
    assert [str(h["id"]) for h in result.hits] == ["ev_a"]
    assert "chain_object_match" in result.applied

    step = next((s for s in result.steps if str(s.name) == "chain_object_match"), None)
    assert step is not None
    assert int(step.before) == 3
    assert int(step.after) == 1
    assert str(step.details.get("object_name", "")) == "door"

