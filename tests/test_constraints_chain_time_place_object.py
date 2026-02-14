from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import QueryCandidate, QueryPlan
from pov_compiler.retrieval.reranker import rerank
from pov_compiler.retrieval.reranker_config import WeightConfig
from pov_compiler.schemas import Output


def _hits() -> list[dict]:
    return [
        {
            "kind": "event",
            "id": "ev_1",
            "t0": 1.0,
            "t1": 2.0,
            "score": 0.5,
            "source_query": "q",
            "meta": {"place_segment_id": "place_a", "interaction_primary_object": "door"},
        },
        {
            "kind": "event",
            "id": "ev_2",
            "t0": 3.0,
            "t1": 4.0,
            "score": 0.5,
            "source_query": "q",
            "meta": {"place_segment_id": "place_b", "interaction_primary_object": "door"},
        },
        {
            "kind": "event",
            "id": "ev_3",
            "t0": 5.0,
            "t1": 6.0,
            "score": 0.5,
            "source_query": "q",
            "meta": {"place_segment_id": "place_a", "interaction_primary_object": "cup"},
        },
    ]


def test_chain_time_place_object_hard_filters_apply() -> None:
    plan = QueryPlan(
        intent="mixed",
        candidates=[QueryCandidate(query="token=SCENE_CHANGE top_k=6", reason="test", priority=0)],
        constraints={
            "chain_time_mode": "hard",
            "chain_time_min_s": 0.0,
            "chain_time_max_s": 10.0,
            "chain_place_mode": "hard",
            "chain_place_value": "place_a",
            "chain_object_mode": "hard",
            "chain_object_value": "door",
        },
    )
    result = apply_constraints_detailed(_hits(), plan, cfg=HardConstraintConfig(), output=None)
    assert result.filtered_before == 3
    assert result.filtered_after == 1
    assert [h["id"] for h in result.hits] == ["ev_1"]
    assert "chain_time_range" in result.applied
    assert "chain_place_match" in result.applied
    assert "chain_object_match" in result.applied


def test_chain_soft_modes_keep_hits_and_add_rerank_bonus() -> None:
    plan = QueryPlan(
        intent="mixed",
        candidates=[QueryCandidate(query="token=SCENE_CHANGE top_k=6", reason="test", priority=0)],
        constraints={
            "chain_time_mode": "off",
            "chain_place_mode": "soft",
            "chain_place_value": "place_a",
            "chain_object_mode": "soft",
            "chain_object_value": "door",
        },
    )
    result = apply_constraints_detailed(_hits(), plan, cfg=HardConstraintConfig(), output=None)
    assert result.filtered_before == 3
    assert result.filtered_after == 3
    assert "chain_place_match" in result.applied
    assert "chain_object_match" in result.applied

    ranked = rerank(
        result.hits,
        plan=plan,
        context=Output(video_id="demo", meta={"duration_s": 10.0}),
        cfg=WeightConfig(),
        distractors=None,
        constraint_trace=None,
    )
    assert ranked
    top = ranked[0]
    components = dict(top.get("meta", {}).get("rerank", {}).get("components", {}))
    assert "place_match_bonus" in components
    assert "object_match_bonus" in components
    # ev_1 matches both place/object soft constraints; ev_2/ev_3 only one each.
    assert str(top["id"]) == "ev_1"
