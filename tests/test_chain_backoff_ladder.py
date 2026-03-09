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
            "id": "ev_1",
            "t0": 1.0,
            "t1": 2.0,
            "score": 0.4,
            "source_query": "q",
            "meta": {"interaction_primary_object": "cup"},
        },
        {
            "kind": "event",
            "id": "ev_2",
            "t0": 3.0,
            "t1": 4.0,
            "score": 0.3,
            "source_query": "q",
            "meta": {"interaction_primary_object": "bottle"},
        },
    ]


def test_chain_backoff_ladder_relaxes_until_non_empty() -> None:
    plan = QueryPlan(
        intent="mixed",
        candidates=[QueryCandidate(query="token=SCENE_CHANGE top_k=6", reason="test", priority=0)],
        constraints={
            "chain_time_mode": "hard",
            "chain_time_min_s": 0.0,
            "chain_time_max_s": 10.0,
            "chain_object_mode": "hard",
            "chain_object_value": "door",
            "object_name": "door",
            "need_object_match": True,
        },
    )
    result = apply_constraints_detailed(_hits(), plan, cfg=HardConstraintConfig(), output=None)
    assert result.chain_backoff_enabled is True
    assert result.chain_backoff_exhausted is False
    assert result.chain_backoff_chosen_level == 2
    assert len(result.hits) == 2
    assert "chain_time_range" in result.applied
    assert "chain_object_match" not in result.applied
    assert "object_match" not in result.applied

    attempts = list(result.chain_backoff_attempts)
    assert [int(x.get("level", -1)) for x in attempts[:3]] == [0, 1, 2]
    assert int(attempts[0].get("after", -1)) == 0
    assert int(attempts[1].get("after", -1)) == 0
    assert int(attempts[2].get("after", -1)) == 2
