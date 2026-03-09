from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import QueryPlan


def _hits_for_lost_object() -> list[dict]:
    hits: list[dict] = []
    for idx in range(28):
        obj = "door" if idx == 17 else "cup"
        hits.append(
            {
                "kind": "event",
                "id": f"event_{idx:04d}",
                "t0": float(idx),
                "t1": float(idx) + 0.8,
                "score": 1.0,
                "source_query": "lost_object=door top_k=6",
                "meta": {
                    "interaction_primary_object": obj,
                    "object_name": obj,
                },
            }
        )
    return hits


def test_lost_object_constraint_steps_include_object_match() -> None:
    plan = QueryPlan(
        intent="mixed",
        constraints={
            "lost_object": "door",
            "object_name": "door",
            "which": "last",
            "need_object_match": True,
        },
        candidates=[],
        debug={},
    )
    cfg = HardConstraintConfig(
        enable_after_scene_change=False,
        enable_first_last=True,
        enable_type_match=False,
        enable_interaction=False,
        enable_object_match=True,
        enable_place=False,
        relax_on_empty=False,
    )
    result = apply_constraints_detailed(hits=_hits_for_lost_object(), query_plan=plan, cfg=cfg, output=None)

    assert result.filtered_before == 28
    assert result.filtered_after == 1
    assert "object_match" in result.applied
    assert "first_last" in result.applied

    by_name = {step.name: step for step in result.steps}
    assert "object_match" in by_name
    assert by_name["object_match"].before == 28
    assert by_name["object_match"].after == 1
    assert "first_last" in by_name
    assert by_name["first_last"].before == 1
    assert by_name["first_last"].after == 1
