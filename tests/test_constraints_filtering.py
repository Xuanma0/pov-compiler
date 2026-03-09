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
from pov_compiler.schemas import Event, Output, Token, TokenCodec


def _output_with_scene(t0: float = 10.0, t1: float = 10.1) -> Output:
    return Output(
        video_id="demo",
        meta={"duration_s": 100.0},
        events=[Event(id="event_0001", t0=0.0, t1=100.0, scores={}, anchors=[])],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[Token(id="tok_scene", t0=t0, t1=t1, type="SCENE_CHANGE", conf=0.9, source_event="event_0001")],
        ),
    )


def test_after_scene_and_first_last_filtering() -> None:
    hits: list[Hit] = [
        Hit(kind="highlight", id="h0", t0=2.0, t1=3.0, score=0.9, source_query="q", meta={}),
        Hit(kind="highlight", id="h1", t0=11.0, t1=12.0, score=0.8, source_query="q", meta={}),
        Hit(kind="highlight", id="h2", t0=20.0, t1=21.0, score=0.7, source_query="q", meta={}),
    ]
    plan = QueryPlan(
        intent="anchor",
        candidates=[],
        constraints={"after_scene_change": True, "which": "first"},
        debug={},
    )
    cfg = HardConstraintConfig(
        enable_after_scene_change=True,
        enable_first_last=True,
        enable_type_match=False,
        relax_on_empty=True,
    )
    result = apply_constraints_detailed(hits, plan, cfg=cfg, output=_output_with_scene())
    assert result.hits
    assert str(result.hits[0]["id"]) == "h1"
    assert "after_scene_change" in result.applied
    assert "first_last" in result.applied
    assert not result.used_fallback


def test_relax_on_empty_fallback() -> None:
    hits: list[Hit] = [
        Hit(kind="highlight", id="h0", t0=2.0, t1=3.0, score=0.9, source_query="q", meta={}),
        Hit(kind="highlight", id="h1", t0=4.0, t1=5.0, score=0.8, source_query="q", meta={}),
    ]
    plan = QueryPlan(
        intent="anchor",
        candidates=[],
        constraints={"after_scene_change": True},
        debug={},
    )
    cfg = HardConstraintConfig(
        enable_after_scene_change=True,
        enable_first_last=False,
        enable_type_match=False,
        relax_on_empty=True,
        relax_order=["after_scene_change"],
    )
    # scene change after all hits -> empty then relax back
    result = apply_constraints_detailed(hits, plan, cfg=cfg, output=_output_with_scene(t0=50.0, t1=50.1))
    assert len(result.hits) == len(hits)
    assert not result.used_fallback
    assert "after_scene_change" in result.relaxed
