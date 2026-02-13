from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker import Hit, rerank
from pov_compiler.schemas import Event, Output, Token, TokenCodec


def _context_with_scene(scene_t0: float = 10.0, scene_t1: float = 10.2) -> Output:
    return Output(
        video_id="demo",
        meta={"duration_s": 120.0},
        events=[Event(id="event_0001", t0=0.0, t1=120.0, scores={}, anchors=[])],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(
                    id="tok_scene",
                    t0=scene_t0,
                    t1=scene_t1,
                    type="SCENE_CHANGE",
                    conf=0.9,
                    source_event="event_0001",
                )
            ],
        ),
    )


def test_reranker_first_constraint_prefers_earliest_hit() -> None:
    hits: list[Hit] = [
        Hit(kind="highlight", id="hl_late", t0=20.0, t1=22.0, score=0.8, source_query="q1", meta={}),
        Hit(kind="highlight", id="hl_early", t0=5.0, t1=7.0, score=0.8, source_query="q1", meta={}),
        Hit(kind="highlight", id="hl_mid", t0=12.0, t1=14.0, score=0.8, source_query="q1", meta={}),
    ]
    plan = QueryPlan(intent="anchor", candidates=[], constraints={"which": "first"}, debug={})
    ranked = rerank(hits, plan=plan, context=_context_with_scene())
    assert ranked
    assert str(ranked[0]["id"]) == "hl_early"


def test_reranker_after_scene_change_penalizes_before_scene_hits() -> None:
    hits: list[Hit] = [
        Hit(kind="highlight", id="hl_before", t0=3.0, t1=4.0, score=0.95, source_query="q1", meta={}),
        Hit(kind="highlight", id="hl_after", t0=12.0, t1=13.0, score=0.5, source_query="q1", meta={}),
    ]
    plan = QueryPlan(intent="anchor", candidates=[], constraints={"after_scene_change": True}, debug={})
    ranked = rerank(hits, plan=plan, context=_context_with_scene(scene_t0=10.0, scene_t1=10.1))
    assert ranked
    assert str(ranked[0]["id"]) == "hl_after"
