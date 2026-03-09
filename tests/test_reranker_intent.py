from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker import Hit, rerank
from pov_compiler.schemas import Event, Output, TokenCodec


def _context() -> Output:
    return Output(
        video_id="demo",
        meta={"duration_s": 60.0},
        events=[Event(id="event_0001", t0=0.0, t1=60.0, scores={}, anchors=[])],
        token_codec=TokenCodec(version="0.2", vocab=[], tokens=[]),
    )


def test_reranker_token_intent_prefers_token_kind() -> None:
    hits: list[Hit] = [
        Hit(kind="highlight", id="hl_1", t0=10.0, t1=12.0, score=0.8, source_query="q_hl", meta={}),
        Hit(kind="token", id="tok_1", t0=10.5, t1=11.0, score=0.8, source_query="q_tok", meta={"token_type": "SCENE_CHANGE"}),
    ]
    plan = QueryPlan(intent="token", candidates=[], constraints={}, debug={})
    ranked = rerank(hits, plan=plan, context=_context())
    assert ranked
    assert str(ranked[0]["kind"]) == "token"
