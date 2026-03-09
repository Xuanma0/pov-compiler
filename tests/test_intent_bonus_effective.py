from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker import Hit, rerank
from pov_compiler.retrieval.reranker_config import WeightConfig
from pov_compiler.schemas import Event, Output, TokenCodec


def _context() -> Output:
    return Output(
        video_id="demo",
        meta={"duration_s": 30.0},
        events=[Event(id="event_0001", t0=0.0, t1=30.0, scores={}, anchors=[])],
        token_codec=TokenCodec(version="0.2", vocab=[], tokens=[]),
    )


def test_intent_bonus_changes_top1_for_token_intent() -> None:
    hits: list[Hit] = [
        Hit(kind="highlight", id="hl_1", t0=10.0, t1=12.0, score=0.7, source_query="q_hl", meta={}),
        Hit(kind="token", id="tok_1", t0=10.5, t1=11.0, score=0.7, source_query="q_tok", meta={"token_type": "SCENE_CHANGE"}),
    ]
    plan = QueryPlan(intent="token", candidates=[], constraints={}, debug={})
    cfg = WeightConfig.from_dict(
        {
            "bonus_intent_token_on_token": 1.0,
            "bonus_intent_token_on_highlight": 0.0,
            "bonus_intent_token_on_decision": 0.0,
            "bonus_intent_token_on_event": 0.0,
            "bonus_conf_scale": 0.0,
            "bonus_boundary_scale": 0.0,
            "bonus_priority_scale": 0.0,
            "penalty_distractor_near": 0.0,
        }
    )
    ranked = rerank(hits, plan=plan, context=_context(), cfg=cfg)
    assert ranked
    assert str(ranked[0]["kind"]) == "token"

