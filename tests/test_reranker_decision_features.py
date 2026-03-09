from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker import Hit, score_hit_components
from pov_compiler.schemas import Event, Output, TokenCodec


def _context() -> Output:
    return Output(
        video_id="demo",
        meta={"duration_s": 60.0},
        events=[Event(id="event_0001", t0=0.0, t1=60.0, scores={}, anchors=[])],
        token_codec=TokenCodec(version="0.2", vocab=[], tokens=[]),
    )


def test_decision_feature_trigger_match_increases_score() -> None:
    plan = QueryPlan(
        intent="decision",
        candidates=[],
        constraints={"anchor_type": "turn_head", "decision_type": "ATTENTION_TURN_HEAD"},
        debug={},
    )
    good: Hit = Hit(
        kind="decision",
        id="dp_good",
        t0=10.0,
        t1=11.0,
        score=0.4,
        source_query="decision=ATTENTION_TURN_HEAD top_k=6",
        meta={
            "action_type": "ATTENTION_TURN_HEAD",
            "trigger_anchor_types": ["turn_head"],
            "state_scene_change_nearby": True,
            "evidence_coverage": 0.8,
        },
    )
    bad: Hit = Hit(
        kind="decision",
        id="dp_bad",
        t0=10.0,
        t1=11.0,
        score=0.4,
        source_query="decision=ATTENTION_TURN_HEAD top_k=6",
        meta={
            "action_type": "ATTENTION_TURN_HEAD",
            "trigger_anchor_types": ["stop_look"],
            "state_scene_change_nearby": True,
            "evidence_coverage": 0.8,
        },
    )

    good_parts = score_hit_components(hit=good, plan=plan, context=_context())
    bad_parts = score_hit_components(hit=bad, plan=plan, context=_context())

    assert float(good_parts["trigger_match"]) > float(bad_parts["trigger_match"])
    assert float(good_parts["decision_align_score"]) > float(bad_parts["decision_align_score"])
    assert float(good_parts["total"]) > float(bad_parts["total"])


def test_decision_features_missing_fields_are_safe() -> None:
    plan = QueryPlan(intent="decision", candidates=[], constraints={}, debug={})
    hit: Hit = Hit(kind="decision", id="dp_1", t0=1.0, t1=1.2, score=0.1, source_query="", meta={})
    parts = score_hit_components(hit=hit, plan=plan, context=_context())
    assert float(parts["decision_align_score"]) >= 0.0
    assert "trigger_match" in parts
    assert "evidence_quality" in parts


def test_non_decision_hit_keeps_decision_features_zero() -> None:
    plan = QueryPlan(intent="token", candidates=[], constraints={"token_type": "SCENE_CHANGE"}, debug={})
    hit: Hit = Hit(kind="token", id="tok_1", t0=2.0, t1=2.1, score=0.3, source_query="", meta={"token_type": "SCENE_CHANGE"})
    parts = score_hit_components(hit=hit, plan=plan, context=_context())
    assert float(parts["decision_align_score"]) == 0.0
    assert float(parts["trigger_match"]) == 0.0
    assert float(parts["action_match"]) == 0.0
