from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.eval_cross_variant import evaluate_cross_variant
from pov_compiler.eval.fixed_queries import FixedQuery
from pov_compiler.schemas import Anchor, DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _build_full_output() -> Output:
    return Output(
        video_id="demo_cross_variant",
        meta={"duration_s": 30.0},
        events=[
            Event(
                id="event_0001",
                t0=0.0,
                t1=30.0,
                scores={"boundary_conf": 0.9},
                anchors=[Anchor(type="turn_head", t=20.0, conf=0.85, meta={})],
            )
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=8.0,
                t1=12.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=10.0,
                conf=0.9,
                meta={"anchor_types": ["turn_head"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(
                    id="tok_scene",
                    t0=20.0,
                    t1=21.0,
                    type="SCENE_CHANGE",
                    conf=0.9,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_turn",
                    t0=19.8,
                    t1=20.6,
                    type="ATTENTION_TURN_HEAD",
                    conf=0.85,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_000001",
                t=20.2,
                t0=19.5,
                t1=21.2,
                source_event="event_0001",
                source_highlight=None,
                trigger={"anchor_types": ["turn_head"], "token_ids": ["tok_turn", "tok_scene"]},
                state={"nearby_tokens": ["ATTENTION_TURN_HEAD", "SCENE_CHANGE"], "evidence": {"token_ids": ["tok_scene"]}},
                action={"type": "ATTENTION_TURN_HEAD", "conf": 0.88},
                constraints=[],
                outcome={"type": "SCENE_CHANGED", "conf": 0.82, "evidence": {"token_ids": ["tok_scene"]}},
                alternatives=[],
                conf=0.86,
            )
        ],
    )


def _query_lookup(rows: list[dict[str, object]], variant: str, qtype: str) -> dict[str, object]:
    for row in rows:
        if str(row.get("variant")) == variant and str(row.get("query_type")) == qtype:
            return row
    raise AssertionError(f"missing row for variant={variant}, query_type={qtype}")


def test_cross_variant_token_and_decision_gain() -> None:
    output = _build_full_output()
    queries = [
        FixedQuery(
            qid="q_000001",
            type="token",
            query="token=SCENE_CHANGE top_k=6",
            top_k=6,
            relevant={
                "highlights": [],
                "events": ["event_0001"],
                "decisions": ["dp_000001"],
                "tokens": ["tok_scene"],
            },
            meta={},
        ),
        FixedQuery(
            qid="q_000002",
            type="decision",
            query="decision=ATTENTION_TURN_HEAD top_k=6",
            top_k=6,
            relevant={
                "highlights": [],
                "events": ["event_0001"],
                "decisions": ["dp_000001"],
                "tokens": ["tok_turn"],
            },
            meta={},
        ),
        FixedQuery(
            qid="q_000003",
            type="hard_time",
            query="time=19.000-22.000 token=SCENE_CHANGE top_k=6",
            top_k=6,
            time={"t0": 19.0, "t1": 22.0},
            relevant={
                "highlights": [],
                "events": ["event_0001"],
                "decisions": ["dp_000001"],
                "tokens": ["tok_scene"],
            },
            meta={},
        ),
    ]

    result = evaluate_cross_variant(
        full_output=output,
        queries=queries,
        variants=[
            "raw_events_only",
            "highlights_only",
            "highlights_plus_tokens",
            "highlights_plus_decisions",
            "full",
        ],
        budgets={"max_total_s": [60], "max_tokens": [200], "max_decisions": [12]},
        sweep=False,
        retriever_config={"default_top_k": 6},
    )

    by_type = result["by_query_type_rows"]
    per_query = result["per_query_rows"]
    assert len(per_query) == len(queries) * 5

    token_full = float(_query_lookup(by_type, "full", "token")["hit_at_k"])
    token_hl_only = float(_query_lookup(by_type, "highlights_only", "token")["hit_at_k"])
    token_plus = float(_query_lookup(by_type, "highlights_plus_tokens", "token")["hit_at_k"])
    assert token_full > token_hl_only
    assert token_plus > token_hl_only

    decision_full = float(_query_lookup(by_type, "full", "decision")["hit_at_k"])
    decision_hl_only = float(_query_lookup(by_type, "highlights_only", "decision")["hit_at_k"])
    decision_plus = float(_query_lookup(by_type, "highlights_plus_decisions", "decision")["hit_at_k"])
    assert decision_full > decision_hl_only
    assert decision_plus > decision_hl_only

    hard_full = float(_query_lookup(by_type, "full", "hard_time")["hit_at_k_event"])
    hard_raw = float(_query_lookup(by_type, "raw_events_only", "hard_time")["hit_at_k_event"])
    assert hard_full > hard_raw
