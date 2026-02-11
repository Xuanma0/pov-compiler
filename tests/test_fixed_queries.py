from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.fixed_queries import generate_fixed_queries
from pov_compiler.schemas import Anchor, DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _build_full_output() -> Output:
    return Output(
        video_id="demo_fixed_queries",
        meta={"duration_s": 40.0},
        events=[
            Event(
                id="event_0001",
                t0=0.0,
                t1=40.0,
                scores={"boundary_conf": 0.8},
                anchors=[
                    Anchor(type="turn_head", t=10.0, conf=0.9, meta={}),
                    Anchor(type="stop_look", t=22.0, conf=0.85, meta={"duration_s": 1.2}),
                ],
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
                    id="tok_000001",
                    t0=8.0,
                    t1=12.0,
                    type="ATTENTION_TURN_HEAD",
                    conf=0.9,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_000002",
                    t0=20.0,
                    t1=21.0,
                    type="SCENE_CHANGE",
                    conf=0.8,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_000001",
                t=20.5,
                t0=20.0,
                t1=21.0,
                source_event="event_0001",
                source_highlight=None,
                trigger={"anchor_types": ["turn_head"], "token_ids": ["tok_000002"]},
                state={"nearby_tokens": ["SCENE_CHANGE"], "evidence": {"token_ids": ["tok_000002"]}},
                action={"type": "ATTENTION_TURN_HEAD", "conf": 0.8},
                constraints=[],
                outcome={"type": "SCENE_CHANGED", "conf": 0.8, "evidence": {"token_ids": ["tok_000002"]}},
                alternatives=[],
                conf=0.8,
            )
        ],
    )


def test_fixed_queries_deterministic_with_seed() -> None:
    output = _build_full_output()
    q1 = generate_fixed_queries(
        output=output,
        seed=7,
        n_time=4,
        n_anchor=2,
        n_token=2,
        n_decision=2,
        n_hard_time=3,
        time_window_s=6.0,
        default_top_k=6,
        hard_overlap_thresh=0.1,
    )
    q2 = generate_fixed_queries(
        output=output,
        seed=7,
        n_time=4,
        n_anchor=2,
        n_token=2,
        n_decision=2,
        n_hard_time=3,
        time_window_s=6.0,
        default_top_k=6,
        hard_overlap_thresh=0.1,
    )
    assert [x.to_dict() for x in q1] == [x.to_dict() for x in q2]
    assert len(q1) > 0
    kinds = {q.type for q in q1}
    assert "time" in kinds
    assert "token" in kinds
    assert "decision" in kinds
    assert "hard_time" in kinds
