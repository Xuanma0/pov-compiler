from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.l3_decisions.decision_compiler import DecisionCompiler
from pov_compiler.schemas import Anchor, Event, KeyClip, Output, Token, TokenCodec


def test_decision_compiler_basics() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 20.0},
        events=[
            Event(
                id="event_0001",
                t0=0.0,
                t1=10.0,
                scores={"boundary_conf": 0.7},
                anchors=[Anchor(type="stop_look", t=3.0, conf=0.9, meta={"duration_s": 1.0})],
            ),
            Event(
                id="event_0002",
                t0=10.0,
                t1=20.0,
                scores={"boundary_conf": 0.8},
                anchors=[Anchor(type="turn_head", t=12.0, conf=0.85, meta={})],
            ),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=4.5,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=3.0,
                conf=0.88,
                meta={"anchor_types": ["stop_look"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=11.0,
                t1=13.0,
                source_event="event_0002",
                anchor_type="turn_head",
                anchor_t=12.0,
                conf=0.86,
                meta={"anchor_types": ["turn_head"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(
                    id="tok_000001",
                    t0=2.0,
                    t1=4.5,
                    type="HIGHLIGHT",
                    conf=0.88,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_000002",
                    t0=11.0,
                    t1=13.0,
                    type="HIGHLIGHT",
                    conf=0.86,
                    source_event="event_0002",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_000003",
                    t0=11.8,
                    t1=12.2,
                    type="SCENE_CHANGE",
                    conf=0.91,
                    source_event="event_0002",
                    source={},
                    meta={},
                ),
            ],
        ),
        debug={
            "signals": {
                "time": [float(i) for i in range(20)],
                "motion_energy": [0.1, 0.2, 0.1, 0.12, 0.2, 0.3, 0.4, 0.35, 0.3, 0.25, 0.2, 0.3, 0.7, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1],
                "embed_dist": [0.1, 0.1, 0.12, 0.15, 0.2, 0.25, 0.2, 0.2, 0.18, 0.2, 0.3, 0.55, 0.7, 0.6, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
                "boundary_score": [0.1, 0.2, 0.15, 0.2, 0.25, 0.3, 0.35, 0.25, 0.2, 0.25, 0.4, 0.7, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1],
            }
        },
    )

    compiler = DecisionCompiler(config={"pre_s": 2.0, "post_s": 2.0, "merge_iou": 0.7})
    decisions = compiler.compile(output)
    assert len(decisions) > 0
    assert all(len(decision.alternatives) >= 2 for decision in decisions)
    assert all(str(decision.outcome.get("type", "")) != "" for decision in decisions)
    assert [decision.id for decision in decisions] == [f"dp_{i:06d}" for i in range(1, len(decisions) + 1)]
    assert all(decisions[i].t <= decisions[i + 1].t for i in range(len(decisions) - 1))
