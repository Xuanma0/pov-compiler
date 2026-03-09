from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import convert_output_to_events_v1, ensure_events_v1
from pov_compiler.schemas import Anchor, DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="demo",
        meta={"duration_s": 20.0},
        events=[
            Event(
                id="event_0001",
                t0=0.0,
                t1=10.0,
                scores={"boundary_conf": 0.7},
                anchors=[Anchor(type="turn_head", t=3.0, conf=0.8, meta={})],
            )
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=4.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=3.0,
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
                    t0=2.0,
                    t1=4.0,
                    type="HIGHLIGHT",
                    conf=0.9,
                    source_event="event_0001",
                    source={},
                    meta={},
                )
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_000001",
                t=3.0,
                t0=2.5,
                t1=3.5,
                source_event="event_0001",
                source_highlight="hl_0001",
                trigger={},
                state={},
                action={"type": "ATTENTION_TURN_HEAD"},
                constraints=[],
                outcome={},
                alternatives=[],
                conf=0.8,
            )
        ],
        perception={
            "frames": [
                {
                    "t": 3.1,
                    "contact": {
                        "active": {
                            "object_id": "obj_1",
                            "label": "cup",
                            "hand_id": "hand_1",
                            "score": 0.72,
                        }
                    },
                }
            ]
        },
    )


def test_events_v1_converter_collects_evidence() -> None:
    output = _make_output()
    events_v1 = convert_output_to_events_v1(output, rerank_cfg_hash="abc123")
    assert events_v1
    event = events_v1[0]
    assert event.id == "event_0001"
    assert event.label
    evidence_types = {e.type for e in event.evidence}
    assert {"anchor", "highlight", "token", "decision", "contact"}.issubset(evidence_types)
    assert event.meta.get("constraint_trace")
    assert event.meta.get("retrieval_hit")
    for evd in event.evidence:
        assert evd.retrieval_hit is not None
        assert float(evd.t1) >= float(evd.t0)
        assert float(evd.retrieval_hit.t1) >= float(evd.retrieval_hit.t0)
        assert str(evd.retrieval_hit.id) == str(evd.id)


def test_ensure_events_v1_is_idempotent() -> None:
    output = _make_output()
    out1 = ensure_events_v1(output)
    out2 = ensure_events_v1(out1)
    assert len(out1.events_v1) == len(out2.events_v1)
    assert out2.events_v1[0].id == out1.events_v1[0].id
