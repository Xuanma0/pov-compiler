from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.retrieval.trace import trace_query
from pov_compiler.schemas import Anchor, Event, KeyClip, Output, Token, TokenCodec


def _make_output() -> Output:
    out = Output(
        video_id="trace_demo",
        meta={"duration_s": 16.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=8.0, scores={"boundary_conf": 0.6}, anchors=[Anchor(type="turn_head", t=2.5, conf=0.8)]),
            Event(id="event_0002", t0=8.0, t1=16.0, scores={"boundary_conf": 0.7}, anchors=[Anchor(type="stop_look", t=10.0, conf=0.75)]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=3.5,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=2.5,
                conf=0.85,
                meta={"anchor_types": ["turn_head"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=2.0, t1=3.5, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001")
            ],
        ),
    )
    return ensure_events_v1(out)


def test_trace_query_has_required_fields() -> None:
    trace = trace_query(output_json=_make_output(), query="When did I start looking around?", top_k=4)
    assert trace["video_id"] == "trace_demo"
    assert "plan" in trace
    assert "constraint_trace" in trace
    assert "hits" in trace
    assert isinstance(trace["hits"], list)
    assert trace["constraint_trace"]["filtered_hits_before"] >= 0


def test_trace_query_hits_have_evidence_spans() -> None:
    trace = trace_query(output_json=_make_output(), query="anchor=turn_head top_k=4", top_k=4)
    hits = trace["hits"]
    assert hits
    assert any(len(hit.get("evidence_spans", [])) > 0 for hit in hits)

