from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Anchor, Event, KeyClip, Output, Token, TokenCodec


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
                anchors=[Anchor(type="stop_look", t=3.0, conf=0.8, meta={"duration_s": 1.0})],
            ),
            Event(
                id="event_0002",
                t0=10.0,
                t1=20.0,
                scores={"boundary_conf": 0.9},
                anchors=[Anchor(type="turn_head", t=12.0, conf=0.85, meta={})],
            ),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=5.0,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=3.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=11.0,
                t1=13.0,
                source_event="event_0002",
                anchor_type="turn_head",
                anchor_t=12.0,
                conf=0.9,
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
                    t1=5.0,
                    type="HIGHLIGHT",
                    conf=0.8,
                    source_event="event_0001",
                    source={"highlight_id": "hl_0001"},
                    meta={},
                ),
                Token(
                    id="tok_000002",
                    t0=3.0,
                    t1=4.0,
                    type="ATTENTION_STOP_LOOK",
                    conf=0.8,
                    source_event="event_0001",
                    source={},
                    meta={},
                ),
                Token(
                    id="tok_000003",
                    t0=11.0,
                    t1=13.0,
                    type="HIGHLIGHT",
                    conf=0.9,
                    source_event="event_0002",
                    source={"highlight_id": "hl_0002"},
                    meta={},
                ),
                Token(
                    id="tok_000004",
                    t0=11.5,
                    t1=12.5,
                    type="ATTENTION_TURN_HEAD",
                    conf=0.85,
                    source_event="event_0002",
                    source={},
                    meta={},
                ),
            ],
        ),
    )


def test_retriever_time_query() -> None:
    retriever = Retriever(output_json=_make_output())
    result = retriever.retrieve("time=1-6 top_k=6")
    assert "hl_0001" in result["selected_highlights"]
    assert "hl_0002" not in result["selected_highlights"]


def test_retriever_anchor_query() -> None:
    retriever = Retriever(output_json=_make_output())
    result = retriever.retrieve("anchor=stop_look top_k=6")
    assert result["selected_highlights"] == ["hl_0001"]


def test_retriever_text_query_without_clip_or_index() -> None:
    retriever = Retriever(output_json=_make_output())
    result = retriever.retrieve("text=door top_k=3")
    assert result["selected_highlights"] == []
    reason = str(result.get("debug", {}).get("reason", "")).lower()
    assert "text query" in reason
