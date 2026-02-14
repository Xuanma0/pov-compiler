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
        video_id="chain_retriever_demo",
        meta={"duration_s": 20.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=8.0, anchors=[Anchor(type="stop_look", t=3.0, conf=0.8)]),
            Event(id="event_0002", t0=8.0, t1=20.0, anchors=[Anchor(type="turn_head", t=12.0, conf=0.85)]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=4.0,
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
                    id="tok_0001",
                    t0=2.0,
                    t1=3.9,
                    type="HIGHLIGHT",
                    conf=0.8,
                    source_event="event_0001",
                    source={"highlight_id": "hl_0001"},
                ),
                Token(
                    id="tok_0002",
                    t0=11.0,
                    t1=13.0,
                    type="HIGHLIGHT",
                    conf=0.9,
                    source_event="event_0002",
                    source={"highlight_id": "hl_0002"},
                ),
            ],
        ),
    )


def test_retrieve_chain_derives_time_constraints() -> None:
    retriever = Retriever(output_json=_make_output(), config={"default_top_k": 6})
    result = retriever.retrieve("anchor=stop_look top_k=6 then token=HIGHLIGHT top_k=6")
    debug = dict(result.get("debug", {}))
    chain = dict(debug.get("chain", {}))
    assert bool(chain.get("is_chain", False))
    assert str(chain.get("chain_rel", "")) == "after"
    derived = dict(chain.get("derived_constraints", {}))
    assert float(derived.get("t_min_s", 0.0)) >= 4.0

    step2 = dict(chain.get("step2", {}))
    before = int(step2.get("filtered_hits_before", 0))
    after = int(step2.get("filtered_hits_after", 0))
    assert before >= after
    assert after >= 1
    assert "time=" in str(step2.get("query_derived", ""))

