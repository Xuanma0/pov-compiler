from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, KeyClip, Output, Token, TokenCodec
from pov_compiler.streaming.budget_policy import parse_budget_keys
from pov_compiler.streaming.runner import StreamingConfig, run_streaming


def _output() -> Output:
    return Output(
        video_id="chain_budget_split_demo",
        meta={"duration_s": 32.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=12.0, anchors=[Anchor(type="turn_head", t=4.0, conf=0.8)]),
            Event(id="event_0002", t0=12.0, t1=24.0, anchors=[Anchor(type="stop_look", t=16.0, conf=0.7)]),
            Event(id="event_0003", t0=24.0, t1=32.0, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=3.5,
                t1=4.5,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=4.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=15.5,
                t1=16.5,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=16.0,
                conf=0.75,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=3.5, t1=4.5, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=15.0, t1=17.0, type="SCENE_CHANGE", conf=0.72, source_event="event_0002"),
            ],
        ),
    )


def test_streaming_chain_uses_min_budget_for_step1() -> None:
    budgets = parse_budget_keys("20/50/4,60/200/12")
    payload = run_streaming(
        _output(),
        config=StreamingConfig(
            step_s=8.0,
            top_k=6,
            queries=[
                "lost_object=door which=last top_k=6 then token=SCENE_CHANGE which=last top_k=6 chain_derive=time+object chain_object_mode=hard"
            ],
            budgets=budgets,
            budget_policy="safety_latency_chain",
            latency_cap_ms=100.0,
            max_trials_per_query=2,
            mode="budgeted",
        ),
    )
    rows = [r for r in list(payload.get("query_rows", [])) if int(r.get("final_trial", 0)) == 1]
    assert rows
    row = rows[0]
    assert int(row.get("is_chain", 0)) == 1
    assert float(row.get("chain_step1_budget_seconds", 0.0)) == float(budgets[0].seconds)
    assert float(row.get("chain_step2_budget_seconds", 0.0)) >= float(budgets[0].seconds)

