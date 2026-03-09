from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import NLQSample
from pov_compiler.bench.nlq.evaluator import evaluate_nlq_samples
from pov_compiler.schemas import Anchor, DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="eval_chain_demo",
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
                Token(id="tok_0001", t0=2.0, t1=3.9, type="HIGHLIGHT", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=11.0, t1=13.0, type="HIGHLIGHT", conf=0.9, source_event="event_0002"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=12.0,
                t0=11.5,
                t1=12.5,
                source_event="event_0002",
                source_highlight="hl_0002",
                action={"type": "ATTENTION_TURN_HEAD"},
            )
        ],
    )


def test_evaluator_chain_metrics_columns() -> None:
    sample = NLQSample(
        qid="cnlq_000001",
        query="anchor=stop_look top_k=6 then token=HIGHLIGHT top_k=6",
        query_type="hard_pseudo_chain",
        gt_span=(11.0, 13.0),
        top_k=6,
        meta={"chain_meta": {"combo": "anchor_to_token", "step1_type": "hard_pseudo_anchor", "step2_type": "hard_pseudo_token"}},
    )
    results = evaluate_nlq_samples(
        output=_make_output(),
        samples=[sample],
        budgets={"max_total_s": [20], "max_tokens": [50], "max_decisions": [4]},
        sweep=False,
        allow_gt_fallback=False,
        variants=["full"],
    )
    per_query = list(results.get("per_query_rows", []))
    assert per_query
    row = dict(per_query[0])
    assert "chain_step1_has_hit" in row
    assert "chain_step2_has_hit" in row
    assert "chain_success" in row
    assert "chain_filtered_ratio_step2" in row

    by_type = list(results.get("by_query_type_rows", []))
    assert by_type
    trow = dict(by_type[0])
    assert "chain_success_rate" in trow
    assert "chain_step1_has_hit_rate" in trow
    assert "chain_step2_has_hit_rate" in trow

