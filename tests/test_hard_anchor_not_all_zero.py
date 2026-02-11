from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import NLQSample
from pov_compiler.bench.nlq.evaluator import evaluate_nlq_samples
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Event, KeyClip, Output, TokenCodec


def test_hard_pseudo_anchor_not_all_zero_for_highlights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 120.0},
        events=[Event(id="event_0001", t0=0.0, t1=120.0, scores={}, anchors=[])],
        highlights=[
            KeyClip(
                id="hl_turn",
                t0=10.0,
                t1=12.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=11.0,
                conf=0.9,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_stop",
                t0=40.0,
                t1=42.0,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=41.0,
                conf=0.85,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(version="0.2", vocab=[], tokens=[]),
        decision_points=[],
    )
    samples = [
        NLQSample(
            qid="hnlq_000001",
            query="什么时候我开始左右张望？",
            query_type="hard_pseudo_anchor",
            gt_span=(10.0, 12.0),
            top_k=6,
            distractors=[(40.0, 42.0)],
        ),
        NLQSample(
            qid="hnlq_000002",
            query="什么时候我停下来观察周围？",
            query_type="hard_pseudo_anchor",
            gt_span=(40.0, 42.0),
            top_k=6,
            distractors=[(10.0, 12.0)],
        ),
    ]

    def _fake_retrieve(self: Retriever, query: str) -> dict:
        if "anchor=turn_head" in query:
            return {
                "selected_events": ["event_0001"],
                "selected_highlights": ["hl_turn"],
                "selected_decisions": [],
                "selected_tokens": [],
                "debug": {"reason": "fake_turn"},
            }
        if "anchor=stop_look" in query:
            return {
                "selected_events": ["event_0001"],
                "selected_highlights": ["hl_stop"],
                "selected_decisions": [],
                "selected_tokens": [],
                "debug": {"reason": "fake_stop"},
            }
        return {
            "selected_events": [],
            "selected_highlights": [],
            "selected_decisions": [],
            "selected_tokens": [],
            "debug": {"reason": "fake_empty"},
        }

    monkeypatch.setattr(Retriever, "retrieve", _fake_retrieve, raising=True)

    result = evaluate_nlq_samples(
        output=output,
        samples=samples,
        budgets={"max_total_s": [60], "max_tokens": [200], "max_decisions": [12]},
        sweep=False,
        retriever_config={},
        index_prefix=None,
        allow_gt_fallback=False,
        variants=["highlights_only"],
    )
    rows = result["by_query_type_rows"]
    target = [
        row
        for row in rows
        if row.get("variant") == "highlights_only" and row.get("query_type") == "hard_pseudo_anchor"
    ]
    assert target
    assert float(target[0]["hit_at_k"]) > 0.0
