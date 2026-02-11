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


def test_distractor_aware_metrics_with_top1_false_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 120.0},
        events=[Event(id="event_0001", t0=0.0, t1=120.0, scores={}, anchors=[])],
        highlights=[
            KeyClip(
                id="hl_gt",
                t0=10.0,
                t1=12.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=11.0,
                conf=0.9,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_dist",
                t0=40.0,
                t1=42.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=41.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
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
            top_k=2,
            distractors=[(40.0, 42.0)],
        )
    ]

    def _fake_retrieve(self: Retriever, query: str) -> dict:
        return {
            "selected_events": ["event_0001"],
            "selected_highlights": ["hl_dist", "hl_gt"],  # top1 is distractor, top2 is GT
            "selected_decisions": [],
            "selected_tokens": [],
            "debug": {"reason": "fake_ranked"},
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

    row = result["per_query_rows"][0]
    assert float(row["hit_at_k"]) == 1.0
    assert float(row["hit_at_1"]) == 0.0
    assert float(row["top1_in_distractor"]) == 1.0
    assert float(row["hit_at_1_strict"]) == 0.0
    assert float(row["hit_at_k_strict"]) == 0.0

    by_type = result["by_query_type_rows"][0]
    assert float(by_type["hit_at_k"]) == 1.0
    assert float(by_type["hit_at_k_strict"]) == 0.0
    assert float(by_type["top1_in_distractor_rate"]) == 1.0
