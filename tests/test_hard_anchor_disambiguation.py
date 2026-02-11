from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import load_hard_pseudo_nlq
from pov_compiler.schemas import Event, KeyClip, Output, Token, TokenCodec


def test_hard_anchor_disambiguation_queries_and_ground_truth_alignment() -> None:
    turn_spans = [(10.0, 12.0), (30.0, 32.0), (60.0, 62.0), (90.0, 92.0)]
    highlights = []
    for i, (t0, t1) in enumerate(turn_spans, start=1):
        highlights.append(
            KeyClip(
                id=f"hl_turn_{i}",
                t0=t0,
                t1=t1,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=(t0 + t1) / 2.0,
                conf=0.8 + i * 0.01,
                meta={"anchor_types": ["turn_head"]},
            )
        )

    output = Output(
        video_id="demo",
        meta={"duration_s": 140.0},
        events=[Event(id="event_0001", t0=0.0, t1=140.0, scores={}, anchors=[])],
        highlights=highlights,
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(
                    id="tok_scene_1",
                    t0=25.0,
                    t1=26.0,
                    type="SCENE_CHANGE",
                    conf=0.9,
                    source_event="event_0001",
                )
            ],
        ),
        decision_points=[],
    )

    samples = load_hard_pseudo_nlq(
        output,
        seed=0,
        n_highlight=4,
        n_token=0,
        n_decision=0,
        top_k=6,
    )

    anchor_samples = [s for s in samples if s.query_type == "hard_pseudo_anchor"]
    assert anchor_samples
    assert any(
        ("第一次" in s.query) or ("最后一次" in s.query) or ("之后" in s.query) or ("first time" in s.query.lower())
        for s in anchor_samples
    )

    first_span = turn_spans[0]
    last_span = turn_spans[-1]
    scene_t = 25.5
    first_after_scene = next(span for span in turn_spans if (span[0] + span[1]) * 0.5 > scene_t)

    disamb_samples = [s for s in anchor_samples if str(s.meta.get("disambiguation", "none")) != "none"]
    assert disamb_samples
    for sample in disamb_samples:
        disamb = str(sample.meta.get("disambiguation"))
        if disamb == "first_occurrence":
            assert tuple(sample.gt_span) == first_span
        elif disamb == "last_occurrence":
            assert tuple(sample.gt_span) == last_span
        elif disamb == "after_scene_change":
            assert tuple(sample.gt_span) == first_after_scene
