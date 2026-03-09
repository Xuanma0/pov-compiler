from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import load_hard_pseudo_nlq
from pov_compiler.bench.nlq.evaluator import evaluate_nlq_samples
from pov_compiler.schemas import Event, KeyClip, Output, Token, TokenCodec


def _build_output() -> Output:
    highlights: list[KeyClip] = []
    # Dense same-type anchors around/after scene-change for satisfiable after-scene queries.
    turn_spans = [
        (10.0, 12.0),
        (20.0, 22.0),
        (35.0, 37.0),
        (50.0, 52.0),
        (65.0, 67.0),
        (80.0, 82.0),
        (95.0, 97.0),
        (110.0, 112.0),
    ]
    stop_spans = [
        (15.0, 17.0),
        (28.0, 30.0),
        (42.0, 44.0),
        (58.0, 60.0),
        (73.0, 75.0),
        (88.0, 90.0),
        (103.0, 105.0),
        (118.0, 120.0),
    ]
    i = 1
    for t0, t1 in turn_spans:
        highlights.append(
            KeyClip(
                id=f"hl_turn_{i:03d}",
                t0=t0,
                t1=t1,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=(t0 + t1) / 2.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            )
        )
        i += 1
    for t0, t1 in stop_spans:
        highlights.append(
            KeyClip(
                id=f"hl_stop_{i:03d}",
                t0=t0,
                t1=t1,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=(t0 + t1) / 2.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            )
        )
        i += 1

    return Output(
        video_id="demo_after_scene",
        meta={"duration_s": 140.0},
        events=[Event(id="event_0001", t0=0.0, t1=140.0, scores={}, anchors=[])],
        highlights=highlights,
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_scene_1", t0=40.0, t1=41.0, type="SCENE_CHANGE", conf=0.9, source_event="event_0001")
            ],
        ),
        decision_points=[],
    )


def test_after_scene_change_queries_have_nontrivial_success() -> None:
    output = _build_output()
    samples = load_hard_pseudo_nlq(
        output,
        seed=0,
        n_highlight=50,
        n_token=0,
        n_decision=0,
        top_k=6,
    )
    assert samples
    anchor_samples = [s for s in samples if s.query_type == "hard_pseudo_anchor"]
    assert anchor_samples

    result = evaluate_nlq_samples(
        output=output,
        samples=anchor_samples,
        budgets={"max_total_s": [60.0], "max_tokens": [200], "max_decisions": [12]},
        sweep=False,
        allow_gt_fallback=False,
        variants=["highlights_only"],
    )
    rows = result["per_query_rows"]
    assert rows
    present_rows = [row for row in rows if bool(row.get("present_after_scene_change", False))]
    assert len(present_rows) > 0
    relaxed_rows = [row for row in present_rows if bool(row.get("relaxed_after_scene_change", False))]
    relaxed_rate = float(len(relaxed_rows) / len(present_rows))
    assert relaxed_rate < 0.8

