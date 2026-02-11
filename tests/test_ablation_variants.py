from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.ablation import ALL_VARIANTS, apply_variant
from pov_compiler.schemas import DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _sample_output() -> Output:
    return Output(
        video_id="demo",
        meta={"duration_s": 10.0},
        events=[Event(id="event_0001", t0=0.0, t1=10.0, scores={}, anchors=[])],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=1.0,
                t1=3.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=2.0,
                conf=0.8,
                meta={},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[Token(id="tok_000001", t0=1.0, t1=2.0, type="HIGHLIGHT", conf=0.8, source_event="event_0001")],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_000001",
                t=2.0,
                t0=1.0,
                t1=3.0,
                source_event="event_0001",
                source_highlight="hl_0001",
                trigger={},
                state={},
                action={"type": "ATTENTION_TURN_HEAD"},
                constraints=[],
                outcome={"type": "SCENE_CHANGED"},
                alternatives=[],
                conf=0.7,
            )
        ],
    )


def test_ablation_variant_retention() -> None:
    base = _sample_output()
    for variant in ALL_VARIANTS:
        out = apply_variant(base, variant=variant)
        if variant == "raw_events_only":
            assert len(out.highlights) == 0
            assert len(out.token_codec.tokens) == 0
            assert len(out.decision_points) == 0
        elif variant == "highlights_only":
            assert len(out.highlights) == 1
            assert len(out.token_codec.tokens) == 0
            assert len(out.decision_points) == 0
        elif variant == "highlights_plus_tokens":
            assert len(out.highlights) == 1
            assert len(out.token_codec.tokens) == 1
            assert len(out.decision_points) == 0
        elif variant == "highlights_plus_decisions":
            assert len(out.highlights) == 1
            assert len(out.token_codec.tokens) == 0
            assert len(out.decision_points) == 1
        elif variant == "full":
            assert len(out.highlights) == 1
            assert len(out.token_codec.tokens) == 1
            assert len(out.decision_points) == 1
