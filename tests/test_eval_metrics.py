from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.metrics import compute_consistency, compute_coverage
from pov_compiler.schemas import DecisionPoint, KeyClip, Output, Token, TokenCodec


def test_eval_metrics_coverage_and_consistency() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 10.0},
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=0.0,
                t1=2.0,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=1.0,
                conf=0.8,
            ),
            KeyClip(
                id="hl_0002",
                t0=1.0,
                t1=3.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=2.0,
                conf=0.9,
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_1", t0=0.0, t1=2.0, type="HIGHLIGHT", conf=0.9, source_event="event_0001"),
                Token(id="tok_2", t0=2.0, t1=4.0, type="ATTENTION_STOP_LOOK", conf=0.8, source_event="event_0001"),
                Token(id="tok_3", t0=5.0, t1=6.0, type="SCENE_CHANGE", conf=0.7, source_event="event_0001"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_1",
                t=1.5,
                t0=1.0,
                t1=2.0,
                source_event="event_0001",
                source_highlight="hl_0001",
                trigger={"token_ids": ["tok_1"]},
                state={"nearby_tokens": ["HIGHLIGHT"], "evidence": {"token_ids": ["tok_1"]}},
                action={"type": "ATTENTION_STOP_LOOK"},
                constraints=[],
                outcome={"type": "STILL_CONTINUE"},
                alternatives=[],
                conf=0.8,
            ),
            DecisionPoint(
                id="dp_2",
                t=5.5,
                t0=5.0,
                t1=6.0,
                source_event="event_0001",
                source_highlight=None,
                trigger={},
                state={"nearby_tokens": []},
                action={"type": "TRANSITION"},
                constraints=[],
                outcome={"type": "SCENE_CHANGED"},
                alternatives=[],
                conf=0.7,
            ),
        ],
    )

    coverage = compute_coverage(output)
    assert abs(coverage["coverage_ratio"] - 0.3) < 1e-6

    consistency = compute_consistency(output)
    # Token overlap: (2 + 1 + 0) / (2 + 2 + 1) = 0.6
    assert abs(consistency["token_in_highlight_ratio"] - 0.6) < 1e-6
