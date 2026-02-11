from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.context.context_builder import build_context
from pov_compiler.schemas import Alternative, DecisionPoint, Output


def _dp(
    idx: int,
    t: float,
    conf: float,
    source_highlight: str | None,
    action_type: str = "ATTENTION_TURN_HEAD",
) -> DecisionPoint:
    return DecisionPoint(
        id=f"dp_{idx:06d}",
        t=t,
        t0=t - 0.5,
        t1=t + 0.5,
        source_event="event_0001",
        source_highlight=source_highlight,
        trigger={"anchor_type": "turn_head", "anchor_types": ["turn_head"], "token_ids": []},
        state={"evidence": {"token_ids": []}},
        action={"type": action_type, "conf": conf},
        constraints=[],
        outcome={"type": "SCENE_CHANGED", "conf": conf},
        alternatives=[
            Alternative(action_type="LOOK_FORWARD_ONLY", rationale="", expected_outcome="", conf=0.6),
            Alternative(action_type="TURN_HEAD_OPPOSITE", rationale="", expected_outcome="", conf=0.6),
        ],
        conf=conf,
        meta={},
    )


def test_context_decisions_budget_priority() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 30.0},
        events=[],
        highlights=[],
        decision_points=[
            _dp(1, 1.0, 0.90, "hl_0001"),
            _dp(2, 2.0, 0.95, None),
            _dp(3, 3.0, 0.80, "hl_0002"),
            _dp(4, 4.0, 0.75, "hl_0003"),
            _dp(5, 5.0, 0.99, None),
        ],
    )
    context = build_context(
        output_json=output,
        mode="decisions",
        budget={"max_decisions": 3, "max_tokens": 20, "decisions_min_gap_s": 0.0},
    )
    decisions = context["decision_points"]
    assert len(decisions) <= 3
    assert all(decision["source_highlight"] is not None for decision in decisions)
