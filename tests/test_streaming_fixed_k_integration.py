from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, Output
from pov_compiler.streaming.budget_policy import BudgetSpec
from pov_compiler.streaming.runner import StreamingConfig, run_streaming


def _make_output() -> Output:
    return Output(
        video_id="stream_codec_demo",
        meta={"duration_s": 24.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=6.0, anchors=[Anchor(type="turn_head", t=2.0, conf=0.8)]),
            Event(id="event_0002", t0=6.0, t1=12.0, anchors=[Anchor(type="stop_look", t=8.0, conf=0.7)]),
            Event(id="event_0003", t0=12.0, t1=18.0, anchors=[Anchor(type="turn_head", t=14.0, conf=0.85)]),
            Event(id="event_0004", t0=18.0, t1=24.0, anchors=[Anchor(type="stop_look", t=20.0, conf=0.75)]),
        ],
    )


def test_streaming_runner_fixed_k_codec_columns_and_limit() -> None:
    payload = run_streaming(
        _make_output(),
        config=StreamingConfig(
            step_s=6.0,
            top_k=4,
            budgets=[BudgetSpec.parse("20/50/4")],
            budget_policy="fixed",
            fixed_budget="20/50/4",
            codec_name="fixed_k",
            codec_k=2,
            queries=["anchor=turn_head top_k=4"],
        ),
    )
    step_rows = list(payload.get("step_rows", []))
    query_rows = list(payload.get("query_rows", []))
    assert step_rows
    assert query_rows
    for row in step_rows:
        assert "items_written" in row
        assert "candidates_in_step" in row
        assert "codec_name" in row
        assert "codec_k" in row
        assert int(row.get("items_written", 0)) <= 2
    for row in query_rows:
        assert str(row.get("codec_name", "")) == "fixed_k"
        assert int(row.get("codec_k", 0)) == 2
