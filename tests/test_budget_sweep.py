from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.budget_sweep import run_budget_sweep
from pov_compiler.schemas import Event, KeyClip, Output


def test_budget_sweep_monotonicity() -> None:
    output = Output(
        video_id="demo",
        meta={"duration_s": 120.0},
        events=[Event(id="event_0001", t0=0.0, t1=120.0, scores={}, anchors=[])],
        highlights=[
            KeyClip(id="hl_0001", t0=0.0, t1=18.0, source_event="event_0001", anchor_type="a", anchor_t=9.0, conf=0.9),
            KeyClip(id="hl_0002", t0=25.0, t1=43.0, source_event="event_0001", anchor_type="b", anchor_t=34.0, conf=0.8),
            KeyClip(id="hl_0003", t0=50.0, t1=68.0, source_event="event_0001", anchor_type="c", anchor_t=59.0, conf=0.7),
            KeyClip(id="hl_0004", t0=75.0, t1=93.0, source_event="event_0001", anchor_type="d", anchor_t=84.0, conf=0.6),
        ],
    )

    rows = run_budget_sweep(
        output=output,
        variant="highlights_only",
        budgets={"max_total_s": [20, 40, 60], "max_tokens": [100], "max_decisions": [10]},
        eval_config={"num_time_queries": 0},
        retriever_config={},
    )
    assert len(rows) == 3

    rows = sorted(rows, key=lambda r: float(r["budget_max_total_s"]))
    kept = [float(r["kept_duration_s"]) for r in rows]
    compression = [float(r["compression_ratio"]) for r in rows]

    assert kept[0] <= kept[1] <= kept[2]
    assert compression[0] >= compression[1] >= compression[2]
