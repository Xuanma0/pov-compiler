from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.retrieval.trace import trace_query
from pov_compiler.schemas import Anchor, Event, Output


def _output() -> Output:
    out = Output(
        video_id="trace_repo_demo",
        meta={"duration_s": 12.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=6.0, anchors=[Anchor(type="turn_head", t=2.0, conf=0.8)], meta={"label": "navigation"}),
            Event(id="event_0002", t0=6.0, t1=12.0, anchors=[Anchor(type="stop_look", t=8.0, conf=0.75)], meta={"label": "interaction-heavy"}),
        ],
    )
    return ensure_events_v1(out)


def test_trace_query_contains_repo_selection_when_enabled() -> None:
    trace = trace_query(
        output_json=_output(),
        query="anchor=turn_head top_k=6",
        top_k=6,
        use_repo=True,
    )
    assert "repo_selection" in trace
    repo = trace["repo_selection"]
    assert repo.get("enabled") is True
    assert "trace" in repo
    assert "selection_trace" in repo.get("trace", {})
