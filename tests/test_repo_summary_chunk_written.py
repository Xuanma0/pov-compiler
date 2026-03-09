from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository.writer import build_repo_chunks
from pov_compiler.schemas import EventV1, Output


def test_repo_summary_chunk_written() -> None:
    output = Output(
        video_id="repo_summary_demo",
        meta={"duration_s": 30.0},
        events_v1=[
            EventV1(id="ev1", t0=0.0, t1=10.0, label="interaction-heavy", place_segment_id="p1", interaction_primary_object="door", interaction_score=0.8),
            EventV1(id="ev2", t0=10.0, t1=20.0, label="navigation", place_segment_id="p2", interaction_primary_object="", interaction_score=0.1),
        ],
    )
    chunks = build_repo_chunks(
        output,
        cfg={
            "write_policy": {"name": "multiscale+summary_v0", "chunk_step_s": 0.0, "summary_window_s": 60.0},
            "summary": {
                "enabled": True,
                "window_s": 60.0,
                "model": {"enabled": True, "provider": "fake", "model": "fake-summary-v0"},
            },
            "scales": {"event": True, "decision": False, "place": True, "window": False, "segment": False},
        },
    )
    summary = [c for c in chunks if str(c.level) == "summary"]
    assert summary
    payload = dict(summary[0].payload or {})
    assert "summary_v0" in payload
    assert payload["summary_v0"].get("video_id") == "repo_summary_demo"

