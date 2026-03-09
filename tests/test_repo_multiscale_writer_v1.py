from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository.writer import build_repo_chunks
from pov_compiler.schemas import Alternative, DecisionPoint, Event, EventV1, Output


def _make_output() -> Output:
    return Output(
        video_id="repo_v1_demo",
        meta={"duration_s": 24.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=8.0, scores={"boundary_conf": 0.7}, meta={"label": "navigation"}),
            Event(id="event_0002", t0=8.0, t1=16.0, scores={"boundary_conf": 0.8}, meta={"label": "interaction-heavy"}),
            Event(id="event_0003", t0=16.0, t1=24.0, scores={"boundary_conf": 0.6}, meta={"label": "idle"}),
        ],
        events_v1=[
            EventV1(
                id="ev1_1",
                t0=0.0,
                t1=8.0,
                label="navigation",
                place_segment_id="place_a",
                interaction_primary_object="",
                interaction_score=0.0,
                scores={"boundary_conf": 0.7},
            ),
            EventV1(
                id="ev1_2",
                t0=8.0,
                t1=16.0,
                label="interaction-heavy",
                place_segment_id="place_b",
                interaction_primary_object="door",
                interaction_score=0.6,
                scores={"boundary_conf": 0.8},
            ),
            EventV1(
                id="ev1_3",
                t0=16.0,
                t1=24.0,
                label="idle",
                place_segment_id="place_b",
                interaction_primary_object="door",
                interaction_score=0.4,
                scores={"boundary_conf": 0.6},
            ),
        ],
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=10.0,
                t0=9.8,
                t1=10.4,
                source_event="event_0002",
                action={"type": "ATTENTION_TURN_HEAD"},
                outcome={"type": "SCENE_CHANGED"},
                alternatives=[Alternative(action_type="LOOK_FORWARD_ONLY", rationale="x", expected_outcome="y", conf=0.5)],
                conf=0.8,
            )
        ],
    )


def test_repo_writer_v1_multiscale_levels() -> None:
    output = _make_output()
    chunks = build_repo_chunks(
        output,
        cfg={
            "write_policy": {"name": "fixed_interval", "chunk_step_s": 0.0},
            "scales": {"event": True, "decision": True, "place": True, "window": True, "segment": True},
            "window_s": 8.0,
            "min_segment_s": 4.0,
        },
    )
    assert chunks
    levels = {str(c.level) for c in chunks}
    assert {"event", "decision", "place"}.issubset(levels)
    for c in chunks:
        assert c.chunk_id
        assert c.t1_ms >= c.t0_ms
        assert 0.0 <= float(c.importance) <= 1.0
        assert isinstance(c.score_fields, dict)

