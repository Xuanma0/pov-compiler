from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository.reader import select_chunks_for_query
from pov_compiler.repository.schema import RepoChunk


def _chunks() -> list[RepoChunk]:
    return [
        RepoChunk(id="a", level="event", t0=0.0, t1=12.0, text="a", importance=0.9),
        RepoChunk(id="b", level="event", t0=15.0, t1=25.0, text="b", importance=0.8),
        RepoChunk(id="c", level="event", t0=21.0, t1=30.0, text="c", importance=0.95),
        RepoChunk(id="d", level="event", t0=0.0, t1=9.0, text="d", importance=0.7),
    ]


def test_repo_chain_time_hard_uses_overlap_semantics() -> None:
    selected, trace = select_chunks_for_query(
        _chunks(),
        query="token=SCENE_CHANGE top_k=6",
        budget={"max_repo_chunks": 10, "max_tokens": 999},
        cfg={"read_policy": {"name": "query_aware"}},
        query_info={"plan_intent": "token", "parsed_constraints": {}, "top_k": 6},
        query_hints={
            "derived_constraints": {
                "time": {"enabled": True, "mode": "hard", "t_min_s": 10.0, "t_max_s": 20.0, "source": "step1_top1"},
                "place": {"enabled": False, "mode": "off"},
                "object": {"enabled": False, "mode": "off"},
            }
        },
        return_trace=True,
    )
    ids = {str(chunk.id) for chunk in selected}
    # overlap window [10,20]: keep A(0-12) and B(15-25), drop C(21-30), D(0-9)
    assert ids == {"a", "b"}
    assert str(trace.get("repo_time_filter_mode", "")) == "overlap"
    assert int(trace.get("hint_filter_before", 0)) == 4
    assert int(trace.get("hint_filter_after", 0)) == 2
    reasons = dict(trace.get("hint_filtered_reason_counts", {}))
    assert int(reasons.get("filtered_by_chain_time", 0)) == 2
    window_ms = dict(trace.get("repo_time_window_ms", {}))
    assert int(float(window_ms.get("t_min_ms", 0.0))) == 10000
    assert int(float(window_ms.get("t_max_ms", 0.0))) == 20000

