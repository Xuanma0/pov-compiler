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
        RepoChunk(
            id="repo_001",
            level="event",
            scale="event",
            t0=0.0,
            t1=6.0,
            text="generic scene context",
            importance=0.95,
            tags=["event", "obj:cup"],
            meta={"primary_object": "cup"},
        ),
        RepoChunk(
            id="repo_002",
            level="decision",
            scale="decision",
            t0=6.0,
            t1=9.0,
            text="interaction with door handle",
            importance=0.55,
            tags=["decision", "obj:door"],
            meta={"primary_object": "door"},
        ),
        RepoChunk(
            id="repo_003",
            level="place",
            scale="place",
            t0=9.0,
            t1=12.0,
            text="place summary no object",
            importance=0.60,
            tags=["place", "place:place_0001"],
            meta={"place_segment_id": "place_0001"},
        ),
    ]


def test_chain_object_hint_hard_filters_and_scores() -> None:
    chunks = _chunks()
    selected, trace = select_chunks_for_query(
        chunks,
        query="token=SCENE_CHANGE top_k=6",
        budget={"max_repo_chunks": 2, "max_tokens": 256},
        cfg={"read_policy": {"name": "query_aware"}},
        query_info={"plan_intent": "token", "parsed_constraints": {"token_type": "SCENE_CHANGE"}, "top_k": 6},
        query_hints={
            "derived_constraints": {
                "object": {"enabled": True, "mode": "hard", "value": "door", "source": "step1_query"},
                "time": {"enabled": False, "mode": "off"},
                "place": {"enabled": False, "mode": "off"},
            }
        },
        return_trace=True,
    )
    assert selected
    assert all("door" in str(c.meta.get("primary_object", "")).lower() or "obj:door" in {str(t).lower() for t in c.tags} for c in selected)
    st = trace
    assert int(st.get("hint_filter_before", 0)) > int(st.get("hint_filter_after", 0))
    reasons = dict(st.get("hint_filtered_reason_counts", {}))
    assert int(reasons.get("filtered_by_chain_object", 0)) >= 1
    score_fields = dict(st.get("per_chunk_score_fields", {}))
    first_id = str(selected[0].id)
    assert first_id in score_fields
    assert float(score_fields[first_id].get("hint_object", 0.0)) > 0.0
    assert float(score_fields[first_id].get("hint_score", 0.0)) > 0.0
