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
            id="repo_event_hi",
            level="event",
            scale="event",
            t0=0.0,
            t1=6.0,
            text="generic event summary",
            importance=0.95,
            source_ids=["event_0001"],
            tags=["event"],
        ),
        RepoChunk(
            id="repo_decision_lo",
            level="decision",
            scale="decision",
            t0=6.0,
            t1=8.0,
            text="decision turn head and reorient",
            importance=0.40,
            source_ids=["dp_0001", "event_0001"],
            tags=["decision", "action:attention_turn_head"],
        ),
        RepoChunk(
            id="repo_decision_hi2",
            level="decision",
            scale="decision",
            t0=8.0,
            t1=10.0,
            text="decision stop look pause",
            importance=0.65,
            source_ids=["dp_0002", "event_0002"],
            tags=["decision", "action:attention_stop_look"],
        ),
    ]


def test_query_aware_intent_boost_prefers_decision_level() -> None:
    chunks = _chunks()
    baseline = select_chunks_for_query(
        chunks,
        query="decision=ATTENTION_TURN_HEAD top_k=6",
        budget={"max_repo_chunks": 1, "max_tokens": 256},
        cfg={"read_policy": {"name": "budgeted_topk"}},
    )
    assert baseline[0].id == "repo_event_hi"

    selected, trace = select_chunks_for_query(
        chunks,
        query="decision=ATTENTION_TURN_HEAD top_k=6",
        budget={"max_repo_chunks": 1, "max_tokens": 256},
        cfg={"read_policy": {"name": "query_aware"}},
        query_info={"plan_intent": "decision", "parsed_constraints": {"decision_type": "ATTENTION_TURN_HEAD"}, "top_k": 6},
        return_trace=True,
    )
    assert selected
    assert str(selected[0].level).lower() == "decision"
    assert "per_chunk_score_fields" in trace
    first_id = trace["selected_chunk_ids"][0]
    fields = trace["per_chunk_score_fields"][first_id]
    assert float(fields["intent_boost"]) > 0.0


def test_query_aware_max_chunks_per_level_cap() -> None:
    chunks = _chunks()
    selected, trace = select_chunks_for_query(
        chunks,
        query="decision=ATTENTION_TURN_HEAD top_k=6",
        budget={"max_repo_chunks": 3, "max_tokens": 256},
        cfg={"read_policy": {"name": "query_aware", "max_chunks_per_level": {"decision": 1}}},
        query_info={"plan_intent": "decision", "parsed_constraints": {"decision_type": "ATTENTION_TURN_HEAD"}, "top_k": 6},
        return_trace=True,
    )
    decision_count = sum(1 for chunk in selected if str(chunk.level).lower() == "decision")
    assert decision_count <= 1
    assert isinstance(trace.get("dropped_topN", []), list)
    assert trace.get("selected_breakdown_by_level", {})
