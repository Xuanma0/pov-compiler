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
            id="repo_1",
            scale="event",
            t0=0.0,
            t1=8.0,
            text="event turn_head evidence and attention",
            importance=0.85,
            tags=["event", "turn_head"],
        ),
        RepoChunk(
            id="repo_2",
            scale="window",
            t0=8.0,
            t1=16.0,
            text="window with stop_look",
            importance=0.5,
            tags=["window", "stop_look"],
        ),
        RepoChunk(
            id="repo_3",
            scale="segment",
            t0=16.0,
            t1=30.0,
            text="later segment scene change",
            importance=0.65,
            tags=["segment", "scene_change"],
        ),
    ]


def test_repo_reader_budget_and_priority() -> None:
    chunks = _chunks()
    selected = select_chunks_for_query(
        chunks,
        query="anchor=turn_head top_k=6",
        budget={"max_repo_chunks": 2, "max_repo_chars": 200, "repo_strategy": "importance_greedy"},
        cfg={"strategy": "importance_greedy"},
    )
    assert 1 <= len(selected) <= 2
    assert selected[0].id == "repo_1"


def test_repo_reader_recency_strategy() -> None:
    chunks = _chunks()
    selected = select_chunks_for_query(
        chunks,
        query="",
        budget={"max_repo_chunks": 1, "repo_strategy": "recency_greedy"},
        cfg={"strategy": "recency_greedy"},
    )
    assert len(selected) == 1
    assert selected[0].id == "repo_3"
