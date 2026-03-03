from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository.reader import select_chunks_for_query
from pov_compiler.repository.schema import RepoChunk


def test_repo_reader_prefers_summary_under_small_budget() -> None:
    chunks = [
        RepoChunk(
            id="detail_1",
            level="event",
            scale="event",
            t0=0.0,
            t1=10.0,
            text="detailed event text",
            importance=0.9,
            tags=["event"],
        ),
        RepoChunk(
            id="summary_1",
            level="summary",
            scale="summary",
            t0=0.0,
            t1=10.0,
            text="summary mentions door and place first",
            importance=0.45,
            tags=["summary", "obj:door", "place:first"],
        ),
    ]
    selected = select_chunks_for_query(
        chunks,
        query="lost_object=door top_k=6",
        budget={"max_total_s": 20, "max_repo_chunks": 1, "max_tokens": 200},
        cfg={"read_policy": {"name": "budgeted_topk", "max_chunks": 1, "max_tokens": 200}},
    )
    assert len(selected) == 1
    assert str(selected[0].level) == "summary"

