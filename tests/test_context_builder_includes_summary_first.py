from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.context.context_builder import build_context
from pov_compiler.schemas import Output


def test_context_builder_puts_summary_first_when_repo_used() -> None:
    output = Output(video_id="ctx_repo_demo", meta={"duration_s": 20.0})
    output.repository = {
        "chunks": [
            {
                "id": "event_1",
                "chunk_id": "event_1",
                "level": "event",
                "scale": "event",
                "t0": 0.0,
                "t1": 10.0,
                "t0_ms": 0,
                "t1_ms": 10000,
                "text": "event details",
                "importance": 0.8,
                "tags": ["event"],
            },
            {
                "id": "summary_1",
                "chunk_id": "summary_1",
                "level": "summary",
                "scale": "summary",
                "t0": 0.0,
                "t1": 10.0,
                "t0_ms": 0,
                "t1_ms": 10000,
                "text": "summary details",
                "importance": 0.5,
                "tags": ["summary"],
            },
        ]
    }
    ctx = build_context(
        output,
        mode="repo_only",
        budget={
            "use_repo": True,
            "max_total_s": 20,
            "max_seconds": 20,
            "max_repo_chunks": 2,
            "max_repo_chars": 500,
            "max_repo_tokens": 200,
            "repo_read_policy": "budgeted_topk",
            "repo_strategy": "importance_greedy",
        },
    )
    repo_chunks = list(ctx.get("repo_chunks", []))
    assert repo_chunks
    assert str(repo_chunks[0].get("level", "")).lower() == "summary"
    trace = dict(ctx.get("repo_trace", {}))
    assert int(trace.get("repo_selected_summary_chunks_count", 0)) >= 1
    assert "repo_context_char_budget_used" in trace

