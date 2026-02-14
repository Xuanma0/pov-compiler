from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository.dedup import deduplicate_chunks
from pov_compiler.repository.schema import RepoChunk


def test_repo_dedup_text_hash_and_overlap() -> None:
    chunks = [
        RepoChunk(id="c1", scale="event", t0=0.0, t1=10.0, text="Turn head near door", importance=0.6, tags=["event"]),
        RepoChunk(id="c2", scale="event", t0=0.5, t1=10.2, text="turn head near door!!!", importance=0.9, tags=["event"]),
        RepoChunk(id="c3", scale="window", t0=20.0, t1=30.0, text="Another window chunk", importance=0.4, tags=["window"]),
    ]
    out = deduplicate_chunks(chunks, cfg={"iou_thresh": 0.5, "keep_best_importance": True})
    ids = {c.id for c in out}
    assert len(out) == 2
    assert "c2" in ids
    assert "c3" in ids
