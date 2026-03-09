from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository.dedup import deduplicate_chunks
from pov_compiler.repository.schema import RepoChunk


def test_repo_dedup_cross_scale_keeps_dense_chunk_with_reason() -> None:
    chunks = [
        RepoChunk(
            id="event_1",
            chunk_id="event_1",
            scale="event",
            level="event",
            t0=10.0,
            t1=20.0,
            text="interaction with door after turn head",
            importance=0.55,
            tags=["event", "interaction-heavy", "obj:door"],
        ),
        RepoChunk(
            id="decision_1",
            chunk_id="decision_1",
            scale="decision",
            level="decision",
            t0=10.2,
            t1=19.6,
            text="interaction with door after turn head",
            importance=0.9,
            tags=["decision", "action:attention_turn_head", "obj:door"],
        ),
    ]
    out = deduplicate_chunks(
        chunks,
        cfg={"iou_thresh": 0.5, "sim_thresh": 0.8, "cross_scale": True, "keep_best_importance": True},
    )
    assert len(out) == 1
    assert out[0].id == "decision_1"
    reasons = list((out[0].meta or {}).get("dedup_kept_reasons", []))
    assert reasons
    assert any("keep=decision" in str(x) or "sim_replaced" in str(x) for x in reasons)

