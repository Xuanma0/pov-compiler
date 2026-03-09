from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import EventV1
from pov_compiler.streaming.codec import FixedKStreamingCodec


def test_fixed_k_codec_selects_topk_by_score() -> None:
    events = [
        EventV1(id="e1", t0=0.0, t1=1.0, scores={"boundary_conf": 0.2}, interaction_score=0.1, place_segment_id="p1"),
        EventV1(id="e2", t0=1.0, t1=2.0, scores={"boundary_conf": 0.9}, interaction_score=0.8, place_segment_id="p1"),
        EventV1(id="e3", t0=2.0, t1=3.0, scores={"boundary_conf": 0.7}, interaction_score=0.6, place_segment_id="p2"),
    ]
    codec = FixedKStreamingCodec(k=2)
    items = codec.encode_step(events, step_meta={"seen_place_segments": []})
    assert len(items) == 2
    ids = [x.source_id for x in items]
    assert "e2" in ids
    assert all(item.score >= 0.0 for item in items)


def test_fixed_k_codec_respects_k_limit() -> None:
    events = [EventV1(id=f"e{i}", t0=float(i), t1=float(i + 1), interaction_score=float(i) / 10.0) for i in range(10)]
    codec = FixedKStreamingCodec(k=3)
    items = codec.encode_step(events, step_meta={})
    assert len(items) <= 3
