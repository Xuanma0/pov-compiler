from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import aggregate_interaction_signature


def test_interaction_signature_aggregation_burst_rate_object() -> None:
    frames = [
        {"t": 0.0, "contact": {"active": None}},
        {"t": 0.5, "contact": {"active": {"label": "cup", "object_id": "obj_cup", "score": 0.8}}},
        {"t": 1.0, "contact": {"active": {"label": "cup", "object_id": "obj_cup", "score": 0.7}}},
        {"t": 1.5, "contact": {"active": None}},
        {"t": 2.0, "contact": {"active": {"label": "phone", "object_id": "obj_phone", "score": 0.9}}},
        {"t": 2.5, "contact": {"active": None}},
    ]

    sig = aggregate_interaction_signature(frames, t0=0.0, t1=3.0)
    assert sig["contact_burst_count"] == 2
    assert abs(float(sig["contact_rate"]) - (3.0 / 6.0)) < 1e-9
    assert str(sig["active_object_top1"]) == "cup"
    assert float(sig["interaction_score"]) > 0.0
