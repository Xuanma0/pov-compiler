from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.perception.object_memory_v0 import build_object_memory_v0
from pov_compiler.schemas import EventV1


def test_object_memory_v0_last_seen_and_contact() -> None:
    perception = {
        "frames": [
            {
                "t": 0.0,
                "objects": [{"label": "door"}],
                "contact": {"active": None, "active_score": 0.0},
            },
            {
                "t": 1.0,
                "objects": [{"label": "door"}],
                "contact": {"active": {"label": "door", "score": 0.85}, "active_score": 0.85},
            },
        ]
    }
    events = [EventV1(id="ev1", t0=0.0, t1=2.0, place_segment_id="place_0001")]
    items = build_object_memory_v0(perception=perception, events_v1=events, contact_threshold=0.6)
    assert len(items) == 1
    item = items[0]
    assert item.object_name == "door"
    assert item.last_seen_t_ms == 1000
    assert item.last_contact_t_ms == 1000
    assert item.last_place_id == "place_0001"
    assert item.score > 0.0
