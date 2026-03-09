from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.perception.object_memory_v0 import build_object_memory_v0
from pov_compiler.schemas import EventV1


def test_persistent_object_memory_local_smoke() -> None:
    perception = {
        "frames": [
            {
                "t": 0.0,
                "objects": [
                    {
                        "label": "cell_phone",
                        "track_id": "trk_1",
                        "persistent": True,
                        "persistence_count": 2,
                        "persistence_score": 0.7,
                        "mask_area": 1200.0,
                    },
                    {"label": "bowl", "track_id": "trk_2", "persistent": False, "persistence_count": 1},
                ],
                "contact": {"active": None, "active_score": 0.0},
            },
            {
                "t": 1.0,
                "objects": [
                    {
                        "label": "cell phone",
                        "track_id": "trk_1",
                        "persistent": True,
                        "persistence_count": 3,
                        "persistence_score": 0.9,
                        "mask_area": 1600.0,
                    },
                ],
                "contact": {"active": {"label": "cell_phone", "score": 0.8}, "active_score": 0.8},
            },
            {
                "t": 3.2,
                "objects": [
                    {
                        "label": "cell phone",
                        "track_id": "trk_1",
                        "persistent": True,
                        "persistence_count": 4,
                        "persistence_score": 1.0,
                        "mask_area": 1500.0,
                    },
                ],
                "contact": {"active": None, "active_score": 0.0},
            },
        ]
    }
    events = [EventV1(id="ev1", t0=0.0, t1=4.0, place_segment_id="place_0001")]

    baseline = build_object_memory_v0(
        perception=perception,
        events_v1=events,
        contact_threshold=0.6,
        logic_variant="persistence_v1",
        persistence_min_frames=2,
        persistence_score_min=0.5,
        enable_alias_merge=True,
    )
    uplift = build_object_memory_v0(
        perception=perception,
        events_v1=events,
        contact_threshold=0.6,
        logic_variant="persistent_v2",
        persistence_min_frames=2,
        persistence_score_min=0.5,
        enable_alias_merge=True,
    )

    assert len(baseline) == 2
    assert len(uplift) == 2
    baseline_phone = next(item for item in baseline if item.object_name == "cell phone")
    uplift_phone = next(item for item in uplift if item.object_name == "cell phone")
    assert baseline_phone.meta.get("reappearance_count", 0) == 0
    assert uplift_phone.meta.get("reappearance_count", 0) >= 1
    assert uplift_phone.meta.get("memory_tier") == "long_term"
    assert uplift_phone.meta.get("long_term_score", 0.0) > 0.0
    assert uplift_phone.meta.get("reappearance_score", 0.0) > 0.0
    assert uplift_phone.meta.get("last_interacted_t_ms") == 1000
    assert uplift_phone.meta.get("tracked_sightings_total", 0) >= 3
