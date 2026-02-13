from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.perception.contact import contact_score, select_active_contact


def test_contact_score_and_active_object_selection() -> None:
    hand = {
        "id": "hand_0",
        "bbox": [90.0, 90.0, 140.0, 140.0],
        "landmarks": [[95.0, 95.0], [100.0, 100.0], [105.0, 105.0], [110.0, 110.0], [115.0, 115.0]],
    }
    obj_near = {"id": "obj_a", "label": "cup", "bbox": [100.0, 100.0, 150.0, 150.0]}
    obj_far = {"id": "obj_b", "label": "book", "bbox": [260.0, 260.0, 320.0, 320.0]}

    s_near = contact_score(hand, obj_near, frame_diag=500.0)
    s_far = contact_score(hand, obj_far, frame_diag=500.0)
    assert s_near > s_far
    assert s_near > 0.2

    result = select_active_contact(
        hands=[hand],
        objects=[obj_far, obj_near],
        frame_shape=(360, 640, 3),
        t=1.23,
        min_score=0.1,
    )
    active = result.get("active")
    assert isinstance(active, dict)
    assert str(active.get("object_id")) == "obj_a"

