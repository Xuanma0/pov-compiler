from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.l1_events.event_segmentation_v0 import segment_events_v0


def test_event_segmentation_v0_with_visual_and_contact_peaks() -> None:
    times = [float(i) for i in range(0, 30)]
    visual = [0.05 for _ in times]
    contact = [0.0 for _ in times]
    visual[8] = 1.0
    visual[20] = 0.9
    contact[9] = 0.8
    contact[21] = 0.85

    events, boundary = segment_events_v0(
        times=times,
        visual_change=visual,
        contact_score=contact,
        duration_s=29.0,
        thresh=0.45,
        min_event_s=4.0,
        visual_weight=0.6,
        contact_weight=0.4,
    )
    assert len(boundary) == len(times)
    assert len(events) >= 2
    assert float(events[0].t0) == 0.0
    for event in events:
        assert float(event.t1) > float(event.t0)
        assert str(event.meta.get("label", "")) in {"interaction-heavy", "navigation", "idle"}

