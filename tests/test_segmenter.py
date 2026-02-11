from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.l1_events.event_segmenter import segment_events


def test_segment_events_peak_split() -> None:
    times = np.arange(0, 20, dtype=np.float32)
    score = np.zeros_like(times)
    score[5] = 0.90
    score[9] = 0.85
    score[15] = 0.95

    events = segment_events(
        times=times,
        boundary_score=score,
        duration_s=20.0,
        thresh=0.65,
        min_event_s=3.0,
    )

    assert len(events) == 4
    assert events[0].t0 == 0.0 and events[0].t1 == 5.0
    assert events[1].t0 == 5.0 and events[1].t1 == 9.0
    assert events[2].t0 == 9.0 and events[2].t1 == 15.0
    assert events[3].t0 == 15.0 and events[3].t1 == 20.0


def test_segment_events_merge_short_tail() -> None:
    times = np.arange(0, 12, dtype=np.float32)
    score = np.zeros_like(times)
    score[4] = 0.90
    score[9] = 0.95  # tail duration would be 2s, should be merged away

    events = segment_events(
        times=times,
        boundary_score=score,
        duration_s=11.0,
        thresh=0.65,
        min_event_s=3.0,
    )

    assert len(events) == 2
    assert events[0].t0 == 0.0 and events[0].t1 == 4.0
    assert events[1].t0 == 4.0 and events[1].t1 == 11.0
