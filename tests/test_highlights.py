from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.l1_events.anchor_miner import suppress_event_anchors
from pov_compiler.memory.decision_sampling import build_highlights
from pov_compiler.schemas import Anchor, Event


def test_anchor_suppression() -> None:
    raw = [
        Anchor(type="stop_look", t=0.0, conf=0.95, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=0.4, conf=0.91, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=0.8, conf=0.90, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=3.0, conf=0.89, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=3.4, conf=0.88, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=3.8, conf=0.87, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=6.2, conf=0.86, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=8.4, conf=0.85, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=10.6, conf=0.84, meta={"duration_s": 0.6}),
        Anchor(type="stop_look", t=12.8, conf=0.83, meta={"duration_s": 0.6}),
    ]

    merged = suppress_event_anchors(
        anchors=raw,
        max_stop_look_per_event=3,
        min_gap_s=2.0,
        stop_look_min_conf=0.6,
    )
    stop_look = [a for a in merged if a.type == "stop_look"]
    assert len(stop_look) <= 3
    for prev, curr in zip(stop_look, stop_look[1:]):
        assert curr.t - prev.t >= 2.0 - 1e-6


def test_highlight_merge_and_budget() -> None:
    events = [
        Event(
            id="event_0001",
            t0=0.0,
            t1=15.0,
            scores={},
            anchors=[
                Anchor(type="turn_head", t=5.0, conf=0.9, meta={}),
                Anchor(type="stop_look", t=5.5, conf=0.85, meta={}),
                Anchor(type="stop_look", t=11.0, conf=0.95, meta={}),
            ],
        ),
        Event(
            id="event_0002",
            t0=15.0,
            t1=30.0,
            scores={},
            anchors=[Anchor(type="turn_head", t=20.0, conf=0.7, meta={})],
        ),
    ]

    highlights, stats = build_highlights(
        events=events,
        duration_s=30.0,
        pre_s=1.0,
        post_s=1.0,
        max_total_s=100.0,
        merge_gap_s=0.2,
        priority_map={"interaction": 3, "turn_head": 2, "stop_look": 1},
    )
    assert len(highlights) == 3
    assert stats["kept_duration_s"] > 0.0

    highlights_budget, stats_budget = build_highlights(
        events=events,
        duration_s=30.0,
        pre_s=1.0,
        post_s=1.0,
        max_total_s=2.2,
        merge_gap_s=0.2,
        priority_map={"interaction": 3, "turn_head": 2, "stop_look": 1},
    )
    assert len(highlights_budget) == 1
    assert highlights_budget[0].anchor_type == "turn_head"
    assert float(stats_budget["kept_duration_s"]) <= 2.2 + 1e-6
