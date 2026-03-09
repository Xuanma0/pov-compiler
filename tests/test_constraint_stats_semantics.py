from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.evaluator import _constraint_stats_from_rows


def test_constraint_stats_present_filtered_relaxed_semantics() -> None:
    rows = [
        {
            "present_after_scene_change": True,
            "filtered_after_scene_change": True,
            "relaxed_after_scene_change": False,
            "present_first_last": False,
            "filtered_first_last": False,
            "relaxed_first_last": False,
            "present_type_match": False,
            "filtered_type_match": False,
            "relaxed_type_match": False,
            "used_fallback": False,
            "filtered_hits_before": 10,
            "filtered_hits_after": 3,
        },
        {
            "present_after_scene_change": True,
            "filtered_after_scene_change": False,
            "relaxed_after_scene_change": True,
            "present_first_last": False,
            "filtered_first_last": False,
            "relaxed_first_last": False,
            "present_type_match": False,
            "filtered_type_match": False,
            "relaxed_type_match": False,
            "used_fallback": False,
            "filtered_hits_before": 2,
            "filtered_hits_after": 2,
        },
        {
            "present_after_scene_change": False,
            "filtered_after_scene_change": False,
            "relaxed_after_scene_change": False,
            "present_first_last": False,
            "filtered_first_last": False,
            "relaxed_first_last": False,
            "present_type_match": False,
            "filtered_type_match": False,
            "relaxed_type_match": False,
            "used_fallback": True,
            "filtered_hits_before": 0,
            "filtered_hits_after": 0,
        },
    ]

    stats = _constraint_stats_from_rows(rows)
    assert abs(float(stats["present_after_scene_change_rate"]) - (2.0 / 3.0)) < 1e-9
    assert abs(float(stats["filtered_after_scene_change_rate"]) - (1.0 / 3.0)) < 1e-9
    assert abs(float(stats["relaxed_after_scene_change_rate"]) - (1.0 / 3.0)) < 1e-9
    assert abs(float(stats["used_fallback_rate"]) - (1.0 / 3.0)) < 1e-9

