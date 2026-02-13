from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.reranker_config import WeightConfig


def test_weightconfig_out_of_range_is_clamped_with_warning() -> None:
    with pytest.warns(RuntimeWarning):
        cfg = WeightConfig.from_dict(
            {
                "name": "bad",
                "bonus_intent_token_on_token": 99.0,
                "penalty_before_scene_change": 999.0,
                "distractor_near_window_s": -5.0,
            }
        )
    assert float(cfg.bonus_intent_token_on_token) == 5.0
    assert float(cfg.penalty_before_scene_change) == 5.0
    assert float(cfg.distractor_near_window_s) == 0.0


def test_weightconfig_allow_out_of_range_keeps_values() -> None:
    cfg = WeightConfig.from_dict(
        {
            "name": "no_clamp",
            "bonus_intent_token_on_token": 99.0,
            "penalty_before_scene_change": 999.0,
        },
        allow_out_of_range=True,
    )
    assert float(cfg.bonus_intent_token_on_token) == 99.0
    assert float(cfg.penalty_before_scene_change) == 999.0
