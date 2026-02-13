from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.reranker_config import WeightConfig


def test_weight_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = WeightConfig(
        name="unit_test_cfg",
        bonus_intent_token_on_token=1.23,
        penalty_distractor_near=0.44,
        distractor_near_window_s=4.5,
        penalty_before_scene_change=0.9,
    )
    path = tmp_path / "cfg.yaml"
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        path.write_text("{\"name\":\"unit_test_cfg\",\"bonus_intent_token_on_token\":1.23}", encoding="utf-8")

    loaded = WeightConfig.from_yaml(path)
    assert loaded.name == "unit_test_cfg"
    assert abs(float(loaded.bonus_intent_token_on_token) - 1.23) < 1e-9
    assert abs(float(loaded.penalty_distractor_near) - float(cfg.penalty_distractor_near)) < 1e-9
    assert abs(float(loaded.penalty_before_scene_change) - float(cfg.penalty_before_scene_change)) < 1e-9
    assert isinstance(loaded.short_hash(), str) and len(loaded.short_hash()) == 10
