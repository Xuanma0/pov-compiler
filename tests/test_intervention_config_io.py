from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.intervention_config import InterventionConfig


def test_intervention_config_yaml_roundtrip_and_hash(tmp_path: Path) -> None:
    cfg = InterventionConfig(
        name="custom",
        w_safety=1.25,
        w_latency=0.3,
        w_trials=0.2,
        penalty_budget_up=0.4,
        penalty_retry=0.11,
        penalty_relax=0.18,
        max_trials_cap=6,
    )
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = InterventionConfig.from_yaml(path)

    assert loaded.to_dict() == cfg.to_dict()
    assert loaded.stable_hash() == cfg.stable_hash()

    # Stable hash for same content even when rewritten.
    path2 = tmp_path / "cfg2.yaml"
    loaded.to_yaml(path2)
    loaded2 = InterventionConfig.from_yaml(path2)
    assert loaded2.stable_hash() == cfg.stable_hash()
