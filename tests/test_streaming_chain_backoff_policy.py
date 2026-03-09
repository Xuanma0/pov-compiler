from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.chain_backoff_policy import AdaptiveChainBackoff, FixedChainBackoff, LadderFirstNonZero


def test_fixed_chain_backoff_clamps_level() -> None:
    policy = FixedChainBackoff(level=7)
    out = policy.choose_level({"available_levels": [0, 1, 2, 3, 4]})
    assert out.level == 4
    assert "fixed_level" in out.reason


def test_ladder_first_nonzero_selects_first_hit_level() -> None:
    policy = LadderFirstNonZero()
    out = policy.choose_level(
        {
            "available_levels": [0, 1, 2, 3, 4],
            "level_metrics": {
                0: {"hit_count": 0},
                1: {"hit_count": 0},
                2: {"hit_count": 3},
                3: {"hit_count": 5},
            },
        }
    )
    assert out.level == 2
    assert "first_nonzero" in out.reason


def test_adaptive_chain_backoff_is_deterministic_and_prefers_better_level() -> None:
    policy = AdaptiveChainBackoff(alpha_latency=0.2, beta_safety=0.5, gamma_exhausted=1.0, delta_level=0.1, seed=0)
    context = {
        "available_levels": [0, 1, 2, 3, 4],
        "latency_cap_ms": 25.0,
        "max_trials_per_query": 4,
        "level_metrics": {
            0: {"hit_count": 0, "chain_success": 0.0, "hit_at_k_strict": 0.0, "top1_in_distractor_rate": 0.0, "safety_is_critical_fn": 1.0, "latency_e2e_ms": 1.0},
            1: {"hit_count": 3, "chain_success": 0.9, "hit_at_k_strict": 1.0, "top1_in_distractor_rate": 0.05, "safety_is_critical_fn": 0.0, "latency_e2e_ms": 5.0},
            2: {"hit_count": 4, "chain_success": 0.8, "hit_at_k_strict": 1.0, "top1_in_distractor_rate": 0.30, "safety_is_critical_fn": 1.0, "latency_e2e_ms": 6.0},
            3: {"hit_count": 4, "chain_success": 0.7, "hit_at_k_strict": 1.0, "top1_in_distractor_rate": 0.10, "safety_is_critical_fn": 0.0, "latency_e2e_ms": 30.0},
            4: {"hit_count": 2, "chain_success": 0.4, "hit_at_k_strict": 0.0, "top1_in_distractor_rate": 0.10, "safety_is_critical_fn": 0.0, "latency_e2e_ms": 8.0},
        },
    }
    out1 = policy.choose_level(context)
    out2 = policy.choose_level(context)
    assert out1.level == out2.level
    assert out1.debug.get("scores") == out2.debug.get("scores")
    assert out1.level == 1

