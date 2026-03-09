from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.budget_policy import BudgetSpec, SafetyLatencyBudgetPolicy


def _budgets() -> list[BudgetSpec]:
    return [
        BudgetSpec.parse("20/50/4"),
        BudgetSpec.parse("40/100/8"),
        BudgetSpec.parse("60/200/12"),
    ]


def test_safety_latency_policy_deescalates_on_latency_cap() -> None:
    budgets = _budgets()
    policy = SafetyLatencyBudgetPolicy(
        budgets=budgets,
        latency_cap_ms=5.0,
        max_trials_per_query=3,
        prefer_lower_budget=False,
        recommend_budget_key="60/200/12",
    )

    def eval_budget(b: BudgetSpec) -> dict[str, float]:
        if b.key == "60/200/12":
            return {"hit_at_k_strict": 1.0, "latency_e2e_ms": 12.0, "safety_is_critical_fn": 0.0}
        return {"hit_at_k_strict": 1.0, "latency_e2e_ms": 3.0, "safety_is_critical_fn": 0.0}

    out = policy.select(budgets=budgets, evaluate_budget=eval_budget)
    assert out.chosen_budget_key == "40/100/8"
    assert out.trials_count == 2
    assert out.trial_records[0]["action"] == "deescalate_latency"
    assert out.trial_records[-1]["action"] == "accept"


def test_safety_latency_policy_escalates_on_budget_insufficient() -> None:
    budgets = _budgets()
    policy = SafetyLatencyBudgetPolicy(
        budgets=budgets,
        latency_cap_ms=20.0,
        max_trials_per_query=3,
        prefer_lower_budget=True,
        escalate_on_reasons=["budget_insufficient"],
    )

    def eval_budget(b: BudgetSpec) -> dict[str, float | str]:
        if b.key == "20/50/4":
            return {
                "hit_at_k_strict": 0.0,
                "latency_e2e_ms": 2.0,
                "safety_is_critical_fn": 1.0,
                "safety_reason": "budget_insufficient",
            }
        return {
            "hit_at_k_strict": 1.0,
            "latency_e2e_ms": 3.0,
            "safety_is_critical_fn": 0.0,
            "safety_reason": "",
        }

    out = policy.select(budgets=budgets, evaluate_budget=eval_budget)
    assert out.chosen_budget_key == "40/100/8"
    assert out.trials_count == 2
    assert out.trial_records[0]["action"].startswith("escalate_safety")
    assert out.trial_records[-1]["action"] == "accept"


def test_safety_latency_policy_accepts_on_strict_hit_latency_ok() -> None:
    budgets = _budgets()
    policy = SafetyLatencyBudgetPolicy(
        budgets=budgets,
        latency_cap_ms=20.0,
        max_trials_per_query=3,
        prefer_lower_budget=True,
    )
    out = policy.select(
        budgets=budgets,
        evaluate_budget=lambda b: {
            "hit_at_k_strict": 1.0,
            "latency_e2e_ms": 1.5,
            "safety_is_critical_fn": 0.0,
            "safety_reason": "",
        },
    )
    assert out.chosen_budget_key == "20/50/4"
    assert out.trials_count == 1
    assert out.action == "accept"


def test_safety_latency_policy_give_up_max_trials() -> None:
    budgets = _budgets()
    policy = SafetyLatencyBudgetPolicy(
        budgets=budgets,
        latency_cap_ms=50.0,
        max_trials_per_query=2,
        prefer_lower_budget=True,
        stop_on_non_budget_failure=False,
    )
    out = policy.select(
        budgets=budgets,
        evaluate_budget=lambda b: {
            "hit_at_k_strict": 0.0,
            "latency_e2e_ms": 3.0,
            "safety_is_critical_fn": 0.0,
            "safety_reason": "",
        },
    )
    assert out.trials_count == 2
    assert out.status == "no_budget_passed"
    assert out.action == "give_up_max_trials"
