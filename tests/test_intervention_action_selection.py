from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.budget_policy import BudgetSpec
from pov_compiler.streaming.interventions import (
    InterventionState,
    apply_intervention_action,
    choose_intervention_action,
)


def _budgets() -> list[BudgetSpec]:
    return [
        BudgetSpec.parse("20/50/4"),
        BudgetSpec.parse("60/200/12"),
        BudgetSpec.parse("120/400/24"),
    ]


def test_choose_action_deescalate_on_latency() -> None:
    budgets = _budgets()
    state = InterventionState(budget_idx=2)
    action, reason = choose_intervention_action(
        state=state,
        attribution="budget_insufficient",
        strict_hit=False,
        top1_in_distractor_rate=0.0,
        latency_e2e_ms=12.0,
        latency_cap_ms=5.0,
        trial_idx=1,
        max_trials=5,
        max_top1_in_distractor_rate=0.2,
        budgets=budgets,
    )
    assert action == "deescalate_budget"
    assert "latency" in reason


def test_choose_action_escalate_on_budget_insufficient() -> None:
    budgets = _budgets()
    state = InterventionState(budget_idx=0)
    action, _ = choose_intervention_action(
        state=state,
        attribution="budget_insufficient",
        strict_hit=False,
        top1_in_distractor_rate=0.0,
        latency_e2e_ms=1.0,
        latency_cap_ms=5.0,
        trial_idx=1,
        max_trials=5,
        max_top1_in_distractor_rate=0.2,
        budgets=budgets,
    )
    assert action == "escalate_budget"


def test_choose_action_accept_when_strict_and_risk_ok() -> None:
    action, _ = choose_intervention_action(
        state=InterventionState(budget_idx=0),
        attribution="",
        strict_hit=True,
        top1_in_distractor_rate=0.1,
        latency_e2e_ms=2.0,
        latency_cap_ms=5.0,
        trial_idx=1,
        max_trials=5,
        max_top1_in_distractor_rate=0.2,
        budgets=_budgets(),
    )
    assert action == "accept"


def test_choose_action_give_up_on_max_trials() -> None:
    action, _ = choose_intervention_action(
        state=InterventionState(budget_idx=0),
        attribution="evidence_missing",
        strict_hit=False,
        top1_in_distractor_rate=0.5,
        latency_e2e_ms=3.0,
        latency_cap_ms=5.0,
        trial_idx=5,
        max_trials=5,
        max_top1_in_distractor_rate=0.2,
        budgets=_budgets(),
    )
    assert action == "give_up_max_trials"


def test_apply_action_budget_boundaries() -> None:
    budgets = _budgets()
    low = apply_intervention_action(state=InterventionState(budget_idx=0), action="deescalate_budget", budgets=budgets)
    high = apply_intervention_action(state=InterventionState(budget_idx=2), action="escalate_budget", budgets=budgets)
    assert low.budget_idx == 0
    assert high.budget_idx == 2
