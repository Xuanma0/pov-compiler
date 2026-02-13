from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.budget_policy import (
    AdaptiveMinBudgetPolicy,
    BudgetSpec,
    FixedBudgetPolicy,
    RecommendedBudgetPolicy,
)


def _budgets() -> list[BudgetSpec]:
    return [
        BudgetSpec.parse("20/50/4"),
        BudgetSpec.parse("40/100/8"),
        BudgetSpec.parse("60/200/12"),
    ]


def test_fixed_budget_policy_selects_configured_key() -> None:
    budgets = _budgets()
    policy = FixedBudgetPolicy("40/100/8")
    seen: list[str] = []

    def eval_budget(b: BudgetSpec) -> dict[str, float]:
        seen.append(b.key)
        return {"hit_at_k_strict": 1.0}

    out = policy.select(budgets=budgets, evaluate_budget=eval_budget)
    assert out.chosen_budget_key == "40/100/8"
    assert out.trials_count == 1
    assert seen == ["40/100/8"]


def test_recommended_budget_policy_reads_top1_summary(tmp_path: Path) -> None:
    rec_dir = tmp_path / "recommend"
    rec_dir.mkdir(parents=True)
    (rec_dir / "recommend_summary.json").write_text(
        json.dumps({"top1_budget_key": "20/50/4"}, ensure_ascii=False),
        encoding="utf-8",
    )
    policy = RecommendedBudgetPolicy(rec_dir)
    out = policy.select(
        budgets=_budgets(),
        evaluate_budget=lambda b: {"hit_at_k_strict": 1.0 if b.key == "20/50/4" else 0.0},
    )
    assert out.chosen_budget_key == "20/50/4"
    assert out.trials_count == 1


def test_adaptive_budget_policy_stops_on_first_pass() -> None:
    budgets = _budgets()
    policy = AdaptiveMinBudgetPolicy(
        budgets_sorted=budgets,
        gates={"top1_in_distractor_rate": {"op": "<=", "value": 0.2}},
        targets={"hit_at_k_strict": {"op": ">=", "value": 0.6}},
    )

    def eval_budget(b: BudgetSpec) -> dict[str, float]:
        if b.key == "20/50/4":
            return {"hit_at_k_strict": 0.0, "top1_in_distractor_rate": 0.0}
        if b.key == "40/100/8":
            return {"hit_at_k_strict": 1.0, "top1_in_distractor_rate": 0.0}
        return {"hit_at_k_strict": 1.0, "top1_in_distractor_rate": 0.0}

    out = policy.select(budgets=budgets, evaluate_budget=eval_budget)
    assert out.chosen_budget_key == "40/100/8"
    assert out.trials_count == 2
    assert out.status == "ok"


def test_adaptive_budget_policy_returns_max_when_none_pass() -> None:
    budgets = _budgets()
    policy = AdaptiveMinBudgetPolicy(
        budgets_sorted=budgets,
        gates={"top1_in_distractor_rate": {"op": "<=", "value": 0.2}},
        targets={"hit_at_k_strict": {"op": ">=", "value": 0.6}},
    )
    out = policy.select(
        budgets=budgets,
        evaluate_budget=lambda b: {"hit_at_k_strict": 0.0, "top1_in_distractor_rate": 1.0},
    )
    assert out.chosen_budget_key == "60/200/12"
    assert out.status == "no_budget_passed"
    assert out.trials_count == 3
