from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BudgetSpec:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        return f"{int(round(float(self.max_total_s)))}/{int(self.max_tokens)}/{int(self.max_decisions)}"

    @property
    def seconds(self) -> float:
        return float(self.max_total_s)

    @classmethod
    def parse(cls, key: str) -> "BudgetSpec":
        text = str(key).strip()
        parts = [x.strip() for x in text.split("/") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"invalid budget key: {key}")
        return cls(max_total_s=float(parts[0]), max_tokens=int(parts[1]), max_decisions=int(parts[2]))


@dataclass
class BudgetSelection:
    chosen_budget_key: str
    chosen_budget_seconds: float
    trials_count: int
    tried_budget_keys: list[str] = field(default_factory=list)
    status: str = "ok"
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class FixedBudgetPolicy:
    name = "fixed"

    def __init__(self, budget_key: str):
        self.budget_key = str(budget_key)

    def select(
        self,
        *,
        budgets: list[BudgetSpec],
        evaluate_budget: Callable[[BudgetSpec], dict[str, Any]],
        query_context: dict[str, Any] | None = None,
    ) -> BudgetSelection:
        budget_map = {b.key: b for b in budgets}
        budget = budget_map.get(self.budget_key)
        status = "ok"
        reason = "fixed_budget"
        if budget is None:
            budget = budgets[-1]
            status = "fallback_budget"
            reason = f"fixed budget {self.budget_key} not found, fallback to {budget.key}"
        metrics = evaluate_budget(budget)
        return BudgetSelection(
            chosen_budget_key=budget.key,
            chosen_budget_seconds=budget.seconds,
            trials_count=1,
            tried_budget_keys=[budget.key],
            status=status,
            reason=reason,
            metrics=metrics,
        )


class RecommendedBudgetPolicy:
    name = "recommend"

    def __init__(self, recommend_source: str | Path):
        self.recommend_source = Path(recommend_source)
        self._top1_budget_key = self._load_top1_budget_key(self.recommend_source)

    @staticmethod
    def _load_top1_budget_key(source: Path) -> str | None:
        if source.is_file():
            if source.name.lower().endswith(".csv"):
                return RecommendedBudgetPolicy._top1_from_table(source)
            if source.name.lower().endswith(".json"):
                return RecommendedBudgetPolicy._top1_from_summary(source)
        if source.is_dir():
            summary = source / "recommend_summary.json"
            if summary.exists():
                top1 = RecommendedBudgetPolicy._top1_from_summary(summary)
                if top1:
                    return top1
            table = source / "tables" / "table_budget_recommend.csv"
            if table.exists():
                return RecommendedBudgetPolicy._top1_from_table(table)
        return None

    @staticmethod
    def _top1_from_summary(path: Path) -> str | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        top1 = payload.get("top1_budget_key")
        if isinstance(top1, str) and top1.strip():
            return top1.strip()
        return None

    @staticmethod
    def _top1_from_table(path: Path) -> str | None:
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return None
        accepted = [r for r in rows if str(r.get("status", "")).lower() == "accepted"]
        accepted.sort(key=lambda r: float(r.get("rank", 1e9)) if str(r.get("rank", "")).strip() else 1e9)
        if accepted:
            key = str(accepted[0].get("budget_key", "")).strip()
            return key or None
        if rows:
            key = str(rows[0].get("budget_key", "")).strip()
            return key or None
        return None

    def select(
        self,
        *,
        budgets: list[BudgetSpec],
        evaluate_budget: Callable[[BudgetSpec], dict[str, Any]],
        query_context: dict[str, Any] | None = None,
    ) -> BudgetSelection:
        budget_map = {b.key: b for b in budgets}
        status = "ok"
        reason = "recommend_top1"
        budget = budget_map.get(str(self._top1_budget_key or ""))
        if budget is None:
            budget = budgets[-1]
            status = "fallback_budget"
            reason = f"recommendation missing, fallback to {budget.key}"
        metrics = evaluate_budget(budget)
        return BudgetSelection(
            chosen_budget_key=budget.key,
            chosen_budget_seconds=budget.seconds,
            trials_count=1,
            tried_budget_keys=[budget.key],
            status=status,
            reason=reason,
            metrics=metrics,
        )


class AdaptiveMinBudgetPolicy:
    name = "adaptive"

    def __init__(
        self,
        *,
        budgets_sorted: list[BudgetSpec],
        gates: dict[str, Any] | None = None,
        targets: dict[str, Any] | None = None,
        allow_missing: bool = False,
    ):
        self.budgets_sorted = list(budgets_sorted)
        self.gates = dict(gates or {})
        self.targets = dict(targets or {})
        self.allow_missing = bool(allow_missing)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        if out != out:
            return None
        return out

    def _eval_op(self, lhs: float | None, op: str, rhs: float) -> bool:
        if lhs is None:
            return bool(self.allow_missing)
        if op == "<=":
            return float(lhs) <= float(rhs)
        if op == "<":
            return float(lhs) < float(rhs)
        if op == ">=":
            return float(lhs) >= float(rhs)
        if op == ">":
            return float(lhs) > float(rhs)
        if op == "==":
            return float(lhs) == float(rhs)
        return False

    def _passes(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        for key, cfg in self.gates.items():
            if not isinstance(cfg, dict):
                continue
            op = str(cfg.get("op", "<="))
            rhs = self._to_float(cfg.get("value"))
            if rhs is None:
                continue
            lhs = self._to_float(metrics.get(str(key)))
            if not self._eval_op(lhs, op, rhs):
                return False, f"gate_fail:{key}{op}{rhs}"
        for key, cfg in self.targets.items():
            if not isinstance(cfg, dict):
                continue
            op = str(cfg.get("op", ">="))
            rhs = self._to_float(cfg.get("value"))
            if rhs is None:
                continue
            lhs = self._to_float(metrics.get(str(key)))
            if not self._eval_op(lhs, op, rhs):
                return False, f"target_fail:{key}{op}{rhs}"
        return True, "passed"

    def select(
        self,
        *,
        budgets: list[BudgetSpec],
        evaluate_budget: Callable[[BudgetSpec], dict[str, Any]],
        query_context: dict[str, Any] | None = None,
    ) -> BudgetSelection:
        tried: list[str] = []
        last_budget = budgets[-1]
        last_metrics: dict[str, Any] = {}
        last_reason = ""

        for budget in self.budgets_sorted:
            metrics = evaluate_budget(budget)
            tried.append(budget.key)
            passed, reason = self._passes(metrics)
            last_budget = budget
            last_metrics = metrics
            last_reason = reason
            if passed:
                return BudgetSelection(
                    chosen_budget_key=budget.key,
                    chosen_budget_seconds=budget.seconds,
                    trials_count=len(tried),
                    tried_budget_keys=tried,
                    status="ok",
                    reason="adaptive_min_budget_passed",
                    metrics=metrics,
                )

        return BudgetSelection(
            chosen_budget_key=last_budget.key,
            chosen_budget_seconds=last_budget.seconds,
            trials_count=len(tried) if tried else 1,
            tried_budget_keys=tried if tried else [last_budget.key],
            status="no_budget_passed",
            reason=last_reason or "adaptive_no_budget_passed",
            metrics=last_metrics,
        )


def parse_budget_keys(raw: str) -> list[BudgetSpec]:
    text = str(raw).strip()
    if not text:
        return []
    out: list[BudgetSpec] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        out.append(BudgetSpec.parse(token))
    out.sort(key=lambda b: (float(b.max_total_s), int(b.max_tokens), int(b.max_decisions)))
    return out
