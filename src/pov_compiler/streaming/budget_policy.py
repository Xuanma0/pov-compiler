from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pov_compiler.streaming.interventions import policy_action_order

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
    action: str = ""
    action_reason: str = ""
    trial_records: list[dict[str, Any]] = field(default_factory=list)


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


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
            action="accept",
            action_reason=reason,
            trial_records=[
                {
                    "trial_index": 1,
                    "budget_key": budget.key,
                    "budget_seconds": budget.seconds,
                    "action": "accept",
                    "action_reason": reason,
                    "status": status,
                    "metrics": dict(metrics),
                }
            ],
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
            action="accept",
            action_reason=reason,
            trial_records=[
                {
                    "trial_index": 1,
                    "budget_key": budget.key,
                    "budget_seconds": budget.seconds,
                    "action": "accept",
                    "action_reason": reason,
                    "status": status,
                    "metrics": dict(metrics),
                }
            ],
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
            rhs = _to_float(cfg.get("value"))
            if rhs is None:
                continue
            lhs = _to_float(metrics.get(str(key)))
            if not self._eval_op(lhs, op, rhs):
                return False, f"gate_fail:{key}{op}{rhs}"
        for key, cfg in self.targets.items():
            if not isinstance(cfg, dict):
                continue
            op = str(cfg.get("op", ">="))
            rhs = _to_float(cfg.get("value"))
            if rhs is None:
                continue
            lhs = _to_float(metrics.get(str(key)))
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
        trial_records: list[dict[str, Any]] = []

        for budget in self.budgets_sorted:
            metrics = evaluate_budget(budget)
            tried.append(budget.key)
            passed, reason = self._passes(metrics)
            last_budget = budget
            last_metrics = metrics
            last_reason = reason
            trial_records.append(
                {
                    "trial_index": len(tried),
                    "budget_key": budget.key,
                    "budget_seconds": budget.seconds,
                    "action": "accept" if passed else "continue",
                    "action_reason": reason,
                    "status": "ok" if passed else "target_pending",
                    "metrics": dict(metrics),
                }
            )
            if passed:
                return BudgetSelection(
                    chosen_budget_key=budget.key,
                    chosen_budget_seconds=budget.seconds,
                    trials_count=len(tried),
                    tried_budget_keys=tried,
                    status="ok",
                    reason="adaptive_min_budget_passed",
                    metrics=metrics,
                    action="accept",
                    action_reason=reason,
                    trial_records=trial_records,
                )

        return BudgetSelection(
            chosen_budget_key=last_budget.key,
            chosen_budget_seconds=last_budget.seconds,
            trials_count=len(tried) if tried else 1,
            tried_budget_keys=tried if tried else [last_budget.key],
            status="no_budget_passed",
            reason=last_reason or "adaptive_no_budget_passed",
            metrics=last_metrics,
            action="give_up_max_trials",
            action_reason=last_reason or "adaptive_no_budget_passed",
            trial_records=trial_records,
        )


class SafetyLatencyBudgetPolicy:
    name = "safety_latency"

    def __init__(
        self,
        *,
        budgets: list[BudgetSpec],
        latency_cap_ms: float,
        max_trials_per_query: int = 3,
        prefer_lower_budget: bool = True,
        escalate_on_reasons: list[str] | None = None,
        deescalate_on_latency: bool = True,
        stop_on_non_budget_failure: bool = True,
        recommend_budget_key: str | None = None,
    ):
        self.budgets = sorted(list(budgets), key=lambda b: (float(b.max_total_s), int(b.max_tokens), int(b.max_decisions)))
        self.latency_cap_ms = float(latency_cap_ms)
        self.max_trials_per_query = max(1, int(max_trials_per_query))
        self.prefer_lower_budget = bool(prefer_lower_budget)
        reasons = [str(x).strip().lower() for x in (escalate_on_reasons or ["budget_insufficient"]) if str(x).strip()]
        self.escalate_on_reasons = reasons or ["budget_insufficient"]
        self.deescalate_on_latency = bool(deescalate_on_latency)
        self.stop_on_non_budget_failure = bool(stop_on_non_budget_failure)
        self.recommend_budget_key = str(recommend_budget_key).strip() if recommend_budget_key else ""

    def _start_index(self) -> int:
        if self.prefer_lower_budget:
            return 0
        if self.recommend_budget_key:
            for i, b in enumerate(self.budgets):
                if b.key == self.recommend_budget_key:
                    return i
        return max(0, int(len(self.budgets) // 2))

    def _latency_from_metrics(self, metrics: dict[str, Any]) -> float:
        for key in ("latency_e2e_ms", "query_e2e_ms", "e2e_ms", "latency_ms"):
            val = _to_float(metrics.get(key))
            if val is not None:
                return float(val)
        return 0.0

    def _strict_hit(self, metrics: dict[str, Any]) -> bool:
        for key in ("strict_hit_at_k", "hit_at_k_strict"):
            val = _to_float(metrics.get(key))
            if val is not None:
                return float(val) > 0.0
        return False

    def _safety_critical(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        is_critical = _to_float(metrics.get("safety_is_critical_fn"))
        if is_critical is None:
            # fallback: strict miss treated as critical
            is_critical = 0.0 if self._strict_hit(metrics) else 1.0
        reason = str(metrics.get("safety_reason", "")).strip().lower()
        return bool(float(is_critical) > 0.0), reason

    def select(
        self,
        *,
        budgets: list[BudgetSpec],
        evaluate_budget: Callable[[BudgetSpec], dict[str, Any]],
        query_context: dict[str, Any] | None = None,
    ) -> BudgetSelection:
        available = sorted(list(budgets), key=lambda b: (float(b.max_total_s), int(b.max_tokens), int(b.max_decisions)))
        if not available:
            raise ValueError("budgets is empty")
        # keep current available list instead of constructor if caller overrides
        self.budgets = available
        idx = self._start_index()
        tried: list[str] = []
        trial_records: list[dict[str, Any]] = []
        status = "ok"
        reason = "accept"
        final_action = "accept"
        final_action_reason = "strict_hit_latency_ok"
        chosen_budget = available[idx]
        chosen_metrics: dict[str, Any] = {}

        for trial_idx in range(1, self.max_trials_per_query + 1):
            budget = available[idx]
            metrics = evaluate_budget(budget)
            tried.append(budget.key)
            chosen_budget = budget
            chosen_metrics = dict(metrics)

            latency_ms = self._latency_from_metrics(metrics)
            strict_hit = self._strict_hit(metrics)
            safety_is, safety_reason = self._safety_critical(metrics)
            action = "accept"
            action_reason = "strict_hit_latency_ok"
            should_continue = False

            if latency_ms > self.latency_cap_ms and self.deescalate_on_latency:
                if idx > 0:
                    action = "deescalate_latency"
                    action_reason = f"latency_e2e_ms>{self.latency_cap_ms:.3f}"
                    idx -= 1
                    should_continue = True
                else:
                    action = "give_up_latency_floor"
                    action_reason = f"latency_e2e_ms>{self.latency_cap_ms:.3f} at_min_budget"
                    status = "rejected_by_latency"
                    reason = action_reason
            elif safety_is and safety_reason in self.escalate_on_reasons:
                if idx < len(available) - 1:
                    action = f"escalate_safety_{safety_reason or 'critical'}"
                    action_reason = "safety_critical_fn"
                    idx += 1
                    should_continue = True
                else:
                    action = "give_up_max_budget"
                    action_reason = "safety_critical_fn_at_max_budget"
                    status = "no_budget_passed"
                    reason = action_reason
            elif strict_hit and latency_ms <= self.latency_cap_ms:
                action = "accept"
                action_reason = "strict_hit_latency_ok"
                status = "ok"
                reason = action_reason
            elif (not strict_hit) and self.stop_on_non_budget_failure and (
                (not safety_is) or (safety_reason not in self.escalate_on_reasons)
            ):
                action = "stop_non_budget_failure"
                action_reason = f"safety_reason={safety_reason or 'unknown'}"
                status = "rejected_by_safety"
                reason = action_reason
            else:
                if (
                    not self.stop_on_non_budget_failure
                    and trial_idx < self.max_trials_per_query
                    and idx < len(available) - 1
                ):
                    action = "escalate_search"
                    action_reason = "strict_miss_search_next_budget"
                    idx += 1
                    should_continue = True
                else:
                    action = "give_up_max_trials"
                    action_reason = "max_trials_reached"
                    status = "no_budget_passed"
                    reason = action_reason

            trial_records.append(
                {
                    "trial_index": int(trial_idx),
                    "budget_key": budget.key,
                    "budget_seconds": budget.seconds,
                    "action": action,
                    "action_reason": action_reason,
                    "status": status if not should_continue else "continue",
                    "metrics": dict(metrics),
                }
            )
            final_action = action
            final_action_reason = action_reason

            if should_continue:
                continue
            break
        else:
            status = "no_budget_passed"
            reason = "max_trials_reached"
            final_action = "give_up_max_trials"
            final_action_reason = reason

        return BudgetSelection(
            chosen_budget_key=chosen_budget.key,
            chosen_budget_seconds=chosen_budget.seconds,
            trials_count=len(trial_records),
            tried_budget_keys=tried,
            status=status,
            reason=reason,
            metrics=chosen_metrics,
            action=final_action,
            action_reason=final_action_reason,
            trial_records=trial_records,
        )


class SafetyLatencyInterventionBudgetPolicy:
    name = "safety_latency_intervention"

    def __init__(
        self,
        *,
        budgets: list[BudgetSpec],
        latency_cap_ms: float,
        max_trials_per_query: int = 5,
        strict_threshold: float = 1.0,
        max_top1_in_distractor_rate: float = 0.2,
        prefer_lower_budget: bool = True,
    ):
        self.budgets = sorted(list(budgets), key=lambda b: (float(b.max_total_s), int(b.max_tokens), int(b.max_decisions)))
        self.latency_cap_ms = float(latency_cap_ms)
        self.max_trials_per_query = max(1, int(max_trials_per_query))
        self.strict_threshold = float(strict_threshold)
        self.max_top1_in_distractor_rate = float(max_top1_in_distractor_rate)
        self.prefer_lower_budget = bool(prefer_lower_budget)
        self.action_order = policy_action_order()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "latency_cap_ms": float(self.latency_cap_ms),
            "max_trials_per_query": int(self.max_trials_per_query),
            "strict_threshold": float(self.strict_threshold),
            "max_top1_in_distractor_rate": float(self.max_top1_in_distractor_rate),
            "prefer_lower_budget": bool(self.prefer_lower_budget),
            "budgets": [b.key for b in self.budgets],
            "action_order": self.action_order,
        }


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
