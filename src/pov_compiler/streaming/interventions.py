from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pov_compiler.bench.nlq.safety import classify_failure_reason

ACTION_ORDER_BY_ATTRIBUTION: dict[str, list[str]] = {
    "retrieval_distractor": ["switch_rerank_cfg", "expand_candidates", "escalate_budget"],
    "constraints_over_filtered": ["relax_constraints", "expand_candidates", "escalate_budget"],
    "evidence_missing": ["expand_candidates", "widen_evidence_window", "escalate_budget"],
    "budget_insufficient": ["escalate_budget"],
    "other": ["expand_candidates", "escalate_budget"],
}

STOP_ACTIONS = {"accept", "give_up_max_trials", "stop_non_budget_failure", "stop_at_budget_boundary"}


@dataclass
class InterventionState:
    budget_idx: int
    constraints_stage: int = 0
    rerank_profile: str = "default"
    expand_level: int = 0
    widen_evidence_window: bool = False
    action_cursor: dict[str, int] = field(default_factory=dict)
    action_history: list[str] = field(default_factory=list)

    def to_config(self) -> dict[str, Any]:
        return {
            "budget_idx": int(self.budget_idx),
            "constraints_stage": int(self.constraints_stage),
            "rerank_profile": str(self.rerank_profile),
            "expand_level": int(self.expand_level),
            "widen_evidence_window": bool(self.widen_evidence_window),
        }


def infer_failure_attribution(metrics: dict[str, Any], budget: Any) -> str:
    strict_hit = float(metrics.get("hit_at_k_strict", 0.0) or 0.0) > 0.0
    if strict_hit:
        return ""
    row = {
        "filtered_hits_before": int(metrics.get("filtered_hits_before", 0) or 0),
        "filtered_hits_after": int(metrics.get("filtered_hits_after", 0) or 0),
        "candidate_count": int(metrics.get("filtered_hits_after", 0) or 0),
        "top1_in_distractor": float(metrics.get("top1_in_distractor_rate", 0.0) or 0.0),
        "budget_max_total_s": float(getattr(budget, "max_total_s", 0.0)),
        "budget_max_tokens": int(getattr(budget, "max_tokens", 0)),
        "budget_max_decisions": int(getattr(budget, "max_decisions", 0)),
    }
    reason = str(classify_failure_reason(row)).strip().lower()
    return reason if reason else "other"


def choose_intervention_action(
    *,
    state: InterventionState,
    attribution: str,
    strict_hit: bool,
    top1_in_distractor_rate: float,
    latency_e2e_ms: float,
    latency_cap_ms: float,
    trial_idx: int,
    max_trials: int,
    max_top1_in_distractor_rate: float,
    budgets: list[Any],
) -> tuple[str, str]:
    if strict_hit and top1_in_distractor_rate <= float(max_top1_in_distractor_rate) and latency_e2e_ms <= float(latency_cap_ms):
        return "accept", "strict_and_risk_ok"
    if trial_idx >= int(max_trials):
        return "give_up_max_trials", "max_trials_reached"
    if latency_e2e_ms > float(latency_cap_ms):
        if int(state.budget_idx) > 0:
            return "deescalate_budget", f"latency_e2e_ms>{float(latency_cap_ms):.3f}"
        return "stop_at_budget_boundary", "latency_high_at_min_budget"

    key = attribution if attribution in ACTION_ORDER_BY_ATTRIBUTION else "other"
    seq = ACTION_ORDER_BY_ATTRIBUTION.get(key, ACTION_ORDER_BY_ATTRIBUTION["other"])
    cur = int(state.action_cursor.get(key, 0))
    while cur < len(seq):
        action = str(seq[cur])
        state.action_cursor[key] = cur + 1
        cur += 1
        if action == "escalate_budget" and int(state.budget_idx) >= len(budgets) - 1:
            continue
        return action, f"attribution={key}"
    return "stop_non_budget_failure", f"no_more_actions_for={key}"


def apply_intervention_action(
    *,
    state: InterventionState,
    action: str,
    budgets: list[Any],
) -> InterventionState:
    updated = InterventionState(
        budget_idx=int(state.budget_idx),
        constraints_stage=int(state.constraints_stage),
        rerank_profile=str(state.rerank_profile),
        expand_level=int(state.expand_level),
        widen_evidence_window=bool(state.widen_evidence_window),
        action_cursor=dict(state.action_cursor),
        action_history=list(state.action_history),
    )
    act = str(action)
    updated.action_history.append(act)
    if act == "relax_constraints":
        updated.constraints_stage = min(3, int(updated.constraints_stage) + 1)
    elif act == "switch_rerank_cfg":
        updated.rerank_profile = "anti_distractor" if updated.rerank_profile == "default" else "default"
    elif act == "expand_candidates":
        updated.expand_level = min(3, int(updated.expand_level) + 1)
    elif act == "widen_evidence_window":
        updated.widen_evidence_window = True
    elif act == "escalate_budget":
        updated.budget_idx = min(len(budgets) - 1, int(updated.budget_idx) + 1)
    elif act == "deescalate_budget":
        updated.budget_idx = max(0, int(updated.budget_idx) - 1)
    return updated


def policy_action_order() -> dict[str, list[str]]:
    return {k: list(v) for k, v in ACTION_ORDER_BY_ATTRIBUTION.items()}
