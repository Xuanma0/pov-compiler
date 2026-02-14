from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pov_compiler.bench.nlq.safety import classify_failure_reason
from pov_compiler.streaming.intervention_config import InterventionConfig, resolve_intervention_config

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
    intervention_cfg: InterventionConfig | dict[str, Any] | str | None = None,
) -> tuple[str, str]:
    cfg = resolve_intervention_config(intervention_cfg)
    max_trials_eff = max(1, min(int(max_trials), int(cfg.max_trials_cap)))
    if strict_hit and top1_in_distractor_rate <= float(max_top1_in_distractor_rate) and latency_e2e_ms <= float(latency_cap_ms):
        return "accept", "strict_and_risk_ok"
    if trial_idx >= max_trials_eff:
        return "give_up_max_trials", "max_trials_reached"
    if latency_e2e_ms > float(latency_cap_ms):
        if int(state.budget_idx) > 0:
            return "deescalate_budget", f"latency_e2e_ms>{float(latency_cap_ms):.3f}"
        return "stop_at_budget_boundary", "latency_high_at_min_budget"

    key = attribution if attribution in ACTION_ORDER_BY_ATTRIBUTION else "other"
    seq = list(ACTION_ORDER_BY_ATTRIBUTION.get(key, ACTION_ORDER_BY_ATTRIBUTION["other"]))
    # Score candidate interventions with configurable weights/penalties.
    # Higher score is preferred; actions already attempted for this attribution are skipped.
    def _score(action: str) -> float:
        base = 0.0
        if key == "retrieval_distractor":
            base += {"switch_rerank_cfg": 1.0, "expand_candidates": 0.8, "escalate_budget": 0.4}.get(action, 0.0)
        elif key == "constraints_over_filtered":
            base += {"relax_constraints": 1.0, "expand_candidates": 0.7, "escalate_budget": 0.3}.get(action, 0.0)
        elif key == "evidence_missing":
            base += {"expand_candidates": 1.0, "widen_evidence_window": 0.9, "escalate_budget": 0.35}.get(action, 0.0)
        elif key == "budget_insufficient":
            base += {"escalate_budget": 1.0}.get(action, 0.0)
        else:
            base += {"expand_candidates": 0.7, "escalate_budget": 0.4}.get(action, 0.0)

        # Shared objective terms.
        safety_need = max(0.0, float(cfg.w_safety)) * (1.0 - (1.0 if strict_hit else 0.0))
        latency_pressure = max(0.0, float(cfg.w_latency)) * max(0.0, (float(latency_e2e_ms) - float(latency_cap_ms)) / max(1.0, float(latency_cap_ms)))
        trial_pressure = max(0.0, float(cfg.w_trials)) * (float(trial_idx) / max(1.0, float(max_trials_eff)))
        base += safety_need
        base -= trial_pressure
        if action == "deescalate_budget":
            base += latency_pressure + float(cfg.w_latency)
        if action == "escalate_budget":
            base -= float(cfg.penalty_budget_up)
            # budget_insufficient specifically rewards budget escalation.
            if key == "budget_insufficient":
                base += float(cfg.w_safety) * 0.8
            if latency_pressure > 0.0:
                base -= latency_pressure
        if action in {"switch_rerank_cfg", "expand_candidates", "widen_evidence_window", "relax_constraints"}:
            base -= float(cfg.penalty_retry)
        if action == "relax_constraints":
            base -= float(cfg.penalty_relax)
        return float(base)

    cur = int(state.action_cursor.get(key, 0))
    candidates: list[tuple[float, int, str]] = []
    for i, action in enumerate(seq):
        if i < cur:
            continue
        if action == "escalate_budget" and int(state.budget_idx) >= len(budgets) - 1:
            continue
        candidates.append((_score(action), i, str(action)))
    if candidates:
        candidates.sort(key=lambda x: (-float(x[0]), int(x[1]), str(x[2])))
        _, chosen_idx, chosen_action = candidates[0]
        state.action_cursor[key] = int(chosen_idx) + 1
        return chosen_action, f"attribution={key},cfg={cfg.name}"
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
