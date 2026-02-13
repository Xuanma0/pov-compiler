from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_CRITICAL_QUERY_TYPES = [
    "hard_pseudo_anchor",
    "hard_pseudo_contact",
    "hard_pseudo_decision",
    "hard_pseudo_token",
    "pseudo_anchor",
    "pseudo_decision",
    "pseudo_token",
]


@dataclass
class SafetyGateConfig:
    enabled: bool = True
    max_critical_fn: int = 0
    strict_metric: str = "hit_at_k_strict"
    critical_query_types: list[str] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SafetyGateConfig":
        payload = dict(payload or {})
        query_types = payload.get("critical_query_types", DEFAULT_CRITICAL_QUERY_TYPES)
        if not isinstance(query_types, list):
            query_types = list(DEFAULT_CRITICAL_QUERY_TYPES)
        return cls(
            enabled=bool(payload.get("enabled", True)),
            max_critical_fn=int(payload.get("max_critical_fn", 0)),
            strict_metric=str(payload.get("strict_metric", "hit_at_k_strict")),
            critical_query_types=[str(x) for x in query_types],
        )


def classify_failure_reason(row: dict[str, Any]) -> str:
    filtered_before = int(row.get("filtered_hits_before", 0) or 0)
    filtered_after = int(row.get("filtered_hits_after", 0) or 0)
    candidate_count = int(row.get("candidate_count", 0) or 0)
    top1_in_distractor = float(row.get("top1_in_distractor", 0.0) or 0.0)
    budget_total_s = float(row.get("budget_max_total_s", 0.0) or 0.0)
    budget_tokens = int(float(row.get("budget_max_tokens", 0.0) or 0.0))
    budget_decisions = int(float(row.get("budget_max_decisions", 0.0) or 0.0))

    if filtered_before > 0 and filtered_after == 0:
        return "constraints_over_filtered"
    if top1_in_distractor > 0.0:
        return "retrieval_distractor"
    if budget_total_s < 20.0 or budget_tokens < 80 or budget_decisions < 6:
        return "budget_insufficient"
    if candidate_count <= 0 or filtered_after <= 0:
        return "evidence_missing"
    return "evidence_missing"


def build_safety_report(
    *,
    video_id: str,
    per_query_rows: list[dict[str, Any]],
    gate_cfg: SafetyGateConfig | dict[str, Any] | None = None,
    enforce_gate: bool = False,
) -> dict[str, Any]:
    cfg = gate_cfg if isinstance(gate_cfg, SafetyGateConfig) else SafetyGateConfig.from_dict(gate_cfg)
    metric = str(cfg.strict_metric)
    critical_types = set(str(x) for x in (cfg.critical_query_types or DEFAULT_CRITICAL_QUERY_TYPES))
    count_granularity = "row=(variant,budget,query)"

    critical_failures: list[dict[str, Any]] = []
    evaluated = 0
    variant_stats: dict[str, dict[str, float | int]] = {}
    for row in per_query_rows:
        query_type = str(row.get("query_type", ""))
        if query_type not in critical_types:
            continue
        evaluated += 1
        variant = str(row.get("variant", "unknown"))
        item = variant_stats.setdefault(variant, {"critical_fn_count": 0, "critical_fn_denominator": 0})
        item["critical_fn_denominator"] = int(item["critical_fn_denominator"]) + 1
        metric_value = float(row.get(metric, 0.0) or 0.0)
        if metric_value > 0.0:
            continue
        item["critical_fn_count"] = int(item["critical_fn_count"]) + 1
        critical_failures.append(
            {
                "qid": str(row.get("qid", "")),
                "query_type": query_type,
                "query": str(row.get("query", "")),
                "reason": classify_failure_reason(row),
                "chosen_query": str(row.get("chosen_query", "")),
                "chosen_plan_intent": str(row.get("chosen_plan_intent", "")),
                "filtered_hits_before": int(row.get("filtered_hits_before", 0) or 0),
                "filtered_hits_after": int(row.get("filtered_hits_after", 0) or 0),
                "constraints_applied": str(row.get("constraints_applied", "")),
                "constraints_relaxed": str(row.get("constraints_relaxed", "")),
                "top1_kind": str(row.get("top1_kind", "")),
                "top1_in_distractor": float(row.get("top1_in_distractor", 0.0) or 0.0),
                "budget_max_total_s": float(row.get("budget_max_total_s", 0.0) or 0.0),
                "budget_max_tokens": int(float(row.get("budget_max_tokens", 0.0) or 0.0)),
                "budget_max_decisions": int(float(row.get("budget_max_decisions", 0.0) or 0.0)),
            }
        )

    reason_counts: dict[str, int] = {}
    for item in critical_failures:
        reason = str(item.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    max_critical_fn = int(cfg.max_critical_fn)
    critical_fn_count = len(critical_failures)
    critical_fn_denominator = int(evaluated)
    critical_fn_rate = float(critical_fn_count / max(1, critical_fn_denominator))
    would_pass_gate = (critical_fn_count <= max_critical_fn) if bool(cfg.enabled) else True
    pass_gate = bool(would_pass_gate) if bool(enforce_gate) else True

    for variant, item in variant_stats.items():
        den = int(item.get("critical_fn_denominator", 0))
        num = int(item.get("critical_fn_count", 0))
        item["critical_fn_rate"] = float(num / max(1, den))
        item["variant"] = variant

    return {
        "video_id": str(video_id),
        "enabled": bool(cfg.enabled),
        "gate_enforced": bool(enforce_gate),
        "strict_metric": metric,
        "critical_query_types": sorted(critical_types),
        "count_granularity": count_granularity,
        "critical_fn_denominator": int(critical_fn_denominator),
        "evaluated_critical_queries": int(evaluated),
        "critical_fn_count": int(critical_fn_count),
        "max_critical_fn": int(max_critical_fn),
        "critical_fn_rate": float(critical_fn_rate),
        "would_pass_gate": bool(would_pass_gate),
        "pass_gate": bool(pass_gate),
        "variant_stats": variant_stats,
        "reason_counts": reason_counts,
        "critical_failures": critical_failures,
    }
