from __future__ import annotations

from typing import Any


def compute_objective(
    *,
    hit_at_k_strict: float,
    hit_at_1_strict: float,
    fp_rate: float,
    metric: str,
) -> float:
    metric = str(metric)
    if metric == "hit_at_k_strict":
        return float(hit_at_k_strict)
    if metric == "hit_at_1_strict":
        return float(hit_at_1_strict)
    if metric == "fp_rate":
        return float(-fp_rate)
    if metric == "objective_combo":
        return float(hit_at_k_strict - 0.5 * fp_rate)
    raise ValueError(f"unknown metric: {metric}")


def summarize_variant_metrics(rows: list[dict[str, Any]], variant: str = "full") -> dict[str, float]:
    subset = [r for r in rows if str(r.get("variant", "")) == str(variant)]
    if not subset:
        return {
            "hit_at_k_strict": 0.0,
            "hit_at_1_strict": 0.0,
            "fp_rate": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
        }
    n = float(len(subset))
    return {
        "hit_at_k_strict": float(sum(float(r.get("hit_at_k_strict", 0.0)) for r in subset) / n),
        "hit_at_1_strict": float(sum(float(r.get("hit_at_1_strict", 0.0)) for r in subset) / n),
        "fp_rate": float(sum(float(r.get("top1_in_distractor_rate", 0.0)) for r in subset) / n),
        "hit_at_k": float(sum(float(r.get("hit_at_k", 0.0)) for r in subset) / n),
        "mrr": float(sum(float(r.get("mrr", 0.0)) for r in subset) / n),
    }


def rank_rows_by_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    metric = str(metric)
    reverse = metric != "fp_rate"

    def _value(row: dict[str, Any]) -> float:
        if metric == "hit_at_k_strict":
            return float(row.get("hit_at_k_strict", 0.0))
        if metric == "hit_at_1_strict":
            return float(row.get("hit_at_1_strict", 0.0))
        if metric == "fp_rate":
            return float(row.get("fp_rate", 0.0))
        if metric == "objective_combo":
            return float(row.get("objective", 0.0))
        raise ValueError(f"unknown metric: {metric}")

    return sorted(rows, key=_value, reverse=reverse)
