from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


@dataclass(frozen=True)
class ChainBackoffDecision:
    level: int
    reason: str
    debug: dict[str, Any]


class FixedChainBackoff:
    name = "strict"

    def __init__(self, level: int = 0):
        self.level = int(level)

    def choose_level(self, context: dict[str, Any]) -> ChainBackoffDecision:
        levels = [int(x) for x in list(context.get("available_levels", [0, 1, 2, 3, 4]))]
        levels = sorted(set(levels))
        if not levels:
            levels = [0]
        chosen = min(levels, key=lambda x: abs(int(x) - int(self.level)))
        return ChainBackoffDecision(
            level=int(chosen),
            reason=f"fixed_level={int(chosen)}",
            debug={"strategy": self.name, "available_levels": levels},
        )


class LadderFirstNonZero:
    name = "ladder"

    def choose_level(self, context: dict[str, Any]) -> ChainBackoffDecision:
        levels = [int(x) for x in list(context.get("available_levels", [0, 1, 2, 3, 4]))]
        levels = sorted(set(levels))
        if not levels:
            levels = [0]
        metrics = dict(context.get("level_metrics", {}))
        for level in levels:
            row = metrics.get(level, metrics.get(str(level), {}))
            count = int(_to_float((row or {}).get("hit_count", 0.0), 0.0))
            if count > 0:
                return ChainBackoffDecision(
                    level=int(level),
                    reason="ladder_first_nonzero",
                    debug={"strategy": self.name, "hit_count": count},
                )
        return ChainBackoffDecision(
            level=int(levels[-1]),
            reason="ladder_exhausted",
            debug={"strategy": self.name, "exhausted": True},
        )


class AdaptiveChainBackoff:
    name = "adaptive"

    def __init__(
        self,
        *,
        alpha_latency: float = 0.20,
        beta_safety: float = 0.50,
        gamma_exhausted: float = 1.00,
        delta_level: float = 0.08,
        seed: int = 0,
    ):
        self.alpha_latency = float(alpha_latency)
        self.beta_safety = float(beta_safety)
        self.gamma_exhausted = float(gamma_exhausted)
        self.delta_level = float(delta_level)
        self.seed = int(seed)

    def choose_level(self, context: dict[str, Any]) -> ChainBackoffDecision:
        levels = [int(x) for x in list(context.get("available_levels", [0, 1, 2, 3, 4]))]
        levels = sorted(set(levels))
        if not levels:
            levels = [0]
        metrics = dict(context.get("level_metrics", {}))
        latency_cap_ms = max(1.0, _to_float(context.get("latency_cap_ms", 25.0), 25.0))
        max_trials = max(1.0, _to_float(context.get("max_trials_per_query", 3.0), 3.0))

        scores: dict[int, float] = {}
        details: dict[str, dict[str, Any]] = {}
        max_level = max(levels) if levels else 0
        for level in levels:
            row = metrics.get(level, metrics.get(str(level), {}))
            row = dict(row or {})
            hit_count = int(_to_float(row.get("hit_count", 0.0), 0.0))
            exhausted = hit_count <= 0
            strict_hit = _to_float(row.get("hit_at_k_strict", 0.0), 0.0)
            chain_success = _to_float(row.get("chain_success", strict_hit), strict_hit)
            quality = max(strict_hit, chain_success)

            critical = _to_float(row.get("safety_is_critical_fn", 0.0), 0.0)
            distractor = _to_float(row.get("top1_in_distractor_rate", 0.0), 0.0)
            safety_risk = max(critical, distractor)

            latency = _to_float(row.get("latency_e2e_ms", 0.0), 0.0)
            latency_term = max(0.0, float(latency) / float(latency_cap_ms))
            level_term = float(level) / float(max(1, max_level))
            trials_term = float(level + 1) / float(max_trials)

            objective = (
                float(quality)
                - self.alpha_latency * latency_term
                - self.beta_safety * safety_risk
                - self.gamma_exhausted * (1.0 if exhausted else 0.0)
                - self.delta_level * max(level_term, trials_term)
            )
            scores[int(level)] = float(objective)
            details[str(level)] = {
                "quality": float(quality),
                "safety_risk": float(safety_risk),
                "latency_term": float(latency_term),
                "level_term": float(level_term),
                "trials_term": float(trials_term),
                "exhausted": bool(exhausted),
                "objective": float(objective),
            }

        chosen = min(levels)
        best_score = None
        for level in levels:
            score = float(scores.get(level, -1e9))
            if best_score is None or score > best_score + 1e-12:
                chosen = int(level)
                best_score = float(score)
            elif abs(score - float(best_score)) <= 1e-12 and int(level) < int(chosen):
                chosen = int(level)

        return ChainBackoffDecision(
            level=int(chosen),
            reason=f"adaptive_objective_best(level={int(chosen)})",
            debug={
                "strategy": self.name,
                "seed": int(self.seed),
                "alpha_latency": float(self.alpha_latency),
                "beta_safety": float(self.beta_safety),
                "gamma_exhausted": float(self.gamma_exhausted),
                "delta_level": float(self.delta_level),
                "scores": scores,
                "details": details,
            },
        )

