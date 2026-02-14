from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class WeightConfig:
    name: str = "default"

    # Intent bonuses.
    bonus_intent_token_on_token: float = 1.0
    bonus_intent_token_on_highlight: float = 0.2
    bonus_intent_token_on_decision: float = 0.1
    bonus_intent_token_on_event: float = 0.0

    bonus_intent_decision_on_decision: float = 1.0
    bonus_intent_decision_on_highlight: float = 0.2
    bonus_intent_decision_on_token: float = 0.1
    bonus_intent_decision_on_event: float = 0.0

    bonus_intent_anchor_on_highlight: float = 1.0
    bonus_intent_anchor_on_decision: float = 0.3
    bonus_intent_anchor_on_token: float = 0.0
    bonus_intent_anchor_on_event: float = 0.0

    bonus_intent_time_on_event: float = 0.8
    bonus_intent_time_on_highlight: float = 0.4
    bonus_intent_time_on_decision: float = 0.2
    bonus_intent_time_on_token: float = 0.2

    bonus_intent_mixed_on_event: float = 0.2
    bonus_intent_mixed_on_highlight: float = 0.2
    bonus_intent_mixed_on_decision: float = 0.2
    bonus_intent_mixed_on_token: float = 0.2

    # Constraint matching.
    bonus_anchor_highlight_match: float = 0.6
    bonus_anchor_decision_match: float = 0.4
    penalty_anchor_highlight_mismatch: float = -0.2
    penalty_anchor_decision_mismatch: float = -0.1

    bonus_token_match: float = 0.6
    bonus_token_highlight_overlap: float = 0.2
    penalty_token_mismatch: float = -0.2

    bonus_decision_match: float = 0.6
    penalty_decision_mismatch: float = -0.2

    bonus_first: float = 0.5
    bonus_last: float = 0.5

    # Soft scene-change penalty (hard-filter is handled separately).
    penalty_before_scene_change: float = 0.6

    # Distractor proximity.
    penalty_distractor_near: float = 0.3
    distractor_near_window_s: float = 5.0

    # Confidence based bonuses.
    bonus_conf_scale: float = 0.2
    bonus_boundary_scale: float = 0.15
    bonus_priority_scale: float = 0.05
    bonus_priority_cap: float = 3.0

    # Decision-aligned reranking weights.
    w_trigger: float = 0.3
    w_action: float = 0.35
    w_constraint: float = 0.2
    w_outcome: float = 0.1
    w_evidence: float = 0.15
    w_semantic: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, allow_out_of_range: bool = False) -> "WeightConfig":
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in dict(data).items() if k in valid}
        cfg = cls(**kwargs)
        return cfg.validate(allow_out_of_range=allow_out_of_range)

    @classmethod
    def from_yaml(cls, path: str | Path, *, allow_out_of_range: bool = False) -> "WeightConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"rerank config not found: {p}")
        text = p.read_text(encoding="utf-8")
        payload: dict[str, Any] = {}
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text) or {}
            if isinstance(loaded, dict):
                payload = dict(loaded)
        except Exception:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                payload = dict(loaded)
        return cls.from_dict(payload, allow_out_of_range=allow_out_of_range)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def short_hash(self) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

    def validate(self, *, allow_out_of_range: bool = False) -> "WeightConfig":
        def _clamp(name: str, low: float, high: float) -> None:
            value = float(getattr(self, name))
            if low <= value <= high:
                return
            if allow_out_of_range:
                return
            clipped = min(max(value, low), high)
            warnings.warn(
                f"WeightConfig.{name}={value} out of range [{low}, {high}], clamped to {clipped}",
                RuntimeWarning,
                stacklevel=2,
            )
            setattr(self, name, clipped)

        ranged_fields = [
            "bonus_intent_token_on_token",
            "bonus_intent_token_on_highlight",
            "bonus_intent_token_on_decision",
            "bonus_intent_token_on_event",
            "bonus_intent_decision_on_decision",
            "bonus_intent_decision_on_highlight",
            "bonus_intent_decision_on_token",
            "bonus_intent_decision_on_event",
            "bonus_intent_anchor_on_highlight",
            "bonus_intent_anchor_on_decision",
            "bonus_intent_anchor_on_token",
            "bonus_intent_anchor_on_event",
            "bonus_intent_time_on_event",
            "bonus_intent_time_on_highlight",
            "bonus_intent_time_on_decision",
            "bonus_intent_time_on_token",
            "bonus_intent_mixed_on_event",
            "bonus_intent_mixed_on_highlight",
            "bonus_intent_mixed_on_decision",
            "bonus_intent_mixed_on_token",
            "bonus_anchor_highlight_match",
            "bonus_anchor_decision_match",
            "penalty_anchor_highlight_mismatch",
            "penalty_anchor_decision_mismatch",
            "bonus_token_match",
            "bonus_token_highlight_overlap",
            "penalty_token_mismatch",
            "bonus_decision_match",
            "penalty_decision_mismatch",
            "bonus_first",
            "bonus_last",
            "penalty_before_scene_change",
            "penalty_distractor_near",
            "bonus_conf_scale",
            "bonus_boundary_scale",
            "bonus_priority_scale",
            "w_trigger",
            "w_action",
            "w_constraint",
            "w_outcome",
            "w_evidence",
            "w_semantic",
        ]
        for field_name in ranged_fields:
            _clamp(field_name, -5.0, 5.0)

        _clamp("distractor_near_window_s", 0.0, 30.0)
        _clamp("bonus_priority_cap", 0.0, 20.0)
        return self


def resolve_weight_config(
    cfg: WeightConfig | dict[str, Any] | str | Path | None,
    *,
    allow_out_of_range: bool = False,
) -> WeightConfig:
    if cfg is None:
        return WeightConfig().validate(allow_out_of_range=allow_out_of_range)
    if isinstance(cfg, WeightConfig):
        return cfg.validate(allow_out_of_range=allow_out_of_range)
    if isinstance(cfg, dict):
        return WeightConfig.from_dict(cfg, allow_out_of_range=allow_out_of_range)
    if isinstance(cfg, (str, Path)):
        return WeightConfig.from_yaml(cfg, allow_out_of_range=allow_out_of_range)
    raise TypeError("Unsupported reranker config type")
