from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pov_compiler.retrieval.query_planner import QueryPlan
from pov_compiler.retrieval.reranker import Hit
from pov_compiler.schemas import Output


@dataclass
class HardConstraintConfig:
    enable_after_scene_change: bool = True
    enable_first_last: bool = True
    enable_type_match: bool = False
    enable_interaction: bool = True
    enable_place: bool = True
    relax_on_empty: bool = True
    relax_order: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.relax_order is None:
            self.relax_order = [
                "after_scene_change",
                "interaction_object",
                "interaction_min",
                "place_first_last",
                "type_match",
                "first_last",
            ]
        self.relax_order = [str(x) for x in self.relax_order]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_after_scene_change": bool(self.enable_after_scene_change),
            "enable_first_last": bool(self.enable_first_last),
            "enable_type_match": bool(self.enable_type_match),
            "enable_interaction": bool(self.enable_interaction),
            "enable_place": bool(self.enable_place),
            "relax_on_empty": bool(self.relax_on_empty),
            "relax_order": list(self.relax_order),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardConstraintConfig":
        payload = dict(data)
        return cls(
            enable_after_scene_change=bool(payload.get("enable_after_scene_change", True)),
            enable_first_last=bool(payload.get("enable_first_last", True)),
            enable_type_match=bool(payload.get("enable_type_match", False)),
            enable_interaction=bool(payload.get("enable_interaction", True)),
            enable_place=bool(payload.get("enable_place", True)),
            relax_on_empty=bool(payload.get("relax_on_empty", True)),
            relax_order=list(
                payload.get(
                    "relax_order",
                    ["after_scene_change", "interaction_object", "interaction_min", "place_first_last", "type_match", "first_last"],
                )
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HardConstraintConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"hard constraints config not found: {p}")
        text = p.read_text(encoding="utf-8")
        payload: dict[str, Any] = {}
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text) or {}
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                payload = loaded
        return cls.from_dict(payload)


@dataclass
class ConstraintApplyResult:
    hits: list[Hit]
    applied: list[str]
    relaxed: list[str]
    used_fallback: bool
    filtered_before: int
    filtered_after: int


def _scene_change_time(output: Output | None) -> float | None:
    if output is None:
        return None
    points: list[float] = []
    for token in output.token_codec.tokens:
        if str(token.type).upper() != "SCENE_CHANGE":
            continue
        points.append(0.5 * (float(token.t0) + float(token.t1)))
    if not points:
        return None
    return float(min(points))


def _type_match(hit: Hit, constraints: dict[str, Any]) -> bool:
    kind = str(hit["kind"])
    meta = hit.get("meta", {})
    anchor_type = str(constraints.get("anchor_type", "")).lower()
    token_type = str(constraints.get("token_type", "")).upper()
    decision_type = str(constraints.get("decision_type", "")).upper()

    if anchor_type:
        if kind == "highlight":
            types = meta.get("anchor_types", [])
            if isinstance(types, list) and anchor_type in {str(x).lower() for x in types}:
                return True
            if str(meta.get("anchor_type", "")).lower() == anchor_type:
                return True
        if kind == "decision":
            action_type = str(meta.get("action_type", "")).upper()
            if anchor_type == "turn_head" and "TURN_HEAD" in action_type:
                return True
            if anchor_type == "stop_look" and "STOP_LOOK" in action_type:
                return True
        return False

    if token_type:
        if kind == "token":
            return str(meta.get("token_type", "")).upper() == token_type
        if kind == "highlight":
            types = meta.get("token_types", [])
            if isinstance(types, list):
                return token_type in {str(x).upper() for x in types}
        return False

    if decision_type:
        if kind != "decision":
            return False
        return str(meta.get("action_type", "")).upper() == decision_type

    return True


def _match_interaction_object(hit: Hit, target: str) -> bool:
    value = str(hit.get("meta", {}).get("interaction_primary_object", "")).strip().lower()
    tgt = str(target).strip().lower()
    if not tgt:
        return True
    if not value:
        return False
    return tgt in value or value in tgt


def _match_interaction_min(hit: Hit, threshold: float) -> bool:
    try:
        score = float(hit.get("meta", {}).get("interaction_score", 0.0))
    except Exception:
        score = 0.0
    return float(score) >= float(threshold)


def _apply_place_first_last(hits: list[Hit], which: str) -> list[Hit]:
    grouped: dict[str, list[Hit]] = {}
    for hit in hits:
        seg = str(hit.get("meta", {}).get("place_segment_id", "")).strip() or "__unknown__"
        grouped.setdefault(seg, []).append(hit)
    out: list[Hit] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda x: (float(x["t0"]), float(x["t1"]), str(x["id"])))
        out.append(ordered[0] if which == "first" else ordered[-1])
    out.sort(key=lambda x: (float(x["t0"]), float(x["t1"]), str(x["kind"]), str(x["id"])))
    return out


def _apply_once(
    hits: list[Hit],
    plan: QueryPlan,
    cfg: HardConstraintConfig,
    enabled: set[str],
    output: Output | None,
) -> tuple[list[Hit], list[str]]:
    constraints = dict(plan.constraints)
    working = list(hits)
    applied: list[str] = []

    if "after_scene_change" in enabled and bool(constraints.get("after_scene_change", False)):
        scene_t = _scene_change_time(output)
        if scene_t is not None:
            working = [h for h in working if float(h["t1"]) >= float(scene_t)]
            applied.append("after_scene_change")

    if "interaction_object" in enabled and bool(cfg.enable_interaction):
        target_obj = str(constraints.get("interaction_object", "")).strip().lower()
        if target_obj:
            working = [h for h in working if _match_interaction_object(h, target_obj)]
            applied.append("interaction_object")

    if "interaction_min" in enabled and bool(cfg.enable_interaction):
        if constraints.get("interaction_min", None) is not None:
            try:
                threshold = float(constraints.get("interaction_min"))
            except Exception:
                threshold = 0.0
            working = [h for h in working if _match_interaction_min(h, threshold)]
            applied.append("interaction_min")

    if "type_match" in enabled and bool(cfg.enable_type_match):
        if any(k in constraints for k in ("anchor_type", "token_type", "decision_type")):
            working = [h for h in working if _type_match(h, constraints)]
            applied.append("type_match")

    if "first_last" in enabled and cfg.enable_first_last:
        which = str(constraints.get("which", "")).lower()
        if which in {"first", "last"} and working:
            ordered = sorted(working, key=lambda x: (float(x["t0"]), float(x["t1"]), str(x["id"])))
            working = [ordered[0] if which == "first" else ordered[-1]]
            applied.append("first_last")

    if "place_first_last" in enabled and cfg.enable_place:
        which_place = str(constraints.get("place", "")).lower()
        if which_place in {"first", "last"} and working:
            working = _apply_place_first_last(working, which_place)
            applied.append("place_first_last")

    if "place_segment_id" in enabled and cfg.enable_place:
        raw_ids = constraints.get("place_segment_ids", constraints.get("place_segment_id", []))
        if isinstance(raw_ids, str):
            target_ids = {x.strip() for x in raw_ids.split(",") if x.strip()}
        elif isinstance(raw_ids, list):
            target_ids = {str(x).strip() for x in raw_ids if str(x).strip()}
        else:
            target_ids = set()
        if target_ids:
            working = [h for h in working if str(h.get("meta", {}).get("place_segment_id", "")).strip() in target_ids]
            applied.append("place_segment_id")

    return working, applied


def apply_constraints_detailed(
    hits: list[Hit],
    query_plan: QueryPlan,
    cfg: HardConstraintConfig | dict[str, Any] | str | Path | None = None,
    output: Output | None = None,
) -> ConstraintApplyResult:
    if cfg is None:
        resolved = HardConstraintConfig()
    elif isinstance(cfg, HardConstraintConfig):
        resolved = cfg
    elif isinstance(cfg, dict):
        resolved = HardConstraintConfig.from_dict(cfg)
    else:
        resolved = HardConstraintConfig.from_yaml(cfg)

    before = len(hits)
    enabled: set[str] = set()
    if bool(resolved.enable_after_scene_change):
        enabled.add("after_scene_change")
    if bool(resolved.enable_first_last):
        enabled.add("first_last")
    if bool(resolved.enable_type_match):
        enabled.add("type_match")
    if bool(resolved.enable_interaction):
        enabled.add("interaction_object")
        enabled.add("interaction_min")
    if bool(resolved.enable_place):
        enabled.add("place_first_last")
        enabled.add("place_segment_id")

    filtered, applied = _apply_once(hits, query_plan, resolved, enabled, output)
    relaxed: list[str] = []

    if filtered:
        return ConstraintApplyResult(
            hits=filtered,
            applied=applied,
            relaxed=relaxed,
            used_fallback=False,
            filtered_before=before,
            filtered_after=len(filtered),
        )

    if not resolved.relax_on_empty:
        return ConstraintApplyResult(
            hits=[],
            applied=applied,
            relaxed=relaxed,
            used_fallback=False,
            filtered_before=before,
            filtered_after=0,
        )

    current_enabled = set(enabled)
    for name in resolved.relax_order:
        key = str(name)
        if key not in current_enabled:
            continue
        current_enabled.remove(key)
        relaxed.append(key)
        filtered_try, applied_try = _apply_once(hits, query_plan, resolved, current_enabled, output)
        if filtered_try:
            return ConstraintApplyResult(
                hits=filtered_try,
                applied=applied_try,
                relaxed=relaxed,
                used_fallback=False,
                filtered_before=before,
                filtered_after=len(filtered_try),
            )

    return ConstraintApplyResult(
        hits=list(hits),
        applied=[],
        relaxed=relaxed,
        used_fallback=True,
        filtered_before=before,
        filtered_after=len(hits),
    )


def apply_constraints(
    hits: list[Hit],
    query_plan: QueryPlan,
    cfg: HardConstraintConfig | dict[str, Any] | str | Path | None = None,
    output: Output | None = None,
) -> tuple[list[Hit], list[str], list[str]]:
    result = apply_constraints_detailed(hits=hits, query_plan=query_plan, cfg=cfg, output=output)
    return result.hits, result.applied, result.relaxed
