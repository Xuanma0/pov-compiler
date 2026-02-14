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
    enable_object_match: bool = True
    enable_place: bool = True
    enable_chain_derived: bool = True
    relax_on_empty: bool = True
    relax_order: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.relax_order is None:
            self.relax_order = [
                "after_scene_change",
                "chain_object_match",
                "chain_place_match",
                "chain_time_range",
                "interaction_object",
                "object_match",
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
            "enable_object_match": bool(self.enable_object_match),
            "enable_place": bool(self.enable_place),
            "enable_chain_derived": bool(self.enable_chain_derived),
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
            enable_object_match=bool(payload.get("enable_object_match", True)),
            enable_place=bool(payload.get("enable_place", True)),
            enable_chain_derived=bool(payload.get("enable_chain_derived", True)),
            relax_on_empty=bool(payload.get("relax_on_empty", True)),
            relax_order=list(
                payload.get(
                    "relax_order",
                    [
                        "after_scene_change",
                        "chain_object_match",
                        "chain_place_match",
                        "chain_time_range",
                        "object_match",
                        "interaction_object",
                        "interaction_min",
                        "place_first_last",
                        "type_match",
                        "first_last",
                    ],
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
    steps: list["ConstraintStepTrace"]


@dataclass
class ConstraintStepTrace:
    name: str
    before: int
    after: int
    satisfied: bool
    details: dict[str, Any]


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


def _match_object_name(hit: Hit, target: str) -> bool:
    tgt = str(target).strip().lower()
    if not tgt:
        return True
    meta = hit.get("meta", {})
    values = [
        meta.get("interaction_primary_object", ""),
        meta.get("object_name", ""),
        meta.get("active_object_top1", ""),
        meta.get("label", ""),
        meta.get("name", ""),
    ]
    source = meta.get("source", {})
    if isinstance(source, dict):
        values.extend(
            [
                source.get("interaction_primary_object", ""),
                source.get("object_name", ""),
                source.get("active_object_top1", ""),
                source.get("label", ""),
            ]
        )
    has_explicit_object = False
    for value in values:
        norm = str(value).strip().lower()
        if not norm:
            continue
        has_explicit_object = True
        if tgt in norm or norm in tgt:
            return True
    source_query = str(hit.get("source_query", "")).strip().lower()
    if (not has_explicit_object) and source_query and (
        f"object={tgt}" in source_query or f"lost_object={tgt}" in source_query or f"interaction_object={tgt}" in source_query
    ):
        return True
    return False


def _match_interaction_min(hit: Hit, threshold: float) -> bool:
    try:
        score = float(hit.get("meta", {}).get("interaction_score", 0.0))
    except Exception:
        score = 0.0
    return float(score) >= float(threshold)


def _match_place_segment(hit: Hit, target: str) -> bool:
    tgt = str(target).strip()
    if not tgt:
        return True
    value = str(hit.get("meta", {}).get("place_segment_id", "")).strip()
    if not value:
        return False
    return value == tgt


def _match_chain_time(hit: Hit, t_min_s: float | None, t_max_s: float | None) -> bool:
    t0 = float(hit["t0"])
    t1 = float(hit["t1"])
    if t_min_s is not None and t1 < float(t_min_s):
        return False
    if t_max_s is not None and t0 > float(t_max_s):
        return False
    return True


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
) -> tuple[list[Hit], list[str], list[ConstraintStepTrace]]:
    constraints = dict(plan.constraints)
    working = list(hits)
    applied: list[str] = []
    steps: list[ConstraintStepTrace] = []

    def _run_step(
        name: str,
        should_run: bool,
        fn: Any,
        details: dict[str, Any] | None = None,
    ) -> None:
        nonlocal working
        if not should_run:
            return
        before = len(working)
        next_hits = fn(list(working))
        after = len(next_hits)
        working = list(next_hits)
        applied.append(name)
        steps.append(
            ConstraintStepTrace(
                name=str(name),
                before=int(before),
                after=int(after),
                satisfied=bool(after > 0),
                details=dict(details or {}),
            )
        )

    if "after_scene_change" in enabled and bool(constraints.get("after_scene_change", False)):
        scene_t = _scene_change_time(output)
        if scene_t is not None:
            _run_step(
                "after_scene_change",
                True,
                lambda xs: [h for h in xs if float(h["t1"]) >= float(scene_t)],
                {"scene_t": float(scene_t)},
            )

    if "chain_time_range" in enabled and bool(cfg.enable_chain_derived):
        time_mode = str(constraints.get("chain_time_mode", "")).strip().lower()
        if time_mode in {"hard", "off"}:
            t_min_raw = constraints.get("chain_time_min_s", None)
            t_max_raw = constraints.get("chain_time_max_s", None)
            t_min = float(t_min_raw) if isinstance(t_min_raw, (int, float)) else None
            t_max = float(t_max_raw) if isinstance(t_max_raw, (int, float)) else None
            if time_mode == "hard":
                _run_step(
                    "chain_time_range",
                    True,
                    lambda xs: [h for h in xs if _match_chain_time(h, t_min, t_max)],
                    {
                        "mode": "hard",
                        "t_min_s": t_min,
                        "t_max_s": t_max,
                    },
                )
            else:
                _run_step(
                    "chain_time_range",
                    True,
                    lambda xs: list(xs),
                    {"mode": "off", "t_min_s": t_min, "t_max_s": t_max},
                )

    if "chain_place_match" in enabled and bool(cfg.enable_chain_derived and cfg.enable_place):
        place_mode = str(constraints.get("chain_place_mode", "")).strip().lower()
        place_value = str(constraints.get("chain_place_value", "")).strip()
        if place_mode in {"hard", "soft", "off"} and place_value:
            if place_mode == "hard":
                _run_step(
                    "chain_place_match",
                    True,
                    lambda xs: [h for h in xs if _match_place_segment(h, place_value)],
                    {"mode": "hard", "place_segment_id": place_value},
                )
            else:
                _run_step(
                    "chain_place_match",
                    True,
                    lambda xs: list(xs),
                    {"mode": place_mode, "place_segment_id": place_value},
                )

    if "chain_object_match" in enabled and bool(cfg.enable_chain_derived and cfg.enable_object_match):
        obj_mode = str(constraints.get("chain_object_mode", "")).strip().lower()
        obj_value = str(constraints.get("chain_object_value", "")).strip().lower()
        if obj_mode in {"hard", "soft", "off"} and obj_value:
            if obj_mode == "hard":
                _run_step(
                    "chain_object_match",
                    True,
                    lambda xs: [h for h in xs if _match_object_name(h, obj_value)],
                    {"mode": "hard", "object_name": obj_value},
                )
            else:
                _run_step(
                    "chain_object_match",
                    True,
                    lambda xs: list(xs),
                    {"mode": obj_mode, "object_name": obj_value},
                )

    if "interaction_object" in enabled and bool(cfg.enable_interaction):
        target_obj = str(constraints.get("interaction_object", "")).strip().lower()
        if target_obj and not str(constraints.get("object_name", "")).strip():
            _run_step(
                "interaction_object",
                True,
                lambda xs: [h for h in xs if _match_interaction_object(h, target_obj)],
                {"interaction_object": str(target_obj)},
            )

    if "object_match" in enabled and bool(cfg.enable_object_match):
        target_obj = str(
            constraints.get("object_name", constraints.get("lost_object", constraints.get("object_last_seen", "")))
        ).strip().lower()
        require_object_match = bool(constraints.get("need_object_match", False))
        if target_obj or require_object_match:
            _run_step(
                "object_match",
                True,
                lambda xs: [h for h in xs if _match_object_name(h, target_obj)] if target_obj else list(xs),
                {
                    "object_name": str(target_obj),
                    "need_object_match": bool(require_object_match),
                },
            )

    if "interaction_min" in enabled and bool(cfg.enable_interaction):
        if constraints.get("interaction_min", None) is not None:
            try:
                threshold = float(constraints.get("interaction_min"))
            except Exception:
                threshold = 0.0
            _run_step(
                "interaction_min",
                True,
                lambda xs: [h for h in xs if _match_interaction_min(h, threshold)],
                {"interaction_min": float(threshold)},
            )

    if "type_match" in enabled and bool(cfg.enable_type_match):
        if any(k in constraints for k in ("anchor_type", "token_type", "decision_type")):
            _run_step(
                "type_match",
                True,
                lambda xs: [h for h in xs if _type_match(h, constraints)],
                {
                    "anchor_type": constraints.get("anchor_type", ""),
                    "token_type": constraints.get("token_type", ""),
                    "decision_type": constraints.get("decision_type", ""),
                },
            )

    if "first_last" in enabled and cfg.enable_first_last:
        which = str(constraints.get("which", "")).lower()
        if which in {"first", "last"}:
            _run_step(
                "first_last",
                True,
                lambda xs: (
                    [sorted(xs, key=lambda x: (float(x["t0"]), float(x["t1"]), str(x["id"])))[0 if which == "first" else -1]]
                    if xs
                    else []
                ),
                {"which": str(which)},
            )

    if "place_first_last" in enabled and cfg.enable_place:
        which_place = str(constraints.get("place", "")).lower()
        if which_place in {"first", "last"}:
            _run_step(
                "place_first_last",
                True,
                lambda xs: _apply_place_first_last(xs, which_place),
                {"place": str(which_place)},
            )

    if "place_segment_id" in enabled and cfg.enable_place:
        raw_ids = constraints.get("place_segment_ids", constraints.get("place_segment_id", []))
        if isinstance(raw_ids, str):
            target_ids = {x.strip() for x in raw_ids.split(",") if x.strip()}
        elif isinstance(raw_ids, list):
            target_ids = {str(x).strip() for x in raw_ids if str(x).strip()}
        else:
            target_ids = set()
        if target_ids:
            _run_step(
                "place_segment_id",
                True,
                lambda xs: [h for h in xs if str(h.get("meta", {}).get("place_segment_id", "")).strip() in target_ids],
                {"place_segment_ids": sorted(target_ids)},
            )

    return working, applied, steps


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
    if bool(resolved.enable_object_match):
        enabled.add("object_match")
    if bool(resolved.enable_place):
        enabled.add("place_first_last")
        enabled.add("place_segment_id")
    if bool(resolved.enable_chain_derived):
        enabled.add("chain_time_range")
        enabled.add("chain_place_match")
        enabled.add("chain_object_match")

    filtered, applied, steps = _apply_once(hits, query_plan, resolved, enabled, output)
    relaxed: list[str] = []
    all_steps: list[ConstraintStepTrace] = list(steps)
    applied_union: list[str] = list(dict.fromkeys(applied))

    if filtered:
        return ConstraintApplyResult(
            hits=filtered,
            applied=applied_union,
            relaxed=relaxed,
            used_fallback=False,
            filtered_before=before,
            filtered_after=len(filtered),
            steps=all_steps,
        )

    if not resolved.relax_on_empty:
        return ConstraintApplyResult(
            hits=[],
            applied=applied_union,
            relaxed=relaxed,
            used_fallback=False,
            filtered_before=before,
            filtered_after=0,
            steps=all_steps,
        )

    current_enabled = set(enabled)
    for name in resolved.relax_order:
        key = str(name)
        if key not in current_enabled:
            continue
        current_enabled.remove(key)
        relaxed.append(key)
        filtered_try, applied_try, steps_try = _apply_once(hits, query_plan, resolved, current_enabled, output)
        all_steps.extend(steps_try)
        for step_name in applied_try:
            if step_name not in applied_union:
                applied_union.append(step_name)
        if filtered_try:
            return ConstraintApplyResult(
                hits=filtered_try,
                applied=applied_union,
                relaxed=relaxed,
                used_fallback=False,
                filtered_before=before,
                filtered_after=len(filtered_try),
                steps=all_steps,
            )

    return ConstraintApplyResult(
        hits=list(hits),
        applied=applied_union,
        relaxed=relaxed,
        used_fallback=True,
        filtered_before=before,
        filtered_after=len(hits),
        steps=all_steps,
    )


def apply_constraints(
    hits: list[Hit],
    query_plan: QueryPlan,
    cfg: HardConstraintConfig | dict[str, Any] | str | Path | None = None,
    output: Output | None = None,
) -> tuple[list[Hit], list[str], list[str]]:
    result = apply_constraints_detailed(hits=hits, query_plan=query_plan, cfg=cfg, output=output)
    return result.hits, result.applied, result.relaxed
