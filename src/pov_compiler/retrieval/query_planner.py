from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from pov_compiler.retrieval.query_parser import QueryChain, parse_query_chain


class QueryCandidate(TypedDict):
    query: str
    reason: str
    priority: int


RetrievalPlanLiteral = Literal["baseline", "summary_then_token", "summary_then_decision", "summary_then_event"]


@dataclass
class QueryPlan:
    intent: Literal["anchor", "token", "decision", "time", "mixed"]
    candidates: list[QueryCandidate] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    retrieval_plan: RetrievalPlanLiteral = "baseline"
    plan_steps: list[str] = field(default_factory=lambda: ["baseline"])


@dataclass
class ChainPlan:
    chain_rel: str
    window_s: float
    top1_only: bool
    derive: str = "time_only"
    place_mode: str = "soft"
    object_mode: str = "soft"
    time_mode: str = "hard"
    steps: list[QueryPlan] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _contains_any(text: str, keywords: list[str]) -> bool:
    text = str(text).lower()
    return any(str(keyword).lower() in text for keyword in keywords)


def _append(candidates: list[QueryCandidate], seen: set[str], query: str, reason: str, priority: int) -> None:
    q = str(query).strip()
    if not q or q in seen:
        return
    candidates.append(QueryCandidate(query=q, reason=str(reason), priority=int(priority)))
    seen.add(q)


def _extract_constraints(text_raw: str) -> tuple[dict[str, Any], dict[str, bool]]:
    text = str(text_raw or "").lower()
    constraints: dict[str, Any] = {}
    flags = {"anchor": False, "token": False, "decision": False, "time": False}

    first_keywords = ["first", "earliest", "for the first time", "第一次", "最早", "绗竴娆"]
    last_keywords = ["last", "latest", "for the last time", "最后一次", "最近一次", "鏈€鍚庝竴娆"]
    place_keywords = ["place", "area", "location", "environment", "区域", "地点", "场景", "地段"]
    after_scene_keywords = ["after scene change", "after the scene changed", "场景变化后", "场景变化之后"]

    turn_keywords = [
        "look around",
        "looking around",
        "scan",
        "glance",
        "turn head",
        "look left",
        "look right",
        "张望",
        "左右看",
        "回头",
        "转头",
        "扫视",
        "看一眼",
        "寮犳湜",
        "鍥炲ご",
        "宸﹀彸",
    ]
    stop_keywords = ["stop", "pause", "paused", "stopped", "停下", "停住", "站住", "暂停", "观察", "鍋滀笅", "鍋滀綇"]
    scene_keywords = [
        "door",
        "elevator",
        "stairs",
        "exit",
        "enter",
        "scene change",
        "new area",
        "new environment",
        "门",
        "电梯",
        "楼梯",
        "出口",
        "进入",
        "出去",
        "场景变化",
        "闂",
        "鐢垫",
    ]
    still_keywords = ["still", "stationary", "not moving", "static", "stand still", "静止", "不动", "闈欐"]
    moving_keywords = ["moving", "move", "walking", "running", "start moving", "移动", "走动", "绉诲姩"]
    interaction_keywords = [
        "touch",
        "grab",
        "hold",
        "interact",
        "interaction",
        "object",
        "handling",
        "manipulate",
        "pick up",
        "put down",
        "接触",
        "拿着",
        "操作",
        "处理物体",
    ]
    decision_keywords = ["decision", "决策", "选择", "动作", "行为"]
    time_keywords = ["time=", "when", "什么时候", "何时", "time window", "时间段"]
    object_vocab = [
        "cup",
        "phone",
        "bottle",
        "knife",
        "laptop",
        "book",
        "key",
        "wallet",
        "bag",
        "door",
        "keyboard",
    ]

    if _contains_any(text, first_keywords):
        constraints["which"] = "first"
    elif _contains_any(text, last_keywords):
        constraints["which"] = "last"

    if _contains_any(text, place_keywords):
        constraints["place"] = constraints.get("which", "any")

    if _contains_any(text, after_scene_keywords):
        constraints["after_scene_change"] = True

    turn_hit = _contains_any(text, turn_keywords)
    stop_hit = _contains_any(text, stop_keywords)
    if stop_hit:
        constraints["anchor_type"] = "stop_look"
        constraints.setdefault("decision_type", "ATTENTION_STOP_LOOK")
        flags["anchor"] = True
    elif turn_hit:
        constraints["anchor_type"] = "turn_head"
        constraints.setdefault("decision_type", "ATTENTION_TURN_HEAD")
        flags["anchor"] = True

    if _contains_any(text, scene_keywords):
        constraints["token_type"] = "SCENE_CHANGE"
        flags["token"] = True
    elif _contains_any(text, interaction_keywords):
        constraints["event_label"] = "interaction-heavy"
        constraints["contact_min"] = 0.72
        constraints["interaction_min"] = 0.35
        flags["token"] = True
        flags["time"] = True
    elif _contains_any(text, still_keywords):
        constraints["token_type"] = "MOTION_STILL"
        flags["token"] = True
    elif _contains_any(text, moving_keywords):
        constraints["token_type"] = "MOTION_MOVING"
        flags["token"] = True

    if _contains_any(text, decision_keywords):
        flags["decision"] = True
        if "decision_type" not in constraints:
            if constraints.get("anchor_type") == "turn_head":
                constraints["decision_type"] = "ATTENTION_TURN_HEAD"
            elif constraints.get("anchor_type") == "stop_look":
                constraints["decision_type"] = "ATTENTION_STOP_LOOK"

    for name in object_vocab:
        if f" {name}" in f" {text}" or f"{name} " in f"{text} ":
            constraints.setdefault("interaction_object", name)
            constraints.setdefault("object_name", name)
            constraints.setdefault("need_object_match", True)
            break

    if "last seen" in text or "last touch" in text or "last interacted" in text:
        constraints.setdefault("which", "last")
    if "lost object" in text:
        constraints.setdefault("which", "last")
        constraints.setdefault("prefer_contact", True)
        constraints.setdefault("need_object_match", True)

    if _contains_any(text, time_keywords):
        flags["time"] = True

    return constraints, flags


def _infer_intent(
    constraints: dict[str, Any],
    flags: dict[str, bool],
    text_raw: str,
) -> Literal["anchor", "token", "decision", "time", "mixed"]:
    text = str(text_raw or "").lower()
    if "anchor=" in text:
        flags["anchor"] = True
    if "token=" in text:
        flags["token"] = True
    if "decision=" in text:
        flags["decision"] = True
    if "time=" in text:
        flags["time"] = True

    if constraints.get("anchor_type"):
        return "anchor"
    if constraints.get("decision_type") and flags.get("decision", False):
        return "decision"
    if constraints.get("token_type"):
        return "token"
    if constraints.get("place") is not None:
        return "anchor"
    if constraints.get("interaction_object") or constraints.get("interaction_min") is not None:
        return "mixed"
    if flags.get("decision", False):
        return "decision"
    if flags.get("token", False):
        return "token"
    if flags.get("time", False):
        return "time"
    if flags.get("anchor", False):
        return "anchor"
    return "mixed"


def plan(query_text: str) -> QueryPlan:
    text_raw = str(query_text or "").strip()
    constraints, flags = _extract_constraints(text_raw)

    try:
        parts = shlex.split(text_raw)
    except Exception:
        parts = text_raw.split()
    explicit_object_query = False
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = str(key).strip().lower()
        value = str(value).strip()
        if not value:
            continue
        if key == "place" and value.lower() in {"first", "last", "any"}:
            constraints["place"] = value.lower()
        elif key in {"place_segment_id", "place_segment"}:
            constraints["place_segment_id"] = value
            constraints["place_segment_ids"] = [x.strip() for x in value.split(",") if x.strip()]
        elif key == "interaction_object":
            constraints["interaction_object"] = value.lower()
            constraints.setdefault("object_name", value.lower())
            constraints["need_object_match"] = True
            explicit_object_query = True
        elif key == "object":
            constraints["object_name"] = value.lower()
            constraints.setdefault("interaction_object", value.lower())
            constraints["need_object_match"] = True
            explicit_object_query = True
        elif key == "lost_object":
            constraints["object_name"] = value.lower()
            constraints.setdefault("interaction_object", value.lower())
            constraints["which"] = "last"
            constraints["prefer_contact"] = True
            constraints["need_object_match"] = True
            explicit_object_query = True
        elif key == "object_last_seen":
            constraints["object_name"] = value.lower()
            constraints.setdefault("interaction_object", value.lower())
            constraints["which"] = "last"
            constraints["need_object_match"] = True
            explicit_object_query = True
        elif key == "interaction_min":
            try:
                constraints["interaction_min"] = float(value)
            except Exception:
                pass
        elif key == "which":
            which = value.lower()
            if which in {"first", "last"}:
                constraints["which"] = which
        elif key == "decision":
            constraints["decision_type"] = value.upper()
            flags["decision"] = True
        elif key == "token":
            values = [x.strip().upper() for x in value.split(",") if x.strip()]
            if values:
                constraints["token_type"] = values[0]
                flags["token"] = True
        elif key == "anchor":
            values = [x.strip().lower() for x in value.split(",") if x.strip()]
            if values:
                constraints["anchor_type"] = values[0]
                flags["anchor"] = True
        elif key == "time":
            flags["time"] = True
        elif key == "chain_time_min_s":
            try:
                constraints["chain_time_min_s"] = float(value)
            except Exception:
                pass
        elif key == "chain_time_max_s":
            try:
                constraints["chain_time_max_s"] = float(value)
            except Exception:
                pass
        elif key == "chain_time_mode":
            mode = value.lower().strip()
            if mode in {"hard", "off"}:
                constraints["chain_time_mode"] = mode
        elif key == "chain_place_value":
            pv = value.strip()
            if pv:
                constraints["chain_place_value"] = pv
        elif key == "chain_place_mode":
            mode = value.lower().strip()
            if mode in {"soft", "hard", "off"}:
                constraints["chain_place_mode"] = mode
        elif key == "chain_object_value":
            ov = value.strip().lower()
            if ov:
                constraints["chain_object_value"] = ov
        elif key == "chain_object_mode":
            mode = value.lower().strip()
            if mode in {"soft", "hard", "off"}:
                constraints["chain_object_mode"] = mode

    if explicit_object_query:
        # Explicit object-memory queries should not be hijacked by generic scene keywords like "door".
        constraints.pop("token_type", None)
        constraints.pop("event_label", None)
        flags["token"] = False

    intent = _infer_intent(constraints, flags, text_raw)

    candidates: list[QueryCandidate] = []
    seen: set[str] = set()

    if "=" in text_raw:
        q = text_raw if "top_k=" in text_raw else f"{text_raw} top_k=6"
        _append(candidates, seen, q, "structured_query", 0)

    anchor_type = str(constraints.get("anchor_type", "")).lower()
    if anchor_type == "turn_head":
        _append(candidates, seen, "anchor=turn_head top_k=8", "anchor hint: turn_head", 10)
        _append(candidates, seen, "decision=ATTENTION_TURN_HEAD top_k=8", "decision mirror: turn_head", 20)
    elif anchor_type == "stop_look":
        _append(candidates, seen, "anchor=stop_look top_k=8", "anchor hint: stop_look", 10)
        _append(candidates, seen, "decision=ATTENTION_STOP_LOOK top_k=8", "decision mirror: stop_look", 20)

    token_type = str(constraints.get("token_type", "")).upper()
    if token_type in {"SCENE_CHANGE", "MOTION_STILL", "MOTION_MOVING", "INTERACTION"}:
        _append(candidates, seen, f"token={token_type} top_k=8", f"token hint: {token_type}", 12)

    if constraints.get("interaction_object"):
        _append(
            candidates,
            seen,
            f"interaction_object={str(constraints['interaction_object'])} top_k=8",
            "interaction object hint",
            11,
        )
    if constraints.get("object_name"):
        obj = str(constraints["object_name"])
        which = str(constraints.get("which", "last")).lower()
        _append(
            candidates,
            seen,
            f"object={obj} which={which} top_k=8",
            "object memory hint",
            9,
        )
        _append(
            candidates,
            seen,
            f"lost_object={obj} top_k=8",
            "lost-object hint",
            8,
        )
    if constraints.get("interaction_min") is not None:
        _append(
            candidates,
            seen,
            f"interaction_min={float(constraints['interaction_min']):.2f} top_k=8",
            "interaction threshold hint",
            11,
        )
    if constraints.get("place") is not None:
        _append(candidates, seen, f"place={str(constraints['place'])} top_k=8", "place hint", 11)

    event_label = str(constraints.get("event_label", "")).strip().lower()
    if event_label:
        contact_min = constraints.get("contact_min", None)
        if isinstance(contact_min, (int, float)):
            _append(
                candidates,
                seen,
                f"event_label={event_label} contact_min={float(contact_min):.2f} top_k=8",
                f"event-label+contact hint: {event_label}",
                9,
            )
        else:
            _append(candidates, seen, f"event_label={event_label} top_k=8", f"event-label hint: {event_label}", 11)

    decision_type = str(constraints.get("decision_type", "")).upper()
    if decision_type in {"ATTENTION_TURN_HEAD", "ATTENTION_STOP_LOOK", "REORIENT_AND_SCAN", "TRANSITION"}:
        _append(candidates, seen, f"decision={decision_type} top_k=8", f"decision hint: {decision_type}", 15)

    if constraints.get("after_scene_change", False):
        _append(candidates, seen, "token=SCENE_CHANGE top_k=8", "constraint support: scene-change anchor", 18)

    if not candidates and text_raw:
        _append(candidates, seen, f"text={text_raw} top_k=6", "text fallback", 90)
    elif text_raw:
        _append(candidates, seen, f"text={text_raw} top_k=6", "text fallback", 95)

    candidates.sort(key=lambda x: (int(x["priority"]), str(x["query"])))
    return QueryPlan(
        intent=intent,
        candidates=candidates,
        constraints=constraints,
        debug={
            "query_text": text_raw,
            "intent_flags": dict(flags),
            "candidate_count": len(candidates),
        },
    )


def plan_chain(query_text: str) -> ChainPlan | None:
    chain: QueryChain | None = parse_query_chain(query_text)
    if chain is None:
        return None
    step_plans = [plan(step.raw) for step in chain.steps]
    return ChainPlan(
        chain_rel=str(chain.rel),
        window_s=float(chain.window_s),
        top1_only=bool(chain.top1_only),
        derive=str(chain.derive),
        place_mode=str(chain.place_mode),
        object_mode=str(chain.object_mode),
        time_mode=str(chain.time_mode),
        steps=step_plans,
        debug={
            "chain_query": str(query_text),
            "step_count": len(step_plans),
            "derived_strategy": "step1_top1_to_step2_constraints_v2",
        },
    )


def _normalize_planner_backend(value: str | None) -> str:
    text = str(value or "heuristic").strip().lower()
    if text not in {"heuristic", "model", "auto"}:
        return "heuristic"
    return text


def _planner_api_key_present(planner_model_cfg: dict[str, Any] | None) -> bool:
    from pov_compiler.models.presets import get_preset, normalize_provider
    import os

    cfg = dict(planner_model_cfg or {})
    provider = normalize_provider(str(cfg.get("provider", "fake")))
    if provider == "fake":
        return True
    try:
        preset = get_preset(provider)
        env_name = str(cfg.get("api_key_env") or preset.default_api_key_env)
    except Exception:
        env_name = str(cfg.get("api_key_env", ""))
    if not env_name:
        return False
    return bool(os.environ.get(env_name, ""))


def plan_with_backend(
    query_text: str,
    *,
    planner_backend: str = "heuristic",
    planner_model_cfg: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    signal_overview: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[QueryPlan, dict[str, Any]]:
    backend_req = _normalize_planner_backend(planner_backend)
    backend_used = backend_req
    meta: dict[str, Any] = {
        "planner_backend_requested": backend_req,
        "planner_backend_used": "heuristic",
        "planner_fallback_reason": "",
        "planner_model_meta": {},
        "planner_plan": {},
        "planner_cache_stats": {},
    }
    if backend_req == "auto":
        backend_used = "model" if _planner_api_key_present(planner_model_cfg) else "heuristic"
    if backend_used != "model":
        q = plan(query_text)
        meta["planner_backend_used"] = "heuristic"
        meta["planner_plan"] = {
            "retrieval_plan": str(getattr(q, "retrieval_plan", "baseline")),
            "normalized_query": str(query_text),
            "constraints": dict(getattr(q, "constraints", {})),
            "confidence": 1.0,
            "notes": "heuristic",
        }
        return q, meta

    from pov_compiler.retrieval.model_planner import plan_query_with_model

    planner_plan, planner_meta = plan_query_with_model(
        str(query_text),
        planner_model_cfg=dict(planner_model_cfg or {}),
        budget=dict(budget or {}),
        signal_overview=dict(signal_overview or {}),
        seed=int(seed),
    )

    fallback_reason = str(planner_meta.get("fallback_reason", "")) if isinstance(planner_meta, dict) else ""
    parse_ok = bool(planner_meta.get("parse_ok", False)) if isinstance(planner_meta, dict) else False

    if fallback_reason and not parse_ok:
        q = plan(query_text)
        meta["planner_backend_used"] = "heuristic"
        meta["planner_fallback_reason"] = fallback_reason
        meta["planner_model_meta"] = dict(planner_meta or {})
        meta["planner_cache_stats"] = dict(planner_meta.get("cache", {})) if isinstance(planner_meta, dict) else {}
        meta["planner_plan"] = {
            "retrieval_plan": str(getattr(q, "retrieval_plan", "baseline")),
            "normalized_query": str(query_text),
            "constraints": dict(getattr(q, "constraints", {})),
            "confidence": 1.0,
            "notes": "heuristic_fallback",
        }
        return q, meta

    qplan = planner_plan.to_query_plan(default_priority=0)
    meta["planner_backend_used"] = "model"
    meta["planner_model_meta"] = dict(planner_meta or {})
    meta["planner_cache_stats"] = dict(planner_meta.get("cache", {})) if isinstance(planner_meta, dict) else {}
    meta["planner_fallback_reason"] = str(planner_meta.get("fallback_reason", "")) if isinstance(planner_meta, dict) else ""
    meta["planner_plan"] = planner_plan.to_dict()
    return qplan, meta


def plan_chain_with_backend(
    query_text: str,
    *,
    planner_backend: str = "heuristic",
    planner_model_cfg: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    signal_overview: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[ChainPlan | None, dict[str, Any]]:
    chain = parse_query_chain(str(query_text))
    if chain is None:
        return None, {
            "planner_backend_requested": _normalize_planner_backend(planner_backend),
            "planner_backend_used": "heuristic",
            "planner_fallback_reason": "not_chain",
            "planner_model_meta": {},
            "planner_plan": {},
            "planner_cache_stats": {},
        }

    step1_plan, step1_meta = plan_with_backend(
        chain.steps[0].raw,
        planner_backend=planner_backend,
        planner_model_cfg=planner_model_cfg,
        budget=budget,
        signal_overview=signal_overview,
        seed=seed,
    )
    step2_plan, step2_meta = plan_with_backend(
        chain.steps[1].raw,
        planner_backend=planner_backend,
        planner_model_cfg=planner_model_cfg,
        budget=budget,
        signal_overview=signal_overview,
        seed=seed + 1,
    )
    cplan = ChainPlan(
        chain_rel=str(chain.rel),
        window_s=float(chain.window_s),
        top1_only=bool(chain.top1_only),
        derive=str(chain.derive),
        place_mode=str(chain.place_mode),
        object_mode=str(chain.object_mode),
        time_mode=str(chain.time_mode),
        steps=[step1_plan, step2_plan],
        debug={
            "chain_query": str(query_text),
            "step_count": 2,
            "derived_strategy": "step1_top1_to_step2_constraints_v2",
            "planner_step1": dict(step1_meta),
            "planner_step2": dict(step2_meta),
        },
    )
    merged_meta = {
        "planner_backend_requested": str(step2_meta.get("planner_backend_requested", planner_backend)),
        "planner_backend_used": str(step2_meta.get("planner_backend_used", "heuristic")),
        "planner_fallback_reason": str(step2_meta.get("planner_fallback_reason", "")),
        "planner_model_meta": dict(step2_meta.get("planner_model_meta", {})),
        "planner_plan": dict(step2_meta.get("planner_plan", {})),
        "planner_cache_stats": dict(step2_meta.get("planner_cache_stats", {})),
    }
    return cplan, merged_meta
