from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class QueryCandidate(TypedDict):
    query: str
    reason: str
    priority: int


@dataclass
class QueryPlan:
    intent: Literal["anchor", "token", "decision", "time", "mixed"]
    candidates: list[QueryCandidate] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


def _contains_any(text: str, keywords: list[str]) -> bool:
    text = str(text).lower()
    return any(str(keyword).lower() in text for keyword in keywords)


def _append(candidates: list[QueryCandidate], seen: set[str], query: str, reason: str, priority: int) -> None:
    query = str(query).strip()
    if not query or query in seen:
        return
    candidates.append(QueryCandidate(query=query, reason=str(reason), priority=int(priority)))
    seen.add(query)


def _extract_constraints(text_raw: str) -> tuple[dict[str, Any], dict[str, bool]]:
    text = str(text_raw).lower()
    constraints: dict[str, Any] = {}
    flags = {"anchor": False, "token": False, "decision": False, "time": False}

    first_keywords = ["first", "earliest", "第一次", "最先", "最早"]
    last_keywords = ["last", "latest", "最后一次", "最后", "最近一次"]
    after_scene_keywords = [
        "after scene change",
        "after the scene changed",
        "场景变化后",
        "场景变化之后",
        "之后",
    ]
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
    ]
    stop_keywords = [
        "stop",
        "pause",
        "paused",
        "stopped",
        "停下",
        "停住",
        "站住",
        "暂停",
        "观察",
    ]
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
        "新环境",
    ]
    still_keywords = ["still", "stationary", "not moving", "static", "stand still"]
    moving_keywords = ["moving", "move", "walking", "running", "start moving"]
    interaction_keywords = [
        "touch", "grab", "hold", "handle object", "interact with", "interact directly with",
        "pick up", "put down", "manipulate", "manipulating", "object interaction", "handling",
        "use object", "handle item",
    ]
    decision_keywords = ["decision", "决策", "选择", "动作", "行为"]
    time_keywords = ["time=", "when", "什么时候", "何时", "time window", "区间", "时间段"]

    if _contains_any(text, first_keywords):
        constraints["which"] = "first"
    elif _contains_any(text, last_keywords):
        constraints["which"] = "last"

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

    if _contains_any(text, time_keywords):
        flags["time"] = True

    return constraints, flags


def _infer_intent(constraints: dict[str, Any], flags: dict[str, bool], text_raw: str) -> Literal["anchor", "token", "decision", "time", "mixed"]:
    if "anchor=" in text_raw:
        flags["anchor"] = True
    if "token=" in text_raw:
        flags["token"] = True
    if "decision=" in text_raw:
        flags["decision"] = True
    if "time=" in text_raw:
        flags["time"] = True

    if constraints.get("decision_type") and not constraints.get("anchor_type"):
        flags["decision"] = True
    if constraints.get("token_type"):
        flags["token"] = True
    if constraints.get("event_label"):
        flags["time"] = True
    if constraints.get("anchor_type"):
        flags["anchor"] = True

    # Prefer concrete intent when constraints include multiple hints.
    # Example: "after scene change ... look around" should still route as anchor intent.
    if constraints.get("anchor_type"):
        return "anchor"
    if constraints.get("decision_type") and flags.get("decision", False):
        return "decision"
    if constraints.get("token_type"):
        return "token"
    if flags.get("decision", False):
        return "decision"
    if flags.get("token", False):
        return "token"
    if flags.get("time", False):
        return "time"
    if flags.get("anchor", False):
        return "anchor"

    active = [name for name, on in flags.items() if on]
    if len(active) == 1:
        return active[0]  # type: ignore[return-value]
    return "mixed"


def plan(query_text: str) -> QueryPlan:
    """Create weighted candidate structured queries with intent and constraints."""

    text_raw = str(query_text or "").strip()
    constraints, flags = _extract_constraints(text_raw)
    intent = _infer_intent(constraints, flags, text_raw.lower())

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
            "intent_flags": flags,
            "candidate_count": len(candidates),
        },
    )
