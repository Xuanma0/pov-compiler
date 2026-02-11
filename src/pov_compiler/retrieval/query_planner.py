from __future__ import annotations

from typing import TypedDict


class QueryCandidate(TypedDict):
    query: str
    reason: str
    priority: int


def _contains_any(text: str, keywords: list[str]) -> bool:
    text = str(text).lower()
    return any(str(keyword).lower() in text for keyword in keywords)


def _append(candidates: list[QueryCandidate], seen: set[str], query: str, reason: str, priority: int) -> None:
    query = str(query).strip()
    if not query or query in seen:
        return
    candidates.append(QueryCandidate(query=query, reason=str(reason), priority=int(priority)))
    seen.add(query)


def plan(query_text: str) -> list[QueryCandidate]:
    """Plan structured retrieval candidates from a natural-language query."""

    text_raw = str(query_text or "").strip()
    text = text_raw.lower()
    candidates: list[QueryCandidate] = []
    seen: set[str] = set()

    # Already-structured query keeps top priority.
    if "=" in text_raw:
        q = text_raw if "top_k=" in text_raw else f"{text_raw} top_k=6"
        _append(candidates, seen, q, "structured_query", 0)

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
    still_keywords = ["still", "stationary", "not moving", "静止", "不动", "保持不动"]
    moving_keywords = ["moving", "move", "walking", "running", "移动", "走动", "开始移动"]

    if _contains_any(text, turn_keywords):
        _append(candidates, seen, "anchor=turn_head top_k=6", "keyword look/张望 => anchor turn_head", 10)
        _append(
            candidates,
            seen,
            "decision=ATTENTION_TURN_HEAD top_k=6",
            "keyword look/张望 => decision ATTENTION_TURN_HEAD",
            20,
        )

    if _contains_any(text, stop_keywords):
        _append(candidates, seen, "anchor=stop_look top_k=6", "keyword stop/pause/停下 => anchor stop_look", 10)
        _append(
            candidates,
            seen,
            "decision=ATTENTION_STOP_LOOK top_k=6",
            "keyword stop/pause/停下 => decision ATTENTION_STOP_LOOK",
            20,
        )

    if _contains_any(text, scene_keywords):
        _append(candidates, seen, "token=SCENE_CHANGE top_k=6", "keyword scene-change => token SCENE_CHANGE", 10)

    if _contains_any(text, still_keywords):
        _append(candidates, seen, "token=MOTION_STILL top_k=6", "keyword still => token MOTION_STILL", 10)

    if _contains_any(text, moving_keywords):
        _append(candidates, seen, "token=MOTION_MOVING top_k=6", "keyword moving => token MOTION_MOVING", 10)

    if text_raw:
        _append(candidates, seen, f"text={text_raw} top_k=6", "fallback text retrieval", 90)

    candidates.sort(key=lambda x: (int(x["priority"]), str(x["query"])))
    return candidates
