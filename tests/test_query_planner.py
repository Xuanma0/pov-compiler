from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_planner import QueryPlan, plan


def _idx_of_query_prefix(candidates: list[dict], prefix: str) -> int:
    for i, item in enumerate(candidates):
        if str(item.get("query", "")).startswith(prefix):
            return i
    return -1


def test_query_planner_turn_head_anchor_before_decision() -> None:
    qplan: QueryPlan = plan("什么时候我第一次开始左右张望？")
    idx_anchor = _idx_of_query_prefix(qplan.candidates, "anchor=turn_head")
    idx_decision = _idx_of_query_prefix(qplan.candidates, "decision=ATTENTION_TURN_HEAD")
    assert idx_anchor >= 0
    assert idx_decision >= 0
    assert idx_anchor < idx_decision
    assert qplan.intent in {"anchor", "mixed"}
    assert qplan.constraints.get("which") == "first"


def test_query_planner_stop_anchor_before_decision_and_scene_token() -> None:
    qplan: QueryPlan = plan("When did I pause to look around after the scene changed near the door?")
    idx_stop_anchor = _idx_of_query_prefix(qplan.candidates, "anchor=stop_look")
    idx_stop_decision = _idx_of_query_prefix(qplan.candidates, "decision=ATTENTION_STOP_LOOK")
    idx_scene = _idx_of_query_prefix(qplan.candidates, "token=SCENE_CHANGE")
    assert idx_stop_anchor >= 0
    assert idx_stop_decision >= 0
    assert idx_stop_anchor < idx_stop_decision
    assert idx_scene >= 0
    assert bool(qplan.constraints.get("after_scene_change", False))


def test_query_planner_contact_routes_to_event_label() -> None:
    qplan: QueryPlan = plan("When was I actively handling something with my hands?")
    idx_event_label = _idx_of_query_prefix(qplan.candidates, "event_label=interaction-heavy")
    assert idx_event_label >= 0
    idx_contact_gate = _idx_of_query_prefix(qplan.candidates, "event_label=interaction-heavy contact_min=")
    assert idx_contact_gate >= 0
