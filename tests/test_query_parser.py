from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_parser import parse_query


def test_query_parser_structured_fields() -> None:
    q = parse_query(
        'time=110-140 token=HIGHLIGHT,ATTENTION_STOP_LOOK anchor=stop_look,turn_head '
        "decision=ATTENTION_TURN_HEAD event=event_0003 top_k=6 mode=highlights max_tokens=160 "
        "max_highlights=8 max_decisions=5"
    )
    assert q.time_range == (110.0, 140.0)
    assert q.token_types == ["HIGHLIGHT", "ATTENTION_STOP_LOOK"]
    assert q.anchor_types == ["stop_look", "turn_head"]
    assert q.decision_types == ["ATTENTION_TURN_HEAD"]
    assert q.event_ids == ["event_0003"]
    assert q.top_k == 6
    assert q.mode == "highlights"
    assert q.budget_overrides["max_tokens"] == 160
    assert q.budget_overrides["max_highlights"] == 8
    assert q.budget_overrides["max_decisions"] == 5


def test_query_parser_text_and_budget() -> None:
    q = parse_query('text="bicycle door" top_k=3 max_events=5')
    assert q.text == "bicycle door"
    assert q.top_k == 3
    assert q.budget_overrides["max_events"] == 5
