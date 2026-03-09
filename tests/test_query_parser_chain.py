from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_parser import parse_query_chain


def test_parse_query_chain_defaults() -> None:
    chain = parse_query_chain("anchor=turn_head top_k=6 then decision=ATTENTION_TURN_HEAD top_k=4")
    assert chain is not None
    assert len(chain.steps) == 2
    assert chain.rel == "after"
    assert abs(float(chain.window_s) - 30.0) < 1e-6
    assert chain.top1_only is True
    assert chain.steps[0].parsed.anchor_types == ["turn_head"]
    assert chain.steps[1].parsed.decision_types == ["ATTENTION_TURN_HEAD"]


def test_parse_query_chain_explicit_rel_window() -> None:
    chain = parse_query_chain(
        "place=first then lost_object=door which=last chain_rel=around chain_window_s=12 chain_top1_only=false"
    )
    assert chain is not None
    assert chain.rel == "around"
    assert abs(float(chain.window_s) - 12.0) < 1e-6
    assert chain.top1_only is False
    assert chain.steps[1].parsed.need_object_match is True

