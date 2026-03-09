from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_parser import parse_query


def test_query_parser_place_interaction_fields() -> None:
    parsed = parse_query(
        "place=first place_segment_id=place_0001,place_0002 "
        "interaction_min=0.35 interaction_object=cup top_k=6"
    )
    assert parsed.place == "first"
    assert parsed.place_segment_ids == ["place_0001", "place_0002"]
    assert parsed.interaction_min == 0.35
    assert parsed.interaction_object == "cup"
    assert parsed.top_k == 6
    assert "place" in parsed.filters_applied
    assert "interaction_min" in parsed.filters_applied
    assert "interaction_object" in parsed.filters_applied


def test_query_parser_place_interaction_invalid_keeps_warnings() -> None:
    parsed = parse_query("place=middle interaction_min=abc")
    assert parsed.place is None
    assert parsed.interaction_min is None
    assert parsed.parse_warnings
