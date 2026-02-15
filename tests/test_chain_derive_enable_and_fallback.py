from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_parser import parse_query
from pov_compiler.retrieval.retriever import Retriever
from pov_compiler.schemas import Output


def test_chain_derive_object_enabled_from_step1_query_fallback() -> None:
    hit = {
        "kind": "event",
        "id": "ev_0001",
        "t0": 10.0,
        "t1": 11.0,
        "score": 1.0,
        "source_query": "seed",
        "meta": {},
    }
    parsed = parse_query("lost_object=door which=last top_k=6")
    derived = Retriever._derive_constraints_from_hit(  # type: ignore[attr-defined]
        hit,  # type: ignore[arg-type]
        rel="after",
        window_s=30.0,
        derive="time+object",
        place_mode="off",
        object_mode="hard",
        time_mode="hard",
        output=Output(video_id="demo", meta={"duration_s": 30.0}),
        step1_parsed=parsed,
    )
    obj = dict(derived.get("object", {}))
    assert bool(obj.get("enabled", False))
    assert str(obj.get("source", "")) == "step1_query"
    assert str(obj.get("value", "")) == "door"

