from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.query_parser import parse_query_chain


def test_chain_derive_defaults_are_backward_compatible() -> None:
    chain = parse_query_chain("anchor=turn_head top_k=6 then token=SCENE_CHANGE top_k=6")
    assert chain is not None
    assert chain.derive == "time_only"
    assert chain.place_mode == "soft"
    assert chain.object_mode == "soft"
    assert chain.time_mode == "hard"


def test_chain_derive_explicit_modes_parse() -> None:
    chain = parse_query_chain(
        "place=first interaction_object=door top_k=6 "
        "then token=SCENE_CHANGE which=first top_k=6 "
        "chain_derive=time+place+object chain_place_mode=hard chain_object_mode=soft chain_time_mode=off"
    )
    assert chain is not None
    assert chain.derive == "time+place+object"
    assert chain.place_mode == "hard"
    assert chain.object_mode == "soft"
    assert chain.time_mode == "off"
