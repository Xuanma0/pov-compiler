from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, EventV1, KeyClip, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="trace_chain_v2_demo",
        meta={"duration_s": 30.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="stop_look", t=4.0, conf=0.8)]),
            Event(id="event_0002", t0=10.0, t1=30.0, anchors=[Anchor(type="turn_head", t=16.0, conf=0.85)]),
        ],
        events_v1=[
            EventV1(
                id="ev1_0001",
                t0=0.0,
                t1=10.0,
                place_segment_id="place_0001",
                interaction_primary_object="door",
                interaction_score=0.8,
            ),
            EventV1(
                id="ev1_0002",
                t0=10.0,
                t1=30.0,
                place_segment_id="place_0002",
                interaction_primary_object="cup",
                interaction_score=0.4,
            ),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=3.5,
                t1=4.5,
                source_event="ev1_0001",
                anchor_type="stop_look",
                anchor_t=4.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=1.0, t1=1.1, type="SCENE_CHANGE", conf=0.8, source_event="ev1_0001"),
                Token(id="tok_0002", t0=11.0, t1=11.2, type="SCENE_CHANGE", conf=0.8, source_event="ev1_0002"),
            ],
        ),
    )


def test_trace_report_contains_chain_derived_constraints_table(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "trace_out"
    json_path.write_text(json.dumps(_make_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    query = (
        "anchor=stop_look top_k=6 then token=SCENE_CHANGE which=first top_k=6 "
        "chain_derive=time+place+object chain_place_mode=hard chain_object_mode=soft"
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "trace_one_query.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--query",
        query,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "is_chain=true" in proc.stdout
    assert "derived_constraints=" in proc.stdout
    assert "step2_applied_constraints=" in proc.stdout

    report_text = (out_dir / "trace_report.md").read_text(encoding="utf-8")
    assert "## Derived Constraints" in report_text
    assert "chain_time_range" in report_text
    assert "| place |" in report_text
    assert "| object |" in report_text
