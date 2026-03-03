from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pov_compiler.schemas import Anchor, Event, EventV1, KeyClip, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="trace_summary_plan_demo",
        meta={"duration_s": 36.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=12.0, anchors=[Anchor(type="turn_head", t=4.0, conf=0.8)]),
            Event(id="event_0002", t0=12.0, t1=24.0, anchors=[Anchor(type="stop_look", t=16.0, conf=0.9)]),
            Event(id="event_0003", t0=24.0, t1=36.0, anchors=[]),
        ],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=12.0, place_segment_id="place_a", interaction_primary_object="door", interaction_score=0.9),
            EventV1(id="ev1_0002", t0=12.0, t1=24.0, place_segment_id="place_b", interaction_primary_object="cup", interaction_score=0.4),
            EventV1(id="ev1_0003", t0=24.0, t1=36.0, place_segment_id="place_c", interaction_primary_object="bag", interaction_score=0.3),
        ],
        highlights=[
            KeyClip(id="hl_0001", t0=3.8, t1=4.3, source_event="ev1_0001", anchor_type="turn_head", anchor_t=4.0, conf=0.8),
            KeyClip(id="hl_0002", t0=15.8, t1=16.3, source_event="ev1_0002", anchor_type="stop_look", anchor_t=16.0, conf=0.9),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=1.0, t1=1.2, type="SCENE_CHANGE", conf=0.8, source_event="ev1_0001"),
                Token(id="tok_0002", t0=13.0, t1=13.1, type="SCENE_CHANGE", conf=0.8, source_event="ev1_0002"),
                Token(id="tok_0003", t0=30.0, t1=30.2, type="MOTION_STILL", conf=0.8, source_event="ev1_0003"),
            ],
        ),
    )


def test_trace_one_query_summary_plan(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "trace_out"
    json_path.write_text(json.dumps(_make_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "trace_one_query.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--query",
        "lost_object=door top_k=6",
        "--retrieval-plan",
        "summary_then_token",
        "--summary-topk",
        "2",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "retrieval_plan=summary_then_token" in proc.stdout
    assert "stage0_summary_hits_before=" in proc.stdout
    assert "derived_time_window_ms=" in proc.stdout
    assert "stage1_filtered_hits_before=" in proc.stdout

    report = (out_dir / "trace_report.md").read_text(encoding="utf-8")
    assert "## Summary-first Planning" in report
    assert ("chain_time_range" in report) or ("repo_summary_time_range" in report)
