from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, EventV1, KeyClip, ObjectMemoryItemV0, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="trace_chain_repo_demo",
        meta={"duration_s": 30.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="stop_look", t=4.0, conf=0.8)]),
            Event(id="event_0002", t0=10.0, t1=30.0, anchors=[Anchor(type="turn_head", t=16.0, conf=0.85)]),
        ],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=10.0, place_segment_id="place_0001", interaction_primary_object="door", interaction_score=0.9),
            EventV1(id="ev1_0002", t0=10.0, t1=30.0, place_segment_id="place_0002", interaction_primary_object="cup", interaction_score=0.3),
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
                Token(id="tok_0001", t0=1.0, t1=1.2, type="SCENE_CHANGE", conf=0.9, source_event="ev1_0001"),
                Token(id="tok_0002", t0=11.0, t1=11.2, type="SCENE_CHANGE", conf=0.9, source_event="ev1_0002"),
            ],
        ),
        object_memory_v0=[
            ObjectMemoryItemV0(object_name="door", last_seen_t_ms=9000, last_contact_t_ms=8500, last_place_id="place_0001", score=0.8),
        ],
    )


def test_trace_chain_repo_section_contains_hint_fields(tmp_path: Path) -> None:
    json_path = tmp_path / "trace_chain_repo_v126_v03_decisions.json"
    out_dir = tmp_path / "trace_out"
    json_path.write_text(json.dumps(_make_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "trace_one_query.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--use-repo",
        "--repo-policy",
        "query_aware",
        "--query",
        "lost_object=door which=last top_k=6 then token=SCENE_CHANGE which=last top_k=6 chain_derive=time+object chain_object_mode=hard",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "repo_policy=query_aware" in proc.stdout
    assert "repo_selected_chunks=" in proc.stdout

    report_text = (out_dir / "trace_report.md").read_text(encoding="utf-8")
    assert "## Repo selection (query-aware)" in report_text
    assert "repo_filtered_chunks_before/after" in report_text
    assert "repo_time_filter_mode" in report_text
    assert "hint_object" in report_text

    trace_payload = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
    repo_trace = trace_payload.get("repo_selection", {}).get("trace", {}).get("selection_trace", {})
    assert isinstance(repo_trace, dict)
    assert "query_hints" in repo_trace
    assert str(repo_trace.get("repo_time_filter_mode", "")) == "overlap"
