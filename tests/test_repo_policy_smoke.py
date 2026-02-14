from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import DecisionPoint, Event, EventV1, Output


def _build_output() -> Output:
    return Output(
        video_id="repo_policy_smoke_demo",
        meta={"duration_s": 24.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=8.0, scores={"boundary_conf": 0.7}, meta={"label": "navigation"}),
            Event(id="event_0002", t0=8.0, t1=16.0, scores={"boundary_conf": 0.8}, meta={"label": "interaction-heavy"}),
            Event(id="event_0003", t0=16.0, t1=24.0, scores={"boundary_conf": 0.65}, meta={"label": "idle"}),
        ],
        events_v1=[
            EventV1(id="ev1_1", t0=0.0, t1=8.0, label="navigation", place_segment_id="p1", interaction_score=0.0),
            EventV1(
                id="ev1_2",
                t0=8.0,
                t1=16.0,
                label="interaction-heavy",
                place_segment_id="p2",
                interaction_primary_object="door",
                interaction_score=0.7,
            ),
            EventV1(
                id="ev1_3",
                t0=16.0,
                t1=24.0,
                label="idle",
                place_segment_id="p2",
                interaction_primary_object="door",
                interaction_score=0.4,
            ),
        ],
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=10.0,
                t0=9.8,
                t1=10.5,
                source_event="event_0002",
                action={"type": "ATTENTION_TURN_HEAD"},
                conf=0.8,
            )
        ],
    )


def test_repo_policy_smoke_outputs(tmp_path: Path) -> None:
    json_path = tmp_path / "demo_v03_decisions.json"
    out_dir = tmp_path / "repo_policy_out"
    json_path.write_text(json.dumps(_build_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "repo_policy_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--query",
        "anchor=turn_head top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "chunks_before_dedup=" in proc.stdout
    assert "selected_chunks=" in proc.stdout
    assert (out_dir / "repo_chunks.jsonl").exists()
    assert (out_dir / "repo_selected.jsonl").exists()
    assert (out_dir / "context.txt").exists()
    assert (out_dir / "report.md").exists()
    snapshot = out_dir / "snapshot.json"
    assert snapshot.exists()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert "cfg" in payload
    assert "stats" in payload
    assert "chunks_after_dedup" in payload["stats"]

