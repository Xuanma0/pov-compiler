from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import EventV1, Output


def test_place_interaction_smoke_outputs(tmp_path: Path) -> None:
    output = Output(
        video_id="pi_smoke_demo",
        meta={"duration_s": 20.0},
        events_v1=[
            EventV1(
                id="event_0001",
                t0=0.0,
                t1=10.0,
                place_segment_id="place_0001",
                place_segment_conf=0.8,
                place_segment_reason="scene_change",
                interaction_signature={"contact_rate": 0.5, "contact_burst_count": 2, "active_object_top1": "cup"},
                interaction_primary_object="cup",
                interaction_score=0.7,
            ),
            EventV1(
                id="event_0002",
                t0=10.0,
                t1=20.0,
                place_segment_id="place_0002",
                place_segment_conf=0.7,
                place_segment_reason="heuristic_merge",
                interaction_signature={"contact_rate": 0.1, "contact_burst_count": 1, "active_object_top1": "phone"},
                interaction_primary_object="phone",
                interaction_score=0.2,
            ),
        ],
    )
    json_path = tmp_path / "demo_v03_decisions.json"
    if hasattr(output, "model_dump_json"):
        text = output.model_dump_json(indent=2)  # type: ignore[attr-defined]
    else:
        text = output.json(indent=2)
    json_path.write_text(text, encoding="utf-8")

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "place_interaction_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--n",
        "1",
        "--seed",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr

    place_path = out_dir / "place_segments.json"
    interaction_csv = out_dir / "interaction_summary.csv"
    report_md = out_dir / "report.md"
    snapshot_path = out_dir / "snapshot.json"
    assert place_path.exists()
    assert interaction_csv.exists()
    assert report_md.exists()
    assert snapshot_path.exists()

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert "cfg_hash" in snapshot
    assert "outputs" in snapshot
