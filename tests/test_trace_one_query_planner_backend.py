from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, KeyClip, Output


def _make_output() -> Output:
    return Output(
        video_id="planner_trace_demo",
        meta={"duration_s": 20.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="turn_head", t=3.0, conf=0.8)]),
            Event(id="event_0002", t0=10.0, t1=18.0, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.5,
                t1=3.5,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=3.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            )
        ],
    )


def test_trace_one_query_planner_backend_fields(tmp_path: Path) -> None:
    json_path = tmp_path / "planner_trace_demo_v03_decisions.json"
    json_path.write_text(json.dumps(_make_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    out_heur = tmp_path / "trace_heuristic"
    cmd_heur = [
        sys.executable,
        str(ROOT / "scripts" / "trace_one_query.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_heur),
        "--query",
        "anchor=turn_head top_k=4",
        "--planner-backend",
        "heuristic",
    ]
    proc_heur = subprocess.run(cmd_heur, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc_heur.returncode == 0, proc_heur.stderr or proc_heur.stdout
    assert "planner_backend_used=heuristic" in proc_heur.stdout

    out_model = tmp_path / "trace_model"
    cmd_model = [
        sys.executable,
        str(ROOT / "scripts" / "trace_one_query.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_model),
        "--query",
        "anchor=turn_head top_k=4",
        "--planner-backend",
        "model",
        "--planner-provider",
        "fake",
        "--planner-model",
        "fake-planner-v1",
    ]
    proc_model = subprocess.run(cmd_model, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc_model.returncode == 0, proc_model.stderr or proc_model.stdout
    assert "planner_backend_used=model" in proc_model.stdout
    assert "planner_plan=" in proc_model.stdout

    trace_payload = json.loads((out_model / "trace.json").read_text(encoding="utf-8"))
    assert "planner_backend_used" in trace_payload
    assert "planner_plan" in trace_payload
    assert "planner_fallback_reason" in trace_payload

    report_text = (out_model / "trace_report.md").read_text(encoding="utf-8")
    assert "## Planner" in report_text
    assert "planner_backend_used" in report_text
