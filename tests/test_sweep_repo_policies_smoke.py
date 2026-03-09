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


def _output(video_id: str) -> Output:
    return Output(
        video_id=video_id,
        meta={"duration_s": 20.0},
        events=[
            Event(id="event_1", t0=0.0, t1=10.0, scores={"boundary_conf": 0.7}, meta={"label": "navigation"}),
            Event(id="event_2", t0=10.0, t1=20.0, scores={"boundary_conf": 0.8}, meta={"label": "interaction-heavy"}),
        ],
        events_v1=[
            EventV1(id="ev1_1", t0=0.0, t1=10.0, label="navigation", place_segment_id="p1", interaction_score=0.1),
            EventV1(
                id="ev1_2",
                t0=10.0,
                t1=20.0,
                label="interaction-heavy",
                place_segment_id="p2",
                interaction_primary_object="door",
                interaction_score=0.7,
            ),
        ],
        decision_points=[
            DecisionPoint(
                id="dp_1",
                t=12.0,
                t0=11.8,
                t1=12.4,
                source_event="event_2",
                action={"type": "ATTENTION_TURN_HEAD"},
                conf=0.75,
            )
        ],
    )


def test_sweep_repo_policies_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    j1 = json_dir / "000a3525-6c98-4650-aaab-be7d2c7b9402_v03_decisions.json"
    j1.write_text(json.dumps(_output("000a3525-6c98-4650-aaab-be7d2c7b9402").model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    uids_file = tmp_path / "uids.txt"
    uids_file.write_text("000a3525-6c98-4650-aaab-be7d2c7b9402\n", encoding="utf-8")
    out_dir = tmp_path / "repo_sweep_out"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_repo_policies.py"),
        "--pov-json-dir",
        str(json_dir),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,60/200/12",
        "--write-policies",
        "fixed_interval",
        "--read-policies",
        "budgeted_topk",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_metrics_csv=" in proc.stdout
    assert "saved_snapshot=" in proc.stdout
    assert (out_dir / "aggregate" / "metrics_by_setting.csv").exists()
    assert (out_dir / "aggregate" / "metrics_by_setting.md").exists()
    assert (out_dir / "best_report.md").exists()
    assert (out_dir / "figures" / "fig_repo_quality_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_quality_vs_budget_seconds.pdf").exists()
    assert (out_dir / "figures" / "fig_repo_size_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_size_vs_budget_seconds.pdf").exists()
    assert (out_dir / "snapshot.json").exists()

