from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "meta": {"duration_s": 20.0},
        "events_v1": [
            {
                "id": "ev1",
                "t0": 0.0,
                "t1": 10.0,
                "label": "interaction-heavy",
                "source_event_ids": [],
                "evidence": [],
                "retrieval_hints": [],
                "scores": {"boundary_conf": 0.7},
                "place_segment_id": "p1",
                "interaction_primary_object": "door",
                "interaction_score": 0.8,
                "meta": {},
            },
            {
                "id": "ev2",
                "t0": 10.0,
                "t1": 20.0,
                "label": "navigation",
                "source_event_ids": [],
                "evidence": [],
                "retrieval_hints": [],
                "scores": {"boundary_conf": 0.6},
                "place_segment_id": "p2",
                "interaction_primary_object": "",
                "interaction_score": 0.2,
                "meta": {},
            },
        ],
        "events": [],
        "events_v0": [],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_sweep_repo_summary_budgets_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    p1 = json_dir / "uid_a_v03_decisions.json"
    _write_json(p1, "uid_a")
    uids_file = tmp_path / "uids.txt"
    uids_file.write_text("uid_a\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_repo_summary_budgets.py"),
        "--pov-json-dir",
        str(json_dir),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,60/200/12",
        "--provider",
        "fake",
        "--model",
        "fake-summary-v0",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_metrics_csv=" in proc.stdout
    assert (out_dir / "aggregate" / "metrics_by_policy_budget.csv").exists()
    assert (out_dir / "aggregate" / "metrics_by_policy_budget.md").exists()
    assert (out_dir / "figures" / "fig_repo_summary_quality_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_summary_size_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_summary_delta_vs_budget_seconds.png").exists()
    assert (out_dir / "snapshot.json").exists()

