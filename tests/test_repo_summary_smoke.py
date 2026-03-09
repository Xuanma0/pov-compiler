from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_min_output(path: Path) -> None:
    payload = {
        "video_id": "repo_summary_smoke",
        "meta": {"duration_s": 12.0},
        "events_v1": [
            {
                "id": "ev1",
                "t0": 0.0,
                "t1": 6.0,
                "label": "interaction-heavy",
                "source_event_ids": [],
                "evidence": [],
                "retrieval_hints": [],
                "scores": {"boundary_conf": 0.8},
                "place_segment_id": "p1",
                "interaction_primary_object": "door",
                "interaction_score": 0.7,
                "meta": {},
            }
        ],
        "events": [],
        "events_v0": [],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_repo_summary_smoke(tmp_path: Path) -> None:
    in_json = tmp_path / "demo_v03_decisions.json"
    out_dir = tmp_path / "out"
    _write_min_output(in_json)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "repo_summary_smoke.py"),
        "--json",
        str(in_json),
        "--out_dir",
        str(out_dir),
        "--provider",
        "fake",
        "--model",
        "fake-summary-v0",
        "--repo-write-policy",
        "multiscale+summary_v0",
        "--budget",
        "20/50/4",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_repo_summaries=" in proc.stdout
    assert (out_dir / "repo_chunks.jsonl").exists()
    assert (out_dir / "repo_selected.jsonl").exists()
    assert (out_dir / "repo_summaries.jsonl").exists()
    assert (out_dir / "context.txt").exists()
    assert (out_dir / "snapshot.json").exists()

