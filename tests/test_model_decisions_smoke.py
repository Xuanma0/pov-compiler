from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_min_output(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "meta": {"duration_s": 8.0, "fps": 30.0},
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 2.0, "scores": {}, "anchors": []}],
        "events_v1": [
            {
                "id": "ev1_0001",
                "t0": 0.0,
                "t1": 2.0,
                "label": "interaction-heavy",
                "source_event_ids": ["event_0001"],
                "evidence": [],
                "retrieval_hints": [],
                "scores": {},
                "interaction_primary_object": "door",
                "interaction_score": 0.8,
                "meta": {},
            }
        ],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_model_decisions_smoke_fake(tmp_path: Path) -> None:
    in_json = tmp_path / "demo_v03_decisions.json"
    out_dir = tmp_path / "out"
    _write_min_output(in_json, "demo")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "model_decisions_smoke.py"),
        "--json",
        str(in_json),
        "--out_dir",
        str(out_dir),
        "--provider",
        "fake",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_decisions=" in proc.stdout
    assert (out_dir / "decisions_model_v1.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "snapshot.json").exists()
