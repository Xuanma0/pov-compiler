from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_min_output(path: Path) -> None:
    payload = {
        "video_id": "snap_demo",
        "meta": {"duration_s": 6.0},
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "events_v1": [],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_model_smoke_snapshot_redacted(tmp_path: Path) -> None:
    in_json = tmp_path / "input_v03_decisions.json"
    out_dir = tmp_path / "out"
    _write_min_output(in_json)
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
    snap_text = (out_dir / "snapshot.json").read_text(encoding="utf-8").lower()
    banned = ["api_key", "authorization", "bearer ", "aiza", "?key="]
    for token in banned:
        assert token not in snap_text
