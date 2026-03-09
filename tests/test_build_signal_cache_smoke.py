from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_signal_cache_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    out_dir = tmp_path / "signal_cache"
    uid = "dcc4514484c10342f3ce0ae9da0a529b"
    _write_json(
        json_dir / f"{uid}_v03_decisions.json",
        {
            "video_id": uid,
            "events": [{"id": "e0", "t0": 0.0, "t1": 1.5, "label": "walk"}],
            "highlights": [],
            "decision_points": [],
            "token_codec": {"version": "0.2", "tokens": []},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_signal_cache.py"),
        "--pov-json-dir",
        str(json_dir),
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_cache=" in proc.stdout
    assert "saved_snapshot=" in proc.stdout

    uid_dir = out_dir / uid
    assert (uid_dir / "events_v1_meta.json").exists()
    assert (uid_dir / "place_interaction.json").exists()
    assert (uid_dir / "object_memory.json").exists()
    assert (uid_dir / "lost_object_queries.json").exists()
    assert (out_dir / "snapshot.json").exists()

