from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_pov_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events_v1": [{"id": "ev1", "t0": 0.0, "t1": 5.0, "label": "idle", "scores": {"boundary_conf": 0.7}}],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sweep_bye_budgets_strict_uids_mismatch_fails(tmp_path: Path) -> None:
    pov_dir = tmp_path / "json"
    _write_pov_json(pov_dir / "000a3525-6c98-4650-aaab-be7d2c7b9402_v03_decisions.json", "000a3525-6c98-4650-aaab-be7d2c7b9402")
    uids_file = tmp_path / "uids_bad.txt"
    uids_file.write_text("not_exist_uid\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_bye_budgets.py"),
        "--pov-json-dir",
        str(pov_dir),
        "--uids-file",
        str(uids_file),
        "--out-dir",
        str(out_dir),
        "--budgets",
        "20/50/4",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode != 0
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert "uids_file_path" in combined
    assert "uids_missing_count" in combined

