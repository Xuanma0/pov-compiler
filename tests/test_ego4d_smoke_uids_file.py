from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _uid_for_rel(rel: str) -> str:
    return hashlib.md5(rel.encode("utf-8")).hexdigest()


def _write_pipeline_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ego4d_smoke_uids_file_selection(tmp_path: Path) -> None:
    root = tmp_path / "ego_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.mp4").write_bytes(b"0")
    (root / "b.mp4").write_bytes(b"0")
    (root / "c.mp4").write_bytes(b"0")

    uid_a = _uid_for_rel("a.mp4")
    uid_b = _uid_for_rel("b.mp4")
    uid_missing = "ffffffffffffffffffffffffffffffff"

    out_dir = tmp_path / "out"
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    _write_pipeline_json(out_dir / "json" / f"{uid_a}_v03_decisions.json", uid_a)
    _write_pipeline_json(out_dir / "json" / f"{uid_b}_v03_decisions.json", uid_b)
    (out_dir / "cache" / f"{uid_a}.index.npz").write_bytes(b"NPZ")
    (out_dir / "cache" / f"{uid_a}.index_meta.json").write_text("{}", encoding="utf-8")
    (out_dir / "cache" / f"{uid_b}.index.npz").write_bytes(b"NPZ")
    (out_dir / "cache" / f"{uid_b}.index_meta.json").write_text("{}", encoding="utf-8")

    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(f"# demo\n{uid_a}\n{uid_missing}\n{uid_b}.mp4\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ego4d_smoke.py"),
        "--root",
        str(root),
        "--out_dir",
        str(out_dir),
        "--uids-file",
        str(uids_file),
        "--min-size-bytes",
        "0",
        "--probe-candidates",
        "0",
        "--no-proxy",
        "--no-run-eval",
        "--no-run-nlq",
        "--no-run-perception",
        "--jobs",
        "1",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "selection_mode=uids_file" in result.stdout

    summary_csv = out_dir / "summary.csv"
    assert summary_csv.exists()
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    row = rows[0]
    assert row.get("selection_mode") == "uids_file"
    assert int(float(row.get("uids_requested", "0"))) == 3
    assert int(float(row.get("uids_found", "0"))) == 2
    assert int(float(row.get("uids_missing_count", "0"))) == 1

