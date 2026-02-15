from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_pipeline_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events_v1": [
            {
                "id": "e1",
                "t0_ms": 0,
                "t1_ms": 1000,
                "place_segment_id": "p0",
                "interaction_score": 0.7,
                "interaction_primary_object": "door",
                "evidence": [{"t0_ms": 0, "t1_ms": 500}],
            }
        ],
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
        "object_memory_v0": [{"object_name": "door", "last_seen_t_ms": 1000}],
        "lost_object_queries": [{"query": "When did I last interact with the door?"}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_ab_bye_compare_auto_select_smoke(tmp_path: Path) -> None:
    root = tmp_path / "ego_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "demo.mp4").write_bytes(b"0")
    uid = hashlib.md5("demo.mp4".encode("utf-8")).hexdigest()

    out_dir = tmp_path / "ab_auto"
    for run_name in ("run_stub", "run_real"):
        _write_pipeline_json(out_dir / run_name / "json" / f"{uid}_v03_decisions.json", uid)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ab_bye_compare.py"),
        "--root",
        str(root),
        "--out_dir",
        str(out_dir),
        "--jobs",
        "1",
        "--n",
        "1",
        "--auto-select-uids",
        "--signal-min-score",
        "2",
        "--signal-top-k",
        "10",
        "--no-with-streaming-budget",
        "--no-with-figs",
        "--min-size-bytes",
        "0",
        "--probe-candidates",
        "0",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "selection_mode=auto_signal_cache" in proc.stdout
    assert "cache_built=true" in proc.stdout
    assert "selected_uids_count=" in proc.stdout
    assert "coverage_score_stats=" in proc.stdout

    selection_dir = out_dir / "compare" / "selection"
    assert (selection_dir / "coverage.csv").exists()
    assert (selection_dir / "selected_uids.txt").exists()
    assert (selection_dir / "selection_report.md").exists()

    compare_summary = out_dir / "compare" / "compare_summary.json"
    assert compare_summary.exists()
    payload = json.loads(compare_summary.read_text(encoding="utf-8"))
    assert payload.get("selection_mode") == "auto_signal_cache"
    assert payload.get("cache_built") is True
    assert int(payload.get("selected_uids_count", 0)) > 0
    assert isinstance(payload.get("coverage_score_stats"), dict)
    signal_cache_uid_dir = selection_dir / "signal_cache" / uid
    assert signal_cache_uid_dir.exists()
    assert (signal_cache_uid_dir / "events_v1_meta.json").exists()

    with (selection_dir / "coverage.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
