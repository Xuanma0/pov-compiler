from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_signal_coverage_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    out_dir = tmp_path / "audit_out"
    perception_dir = tmp_path / "perception"
    uid = "dcc4514484c10342f3ce0ae9da0a529b"

    _write_json(
        json_dir / f"{uid}_v03_decisions.json",
        {
            "video_id": uid,
            "events_v1": [
                {
                    "id": "e1",
                    "t0_ms": 0,
                    "t1_ms": 1200,
                    "place_segment_id": "p0",
                    "interaction_score": 0.6,
                    "interaction_primary_object": "door",
                    "evidence": [{"t0_ms": 0, "t1_ms": 300}],
                }
            ],
            "object_memory_v0": [{"object_name": "door", "last_seen_t_ms": 1000}],
            "lost_object_queries": [{"query": "When did I last touch the door?"}],
        },
    )
    _write_json(
        perception_dir / uid / "perception.json",
        {
            "video_id": uid,
            "summary": {"frames_processed": 12, "contact_events_count": 3},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "audit_signal_coverage.py"),
        "--pov-json-dir",
        str(json_dir),
        "--out-dir",
        str(out_dir),
        "--auto-build-cache",
        "--cache-out",
        str(out_dir / "signal_cache"),
        "--perception-dir",
        str(perception_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_coverage_csv=" in proc.stdout
    assert "saved_uid_candidates=" in proc.stdout
    assert "saved_snapshot=" in proc.stdout

    coverage_csv = out_dir / "coverage.csv"
    uid_candidates = out_dir / "uid_candidates.txt"
    snapshot = out_dir / "snapshot.json"
    assert coverage_csv.exists()
    assert uid_candidates.exists()
    assert snapshot.exists()

    with coverage_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    required = {
        "uid",
        "events_v1_count",
        "interaction_events_count",
        "object_vocab_size",
        "lost_object_queries_total",
        "perception_contact_events",
        "coverage_score",
    }
    assert required.issubset(set(rows[0].keys()))
    assert "cache_used" in rows[0]
    assert "score_components" in rows[0]
    assert any(float(r.get("cache_used", 0) or 0) > 0 for r in rows)
    assert max(float(r.get("coverage_score", 0) or 0) for r in rows) >= 1.0
