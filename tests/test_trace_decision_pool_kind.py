from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_trace_one_query_decision_pool_kind_visible(tmp_path: Path) -> None:
    payload = {
        "video_id": "trace_decision_demo",
        "meta": {
            "duration_s": 12.0,
            "decisions_model_provider": "fake",
            "decisions_model_name": "fake-v1",
            "decisions_model_base_url": "https://example.invalid/v1?key=***",
        },
        "events": [
            {"id": "event_0001", "t0": 0.0, "t1": 6.0, "scores": {}, "anchors": []},
            {"id": "event_0002", "t0": 6.0, "t1": 12.0, "scores": {}, "anchors": []},
        ],
        "highlights": [],
        "decision_points": [],
        "decisions_model_v1": [
            {
                "decision_type": "ATTENTION_TURN_HEAD",
                "t0_ms": 7000,
                "t1_ms": 8200,
                "conf": 0.88,
                "evidence": {"event_id": "event_0002", "span": "head turns"},
            }
        ],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    in_json = tmp_path / "input.json"
    out_dir = tmp_path / "trace_out"
    in_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "trace_one_query.py"),
        "--json",
        str(in_json),
        "--out_dir",
        str(out_dir),
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "decision_pool_kind=" in proc.stdout
    assert "decision_pool_count=" in proc.stdout

    report = (out_dir / "trace_report.md").read_text(encoding="utf-8")
    assert "decision_pool_kind" in report
    assert "decision_pool_count" in report
