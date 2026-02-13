from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.exporter import export_bye_events_from_output_dict
from pov_compiler.integrations.bye.schema import validate_minimal


def _sample_output() -> dict[str, object]:
    return {
        "video_id": "demo_uid",
        "events_v1": [
            {
                "id": "ev1",
                "t0": 0.5,
                "t1": 2.0,
                "label": "interaction-heavy",
                "scores": {"boundary_conf": 0.7},
            }
        ],
        "highlights": [
            {
                "id": "hl1",
                "t0": 1.0,
                "t1": 1.8,
                "source_event": "ev1",
                "anchor_type": "turn_head",
                "anchor_t": 1.2,
                "conf": 0.8,
            }
        ],
        "token_codec": {
            "tokens": [
                {
                    "id": "tok1",
                    "t0": 1.1,
                    "t1": 1.6,
                    "type": "ATTENTION_TURN_HEAD",
                    "source_event": "ev1",
                    "conf": 0.75,
                }
            ]
        },
        "decision_points": [
            {
                "id": "dp1",
                "t": 1.3,
                "t0": 1.1,
                "t1": 1.7,
                "source_event": "ev1",
                "conf": 0.82,
                "trigger": {"anchor_type": "turn_head"},
                "state": {"motion_state_before": "MOVING"},
                "action": {"type": "ATTENTION_TURN_HEAD"},
                "constraints": [{"type": "STABILITY_CONSTRAINT", "score": 0.3}],
                "outcome": {"type": "MOTION_INCREASE"},
            }
        ],
    }


def test_export_bye_events_minimal_and_deterministic() -> None:
    payload = _sample_output()
    out1 = export_bye_events_from_output_dict(payload)
    out2 = export_bye_events_from_output_dict(payload)

    assert out1
    assert out1 == out2
    for row in out1:
        validate_minimal(row)

    keys = [int(row["tsMs"]) for row in out1]
    assert keys == sorted(keys)


def test_export_bye_events_cli_smoke(tmp_path: Path) -> None:
    json_path = tmp_path / "demo_v03_decisions.json"
    json_path.write_text(json.dumps(_sample_output(), ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = tmp_path / "bye_out"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_bye_events.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    jsonl_path = out_dir / "events" / "events_v1.jsonl"
    snapshot_path = out_dir / "snapshot.json"
    assert jsonl_path.exists()
    assert snapshot_path.exists()

    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 2
    rows = [json.loads(line) for line in lines]
    for row in rows:
        validate_minimal(row)

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert int(snapshot["counts_total"]) == len(rows)
    assert "counts_by_name" in snapshot
    assert snapshot["video_id"] == "demo_uid"

