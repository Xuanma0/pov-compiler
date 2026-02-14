from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_min_output(path: Path) -> None:
    payload = {
        "video_id": "demo_uid",
        "meta": {"duration_s": 30.0},
        "events": [
            {"id": "event_0001", "t0": 0.0, "t1": 10.0},
            {"id": "event_0002", "t0": 10.0, "t1": 20.0},
        ],
        "events_v1": [
            {
                "id": "ev1",
                "t0": 2.0,
                "t1": 4.0,
                "label": "interaction-heavy",
                "place_segment_id": "place_a",
                "interaction_primary_object": "door",
                "interaction_score": 0.8,
            },
            {
                "id": "ev2",
                "t0": 8.0,
                "t1": 10.0,
                "label": "navigation",
                "place_segment_id": "place_b",
                "interaction_primary_object": "phone",
                "interaction_score": 0.2,
            },
            {
                "id": "ev3",
                "t0": 12.0,
                "t1": 14.0,
                "label": "interaction-heavy",
                "place_segment_id": "place_c",
                "interaction_primary_object": "door",
                "interaction_score": 0.7,
            },
        ],
        "highlights": [
            {
                "id": "hl_0001",
                "t0": 1.0,
                "t1": 3.0,
                "source_event": "event_0001",
                "anchor_type": "turn_head",
                "anchor_t": 2.0,
                "conf": 0.9,
                "meta": {"anchor_types": ["turn_head"]},
            },
            {
                "id": "hl_0002",
                "t0": 11.0,
                "t1": 13.0,
                "source_event": "event_0002",
                "anchor_type": "stop_look",
                "anchor_t": 12.0,
                "conf": 0.8,
                "meta": {"anchor_types": ["stop_look"]},
            },
        ],
        "perception": {
            "frames": [
                {
                    "t": 0.5,
                    "objects": [{"label": "door"}],
                    "contact": {"active": None, "active_score": 0.0},
                },
                {
                    "t": 12.0,
                    "objects": [{"label": "door"}],
                    "contact": {"active": {"label": "door", "score": 0.9}, "active_score": 0.9},
                },
            ]
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_object_memory_smoke_outputs(tmp_path: Path) -> None:
    json_path = tmp_path / "demo_uid_v03_decisions.json"
    _write_min_output(json_path)

    fake_eval = tmp_path / "fake_eval_nlq.py"
    fake_eval.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import argparse",
                "p=argparse.ArgumentParser()",
                "p.add_argument('--out_dir', required=True)",
                "p.add_argument('--json')",
                "p.add_argument('--mode')",
                "p.add_argument('--seed')",
                "p.add_argument('--n')",
                "p.add_argument('--top-k')",
                "p.add_argument('--no-allow-gt-fallback', action='store_true')",
                "p.add_argument('--no-safety-gate', action='store_true')",
                "args=p.parse_args()",
                "out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)",
                "(out/'nlq_report.md').write_text('# fake nlq\\n', encoding='utf-8')",
                "(out/'nlq_summary.csv').write_text('variant,query_type,hit_at_k_strict\\nfull,hard_pseudo_lost_object,1\\n', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(Path("scripts") / "object_memory_smoke.py"),
        "--pov_json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--n",
        "1",
        "--seed",
        "0",
        "--eval-script",
        str(fake_eval),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "saved_object_memory=" in proc.stdout
    assert "saved_queries=" in proc.stdout
    assert "saved_nlq_report=" in proc.stdout
    assert "saved_snapshot=" in proc.stdout

    uid_dir = out_dir / "demo_uid"
    assert (uid_dir / "object_memory.json").exists()
    assert (uid_dir / "lost_object_queries.json").exists()
    assert (out_dir / "nlq_report.md").exists()
    assert (out_dir / "snapshot.json").exists()

