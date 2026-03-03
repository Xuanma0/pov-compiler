from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, DecisionPoint, Event, EventV1, KeyClip, Output, Token, TokenCodec


def _make_output(uid: str) -> Output:
    return Output(
        video_id=uid,
        meta={"duration_s": 80.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=20.0, anchors=[Anchor(type="turn_head", t=9.0, conf=0.8)]),
            Event(id="event_0002", t0=20.0, t1=45.0, anchors=[]),
            Event(id="event_0003", t0=45.0, t1=70.0, anchors=[]),
        ],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=20.0, place_segment_id="place_0001", interaction_primary_object="door", interaction_score=0.7),
            EventV1(id="ev1_0002", t0=20.0, t1=45.0, place_segment_id="place_0002", interaction_primary_object="cup", interaction_score=0.2),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=8.5,
                t1=9.5,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=9.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=9.0, t1=9.2, type="SCENE_CHANGE", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=46.0, t1=46.2, type="SCENE_CHANGE", conf=0.8, source_event="event_0003"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=9.0,
                t0=8.8,
                t1=9.3,
                source_event="event_0001",
                source_highlight="hl_0001",
                action={"type": "ATTENTION_TURN_HEAD"},
                trigger={"anchor_types": ["turn_head"]},
                conf=0.85,
            )
        ],
    )


def test_run_planner_backend_compare_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    uid = "demo_uid"
    json_path = json_dir / f"{uid}_v03_decisions.json"
    json_path.write_text(json.dumps(_make_output(uid).model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(f"{uid}\n", encoding="utf-8")
    out_dir = tmp_path / "planner_compare"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_planner_backend_compare.py"),
        "--pov-json-dir",
        str(json_dir),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,40/100/8",
        "--queries-total",
        "3",
        "--seed",
        "0",
        "--planner-b-provider",
        "fake",
        "--planner-b-model",
        "fake-planner-v1",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_run_A=" in proc.stdout
    assert "saved_run_B=" in proc.stdout
    assert "saved_compare=" in proc.stdout
    assert "saved_table=" in proc.stdout

    compare_dir = out_dir / "compare"
    table_csv = compare_dir / "tables" / "table_planner_backend_compare.csv"
    table_md = compare_dir / "tables" / "table_planner_backend_compare.md"
    fig_delta = compare_dir / "figures" / "fig_planner_backend_delta.png"
    fig_tradeoff = compare_dir / "figures" / "fig_planner_backend_tradeoff.png"
    snapshot = compare_dir / "snapshot.json"
    commands = compare_dir / "commands.sh"
    readme = compare_dir / "README.md"

    assert table_csv.exists()
    assert table_md.exists()
    assert fig_delta.exists()
    assert fig_tradeoff.exists()
    assert snapshot.exists()
    assert commands.exists()
    assert readme.exists()

    with table_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    cols = list(rows[0].keys())
    expected_prefix = [
        "uid",
        "budget_key",
        "budget_seconds",
        "status_a",
        "status_b",
        "planner_a",
        "planner_b",
        "mrr_strict_a",
        "mrr_strict_b",
        "delta_mrr_strict",
    ]
    assert cols[: len(expected_prefix)] == expected_prefix
    assert "delta_planner_fallback_rate" in cols
