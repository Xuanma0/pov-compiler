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

from pov_compiler.schemas import DecisionPoint, Event, EventV1, KeyClip, ObjectMemoryItemV0, Output, Token, TokenCodec


def _make_output() -> Output:
    return Output(
        video_id="demo_uid",
        meta={"duration_s": 90.0},
        events=[Event(id="event_0001", t0=0.0, t1=30.0), Event(id="event_0002", t0=30.0, t1=60.0)],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=30.0, place_segment_id="place_0001", interaction_primary_object="door", interaction_score=0.72),
            EventV1(id="ev1_0002", t0=30.0, t1=60.0, place_segment_id="place_0002", interaction_primary_object="cup", interaction_score=0.31),
        ],
        highlights=[
            KeyClip(id="hl_0001", t0=8.0, t1=10.0, source_event="event_0001", anchor_type="turn_head", anchor_t=9.0, conf=0.8, meta={"anchor_types": ["turn_head"]}),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=9.0, t1=9.3, type="SCENE_CHANGE", conf=0.9, source_event="event_0001"),
                Token(id="tok_0002", t0=33.0, t1=33.3, type="SCENE_CHANGE", conf=0.9, source_event="event_0002"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=9.0,
                t0=8.5,
                t1=9.2,
                source_event="event_0001",
                source_highlight="hl_0001",
                action={"type": "ATTENTION_TURN_HEAD"},
                trigger={"anchor_types": ["turn_head"]},
                conf=0.8,
            )
        ],
        object_memory_v0=[ObjectMemoryItemV0(object_name="door", last_seen_t_ms=56000, last_contact_t_ms=54000, last_place_id="place_0002", score=0.7)],
    )


def test_run_chain_attribution_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "demo_uid_v03_decisions.json"
    json_path.write_text(json.dumps(_make_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    uids = tmp_path / "uids.txt"
    uids.write_text("demo_uid\n", encoding="utf-8")
    out_dir = tmp_path / "chain_attr"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_chain_attribution.py"),
        "--pov-json-dir",
        str(json_dir),
        "--uids-file",
        str(uids),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,40/100/8",
        "--queries-total",
        "4",
        "--seed",
        "0",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_run_A=" in proc.stdout
    assert "saved_run_B=" in proc.stdout
    assert "saved_run_C=" in proc.stdout
    assert "saved_run_D=" in proc.stdout
    assert "saved_compare=" in proc.stdout
    assert "saved_table=" in proc.stdout

    compare_dir = out_dir / "compare"
    assert (compare_dir / "tables" / "table_chain_attribution.csv").exists()
    assert (compare_dir / "tables" / "table_chain_attribution.md").exists()
    assert (compare_dir / "tables" / "table_chain_failure_breakdown.csv").exists()
    assert (compare_dir / "tables" / "table_chain_failure_breakdown.md").exists()
    assert (compare_dir / "figures" / "fig_chain_attribution_success_vs_budget_seconds.png").exists()
    assert (compare_dir / "figures" / "fig_chain_attribution_delta_success_vs_budget_seconds.png").exists()
    assert (compare_dir / "snapshot.json").exists()
    assert (compare_dir / "commands.sh").exists()
    assert (compare_dir / "README.md").exists()

    with (compare_dir / "tables" / "table_chain_attribution.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "delta_success" in rows[0]

