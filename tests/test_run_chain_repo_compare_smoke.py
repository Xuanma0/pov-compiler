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
        events=[
            Event(id="event_0001", t0=0.0, t1=30.0),
            Event(id="event_0002", t0=30.0, t1=60.0),
            Event(id="event_0003", t0=60.0, t1=90.0),
        ],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=30.0, place_segment_id="place_0001", interaction_primary_object="door", interaction_score=0.75),
            EventV1(id="ev1_0002", t0=30.0, t1=60.0, place_segment_id="place_0002", interaction_primary_object="phone", interaction_score=0.5),
            EventV1(id="ev1_0003", t0=60.0, t1=90.0, place_segment_id="place_0003", interaction_primary_object="door", interaction_score=0.4),
        ],
        highlights=[
            KeyClip(id="hl_0001", t0=10.0, t1=12.0, source_event="event_0001", anchor_type="turn_head", anchor_t=11.0, conf=0.8, meta={"anchor_types": ["turn_head"]}),
            KeyClip(id="hl_0002", t0=45.0, t1=47.0, source_event="event_0002", anchor_type="stop_look", anchor_t=46.0, conf=0.78, meta={"anchor_types": ["stop_look"]}),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=29.8, t1=30.2, type="SCENE_CHANGE", conf=0.9, source_event="event_0001"),
                Token(id="tok_0002", t0=59.8, t1=60.2, type="SCENE_CHANGE", conf=0.9, source_event="event_0002"),
                Token(id="tok_0003", t0=11.0, t1=12.0, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=11.0,
                t0=10.5,
                t1=11.5,
                source_event="event_0001",
                source_highlight="hl_0001",
                action={"type": "ATTENTION_TURN_HEAD"},
                trigger={"anchor_types": ["turn_head"]},
                conf=0.8,
            )
        ],
        object_memory_v0=[
            ObjectMemoryItemV0(object_name="door", last_seen_t_ms=82000, last_contact_t_ms=79000, last_place_id="place_0003", score=0.8),
        ],
    )


def test_run_chain_repo_compare_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "demo_uid_v03_decisions.json"
    json_path.write_text(json.dumps(_make_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    uids = tmp_path / "uids.txt"
    uids.write_text("demo_uid\n", encoding="utf-8")

    out_dir = tmp_path / "chain_repo_compare"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_chain_repo_compare.py"),
        "--pov-json-dir",
        str(json_dir),
        "--uids-file",
        str(uids),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,40/100/8",
        "--repo-policy",
        "query_aware",
        "--n",
        "4",
        "--seed",
        "0",
        "--top-k",
        "6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_compare=" in proc.stdout
    assert "saved_table=" in proc.stdout
    assert "saved_figures=" in proc.stdout

    table_csv = out_dir / "compare" / "tables" / "table_chain_repo_compare.csv"
    table_md = out_dir / "compare" / "tables" / "table_chain_repo_compare.md"
    summary_json = out_dir / "compare" / "compare_summary.json"
    snapshot_json = out_dir / "compare" / "snapshot.json"
    fig_success_png = out_dir / "compare" / "figures" / "fig_chain_repo_compare_success_vs_budget_seconds.png"
    fig_delta_png = out_dir / "compare" / "figures" / "fig_chain_repo_compare_delta.png"
    assert table_csv.exists()
    assert table_md.exists()
    assert summary_json.exists()
    assert snapshot_json.exists()
    assert fig_success_png.exists()
    assert fig_delta_png.exists()

    with table_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "delta_chain_success_rate" in rows[0]
