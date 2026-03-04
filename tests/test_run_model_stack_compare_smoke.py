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
        meta={"duration_s": 50.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=18.0, anchors=[Anchor(type="turn_head", t=6.0, conf=0.8)]),
            Event(id="event_0002", t0=18.0, t1=35.0, anchors=[]),
            Event(id="event_0003", t0=35.0, t1=48.0, anchors=[]),
        ],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=18.0, place_segment_id="place_0001", interaction_primary_object="door", interaction_score=0.7),
            EventV1(id="ev1_0002", t0=18.0, t1=35.0, place_segment_id="place_0002", interaction_primary_object="cup", interaction_score=0.2),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=5.5,
                t1=6.5,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=6.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=6.0, t1=6.2, type="SCENE_CHANGE", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=36.0, t1=36.2, type="SCENE_CHANGE", conf=0.8, source_event="event_0003"),
            ],
        ),
        decision_points=[
            DecisionPoint(
                id="dp_0001",
                t=6.0,
                t0=5.8,
                t1=6.3,
                source_event="event_0001",
                source_highlight="hl_0001",
                action={"type": "ATTENTION_TURN_HEAD"},
                trigger={"anchor_types": ["turn_head"]},
                conf=0.85,
            )
        ],
    )


def test_run_model_stack_compare_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    uid = "demo_uid_stack"
    (json_dir / f"{uid}_v03_decisions.json").write_text(
        json.dumps(_make_output(uid).model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_dir = tmp_path / "model_stack"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_model_stack_compare.py"),
        "--pov-json-dir",
        str(json_dir),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4",
        "--queries-total",
        "2",
        "--seed",
        "0",
        "--provider",
        "fake",
        "--mode",
        "hard_pseudo_nlq",
        "--auto-select-uids",
        "--signal-min-score",
        "0",
        "--signal-top-k",
        "5",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_run_A=" in proc.stdout
    assert "saved_run_D=" in proc.stdout
    assert "saved_compare=" in proc.stdout
    assert "saved_table=" in proc.stdout
    assert "saved_cost_table=" in proc.stdout

    compare_dir = out_dir / "compare"
    table_csv = compare_dir / "tables" / "table_model_stack_compare.csv"
    table_md = compare_dir / "tables" / "table_model_stack_compare.md"
    cost_csv = compare_dir / "tables" / "table_model_cost_compare.csv"
    cost_md = compare_dir / "tables" / "table_model_cost_compare.md"
    fig_delta = compare_dir / "figures" / "fig_model_stack_delta.png"
    fig_tradeoff = compare_dir / "figures" / "fig_model_stack_tradeoff.png"
    fig_cost = compare_dir / "figures" / "fig_model_cost_vs_quality.png"
    fig_parse = compare_dir / "figures" / "fig_model_parse_fail_rate.png"
    summary_json = compare_dir / "compare_summary.json"
    snapshot_json = compare_dir / "snapshot.json"
    commands = compare_dir / "commands.sh"
    readme = compare_dir / "README.md"
    selection_coverage = compare_dir / "selection" / "coverage.csv"
    selection_uids = compare_dir / "selection" / "selected_uids.txt"
    assert table_csv.exists()
    assert table_md.exists()
    assert cost_csv.exists()
    assert cost_md.exists()
    assert fig_delta.exists()
    assert fig_tradeoff.exists()
    assert fig_cost.exists()
    assert fig_parse.exists()
    assert summary_json.exists()
    assert snapshot_json.exists()
    assert commands.exists()
    assert readme.exists()
    assert selection_coverage.exists()
    assert selection_uids.exists()

    with table_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    cols = list(rows[0].keys())
    assert cols[:4] == ["budget_key", "budget_seconds", "variant_code", "variant_label"]
    assert "delta_mrr_vs_A" in cols
