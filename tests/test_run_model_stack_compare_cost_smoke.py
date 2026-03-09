from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, EventV1, KeyClip, Output, Token, TokenCodec


def _make_output(uid: str) -> Output:
    return Output(
        video_id=uid,
        meta={"duration_s": 30.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=14.0, anchors=[Anchor(type="turn_head", t=4.0, conf=0.8)]),
            Event(id="event_0002", t0=14.0, t1=28.0, anchors=[]),
        ],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=14.0, place_segment_id="place_0001", interaction_primary_object="door", interaction_score=0.7),
            EventV1(id="ev1_0002", t0=14.0, t1=28.0, place_segment_id="place_0002", interaction_primary_object="cup", interaction_score=0.3),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=3.8,
                t1=4.5,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=4.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[Token(id="tok_0001", t0=4.0, t1=4.2, type="SCENE_CHANGE", conf=0.8, source_event="event_0001")],
        ),
    )


def test_run_model_stack_compare_cost_outputs(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    uid = "demo_uid_cost"
    (json_dir / f"{uid}_v03_decisions.json").write_text(
        json.dumps(_make_output(uid).model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_dir = tmp_path / "model_stack_cost"
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
    assert "saved_cost_table=" in proc.stdout
    assert "model_cost_stats=" in proc.stdout
    assert "gate_status=ok" in proc.stdout

    compare_dir = out_dir / "compare"
    assert (compare_dir / "tables" / "table_model_cost_compare.csv").exists()
    assert (compare_dir / "tables" / "table_model_cost_compare.md").exists()
    assert (compare_dir / "figures" / "fig_model_cost_vs_quality.png").exists()
    assert (compare_dir / "figures" / "fig_model_parse_fail_rate.png").exists()

    payload = json.loads((compare_dir / "compare_summary.json").read_text(encoding="utf-8"))
    assert "model_cost_stats" in payload
    assert isinstance(payload.get("model_cost_stats"), dict)
