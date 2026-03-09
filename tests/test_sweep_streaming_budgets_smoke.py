from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, KeyClip, Output, Token, TokenCodec


def _build_output(uid: str) -> Output:
    return Output(
        video_id=uid,
        meta={"duration_s": 36.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=12.0, anchors=[Anchor(type="turn_head", t=5.0, conf=0.8)]),
            Event(id="event_0002", t0=12.0, t1=24.0, anchors=[Anchor(type="stop_look", t=17.0, conf=0.7)]),
            Event(id="event_0003", t0=24.0, t1=36.0, anchors=[Anchor(type="turn_head", t=28.0, conf=0.82)]),
        ],
        highlights=[
            KeyClip(id="hl_0001", t0=4.5, t1=5.5, source_event="event_0001", anchor_type="turn_head", anchor_t=5.0, conf=0.8, meta={"anchor_types": ["turn_head"]}),
            KeyClip(id="hl_0002", t0=16.5, t1=17.5, source_event="event_0002", anchor_type="stop_look", anchor_t=17.0, conf=0.75, meta={"anchor_types": ["stop_look"]}),
            KeyClip(id="hl_0003", t0=27.5, t1=28.5, source_event="event_0003", anchor_type="turn_head", anchor_t=28.0, conf=0.8, meta={"anchor_types": ["turn_head"]}),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=4.5, t1=5.5, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=10.0, t1=11.0, type="SCENE_CHANGE", conf=0.7, source_event="event_0001"),
                Token(id="tok_0003", t0=16.5, t1=17.5, type="ATTENTION_STOP_LOOK", conf=0.7, source_event="event_0002"),
                Token(id="tok_0004", t0=22.0, t1=22.8, type="SCENE_CHANGE", conf=0.72, source_event="event_0002"),
                Token(id="tok_0005", t0=27.5, t1=28.5, type="ATTENTION_TURN_HEAD", conf=0.79, source_event="event_0003"),
            ],
        ),
    )


def test_sweep_streaming_budgets_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    uid = "demo_uid"
    json_path = json_dir / f"{uid}_v03_decisions.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_build_output(uid).model_dump(), ensure_ascii=False), encoding="utf-8")
    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(uid + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_streaming_budgets.py"),
        "--json_dir",
        str(json_dir),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,40/100/8,60/200/12",
        "--step-s",
        "8",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "selection_mode=uids_file" in proc.stdout
    assert "saved_metrics_csv=" in proc.stdout

    assert (out_dir / "aggregate" / "metrics_by_budget.csv").exists()
    assert (out_dir / "aggregate" / "metrics_by_budget.md").exists()
    assert (out_dir / "snapshot.json").exists()
    assert (out_dir / "figures" / "fig_streaming_quality_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_streaming_quality_vs_budget_seconds.pdf").exists()
