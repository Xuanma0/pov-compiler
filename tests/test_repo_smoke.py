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


def _build_output() -> Output:
    return Output(
        video_id="repo_smoke_demo",
        meta={"duration_s": 30.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="turn_head", t=4.0, conf=0.8)], scores={"boundary_conf": 0.7}, meta={"label": "navigation"}),
            Event(id="event_0002", t0=10.0, t1=20.0, anchors=[Anchor(type="stop_look", t=14.0, conf=0.72)], scores={"boundary_conf": 0.75}, meta={"label": "interaction-heavy"}),
            Event(id="event_0003", t0=20.0, t1=30.0, anchors=[Anchor(type="turn_head", t=24.0, conf=0.79)], scores={"boundary_conf": 0.66}, meta={"label": "idle"}),
        ],
        highlights=[
            KeyClip(id="hl_0001", t0=13.5, t1=14.5, source_event="event_0002", anchor_type="stop_look", anchor_t=14.0, conf=0.8, meta={"anchor_types": ["stop_look"]}),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=8.8, t1=9.4, type="SCENE_CHANGE", conf=0.71, source_event="event_0001"),
                Token(id="tok_0002", t0=13.5, t1=14.5, type="ATTENTION_STOP_LOOK", conf=0.8, source_event="event_0002"),
            ],
        ),
    )


def test_repo_smoke_outputs(tmp_path: Path) -> None:
    json_path = tmp_path / "demo_v03_decisions.json"
    out_dir = tmp_path / "repo_out"
    json_path.write_text(json.dumps(_build_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "repo_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--query",
        "anchor=turn_head top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_repo_chunks=" in proc.stdout
    assert (out_dir / "repo_chunks.jsonl").exists()
    assert (out_dir / "report.md").exists()
    snap = out_dir / "snapshot.json"
    assert snap.exists()
    payload = json.loads(snap.read_text(encoding="utf-8"))
    assert "cfg" in payload
    assert "stats" in payload
    assert "chunks_before_dedup" in payload["stats"]
    assert "chunks_after_dedup" in payload["stats"]
