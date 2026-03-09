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
        video_id="stream_chain_backoff_cmp_demo",
        meta={"duration_s": 36.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=12.0, anchors=[Anchor(type="turn_head", t=3.5, conf=0.8)]),
            Event(id="event_0002", t0=12.0, t1=24.0, anchors=[Anchor(type="stop_look", t=15.5, conf=0.7)]),
            Event(id="event_0003", t0=24.0, t1=36.0, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=3.0,
                t1=4.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=3.5,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=15.0,
                t1=16.0,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=15.5,
                conf=0.7,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=2.5, t1=4.2, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=14.0, t1=16.0, type="SCENE_CHANGE", conf=0.7, source_event="event_0002"),
            ],
        ),
    )


def test_run_streaming_chain_backoff_compare_smoke(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "out"
    json_path.write_text(json.dumps(_build_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_streaming_chain_backoff_compare.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,60/200/12",
        "--step-s",
        "8",
        "--seed",
        "0",
        "--policies",
        "strict,ladder,adaptive",
        "--query",
        "lost_object=door which=last top_k=6 then token=SCENE_CHANGE which=last top_k=6 chain_derive=time+object chain_object_mode=hard",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    compare_dir = out_dir / "compare"
    assert (compare_dir / "tables" / "table_streaming_chain_backoff_compare.csv").exists()
    assert (compare_dir / "tables" / "table_streaming_chain_backoff_compare.md").exists()
    assert (compare_dir / "compare_summary.json").exists()
    assert (compare_dir / "snapshot.json").exists()
    assert (compare_dir / "commands.sh").exists()
    assert (compare_dir / "README.md").exists()

    for name in (
        "fig_streaming_chain_backoff_success_vs_budget_seconds.png",
        "fig_streaming_chain_backoff_success_vs_budget_seconds.pdf",
        "fig_streaming_chain_backoff_latency_vs_budget_seconds.png",
        "fig_streaming_chain_backoff_latency_vs_budget_seconds.pdf",
        "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds.png",
        "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds.pdf",
        "fig_streaming_chain_backoff_delta.png",
        "fig_streaming_chain_backoff_delta.pdf",
    ):
        assert (compare_dir / "figures" / name).exists()

