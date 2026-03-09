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
        video_id="stream_repo_compare_demo",
        meta={"duration_s": 32.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="turn_head", t=3.2, conf=0.8)]),
            Event(id="event_0002", t0=10.0, t1=22.0, anchors=[Anchor(type="stop_look", t=14.5, conf=0.72)]),
            Event(id="event_0003", t0=22.0, t1=32.0, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=3.0,
                t1=4.0,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=3.2,
                conf=0.82,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=14.2,
                t1=15.0,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=14.5,
                conf=0.73,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=3.0, t1=4.0, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=14.2, t1=15.0, type="ATTENTION_STOP_LOOK", conf=0.72, source_event="event_0002"),
            ],
        ),
        repository={
            "chunks": [
                {
                    "id": "repo_chunk_1",
                    "chunk_id": "repo_chunk_1",
                    "level": "decision",
                    "scale": "decision",
                    "t0": 2.8,
                    "t1": 4.2,
                    "text": "turn_head decision around first place segment",
                    "importance": 0.9,
                    "tags": ["turn_head", "decision"],
                    "source_ids": ["event_0001"],
                },
                {
                    "id": "repo_chunk_2",
                    "chunk_id": "repo_chunk_2",
                    "level": "place",
                    "scale": "place",
                    "t0": 13.8,
                    "t1": 15.2,
                    "text": "stop_look in place segment two",
                    "importance": 0.8,
                    "tags": ["stop_look", "place"],
                    "source_ids": ["event_0002"],
                },
            ]
        },
    )


def test_run_streaming_repo_compare_smoke(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "compare_out"
    json_path.write_text(json.dumps(_build_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_streaming_repo_compare.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        "8",
        "--budgets",
        "20/50/4,60/200/12",
        "--policy-a",
        "safety_latency",
        "--policy-b",
        "safety_latency",
        "--a-use-repo",
        "false",
        "--b-use-repo",
        "true",
        "--b-repo-policy",
        "query_aware",
        "--query",
        "anchor=turn_head top_k=6",
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    compare_dir = out_dir / "compare"
    assert (compare_dir / "tables" / "table_streaming_repo_compare.csv").exists()
    assert (compare_dir / "tables" / "table_streaming_repo_compare.md").exists()
    assert (compare_dir / "compare_summary.json").exists()
    assert (compare_dir / "snapshot.json").exists()
    assert (compare_dir / "commands.sh").exists()
    assert (compare_dir / "README.md").exists()
    for name in (
        "fig_streaming_repo_compare_safety_latency.png",
        "fig_streaming_repo_compare_safety_latency.pdf",
        "fig_streaming_repo_compare_delta.png",
        "fig_streaming_repo_compare_delta.pdf",
    ):
        assert (compare_dir / "figures" / name).exists()

