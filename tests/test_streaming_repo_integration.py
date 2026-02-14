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

from pov_compiler.schemas import Anchor, Event, KeyClip, Output, Token, TokenCodec


def _build_output() -> Output:
    return Output(
        video_id="stream_repo_integration_demo",
        meta={"duration_s": 30.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="turn_head", t=3.0, conf=0.8)]),
            Event(id="event_0002", t0=10.0, t1=20.0, anchors=[Anchor(type="stop_look", t=14.0, conf=0.7)]),
            Event(id="event_0003", t0=20.0, t1=30.0, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.5,
                t1=3.5,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=3.0,
                conf=0.82,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=13.5,
                t1=14.5,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=14.0,
                conf=0.72,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=2.5, t1=3.5, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=13.5, t1=14.5, type="ATTENTION_STOP_LOOK", conf=0.72, source_event="event_0002"),
                Token(id="tok_0003", t0=18.0, t1=19.0, type="SCENE_CHANGE", conf=0.7, source_event="event_0002"),
            ],
        ),
        repository={
            "chunks": [
                {
                    "id": "chunk_repo_1",
                    "chunk_id": "chunk_repo_1",
                    "level": "decision",
                    "scale": "decision",
                    "t0": 2.4,
                    "t1": 3.6,
                    "text": "turn_head decision around doorway",
                    "importance": 0.9,
                    "tags": ["turn_head", "decision", "door"],
                    "source_ids": ["event_0001"],
                },
                {
                    "id": "chunk_repo_2",
                    "chunk_id": "chunk_repo_2",
                    "level": "event",
                    "scale": "event",
                    "t0": 13.0,
                    "t1": 15.0,
                    "text": "short stop_look near object",
                    "importance": 0.75,
                    "tags": ["stop_look", "interaction"],
                    "source_ids": ["event_0002"],
                },
            ],
            "cfg": {"read_policy": {"name": "query_aware"}},
        },
    )


def test_streaming_budget_smoke_with_repo_columns(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "out_repo"
    json_path.write_text(json.dumps(_build_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "streaming_budget_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        "8",
        "--budgets",
        "20/50/4,40/100/8",
        "--policy",
        "fixed",
        "--fixed-budget",
        "40/100/8",
        "--context-use-repo",
        "--repo-read-policy",
        "query_aware",
        "--query",
        "anchor=turn_head top_k=6",
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    queries_csv = out_dir / "queries.csv"
    assert queries_csv.exists()
    with queries_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    header = rows[0].keys()
    for col in (
        "context_use_repo",
        "repo_policy",
        "repo_selected_chunks",
        "repo_selected_by_level_json",
        "repo_cache_hit",
        "context_len_chars",
        "context_approx_tokens",
    ):
        assert col in header

    final_rows = [r for r in rows if int(float(r.get("final_trial", "0") or "0")) == 1]
    assert final_rows
    assert any(float(r.get("repo_selected_chunks", "0") or "0") > 0 for r in final_rows)

