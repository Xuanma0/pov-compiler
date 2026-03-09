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


def _make_output() -> Output:
    return Output(
        video_id="trace_chain_demo",
        meta={"duration_s": 20.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=8.0, anchors=[Anchor(type="stop_look", t=3.0, conf=0.8)]),
            Event(id="event_0002", t0=8.0, t1=20.0, anchors=[Anchor(type="turn_head", t=12.0, conf=0.85)]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.0,
                t1=4.0,
                source_event="event_0001",
                anchor_type="stop_look",
                anchor_t=3.0,
                conf=0.8,
                meta={"anchor_types": ["stop_look"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=11.0,
                t1=13.0,
                source_event="event_0002",
                anchor_type="turn_head",
                anchor_t=12.0,
                conf=0.9,
                meta={"anchor_types": ["turn_head"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=2.0, t1=3.9, type="HIGHLIGHT", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=11.0, t1=13.0, type="HIGHLIGHT", conf=0.9, source_event="event_0002"),
            ],
        ),
    )


def test_trace_one_query_chain_sections(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "trace_out"
    json_path.write_text(json.dumps(_make_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "trace_one_query.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--query",
        "anchor=stop_look top_k=6 then token=HIGHLIGHT top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "is_chain=true" in proc.stdout
    assert "derived_constraints=" in proc.stdout

    report = (out_dir / "trace_report.md").read_text(encoding="utf-8")
    assert "## Chain Step 1" in report
    assert "## Derived Constraints" in report
    assert "## Chain Step 2" in report

