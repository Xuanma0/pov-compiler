from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Anchor, Event, Output


def _build_output(video_id: str) -> Output:
    return Output(
        video_id=video_id,
        meta={"duration_s": 24.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=8.0, anchors=[Anchor(type="turn_head", t=2.0, conf=0.8)]),
            Event(id="event_0002", t0=8.0, t1=16.0, anchors=[Anchor(type="stop_look", t=10.0, conf=0.7)]),
            Event(id="event_0003", t0=16.0, t1=24.0, anchors=[Anchor(type="turn_head", t=18.0, conf=0.85)]),
        ],
    )


def test_sweep_streaming_codec_k_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_id = "00000000-0000-0000-0000-000000000001"
    out = _build_output(video_id)
    payload = out.model_dump() if hasattr(out, "model_dump") else out.dict()
    (json_dir / f"{video_id}_v03_decisions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(video_id + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_streaming_codec_k.py"),
        "--json_dir",
        str(json_dir),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--k-list",
        "2,4",
        "--budgets",
        "20/50/4,40/100/8",
        "--step-s",
        "8",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "aggregate" / "metrics_by_k.csv").exists()
    assert (out_dir / "aggregate" / "metrics_by_k.md").exists()
    for fig in (
        "fig_streaming_quality_vs_k.png",
        "fig_streaming_quality_vs_k.pdf",
        "fig_streaming_safety_vs_k.png",
        "fig_streaming_safety_vs_k.pdf",
        "fig_streaming_latency_vs_k.png",
        "fig_streaming_latency_vs_k.pdf",
    ):
        assert (out_dir / "figures" / fig).exists()
