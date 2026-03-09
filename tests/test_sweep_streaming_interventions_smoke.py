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
        video_id="stream_intervention_sweep_demo",
        meta={"duration_s": 30.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="turn_head", t=3.0, conf=0.8)]),
            Event(id="event_0002", t0=10.0, t1=20.0, anchors=[Anchor(type="stop_look", t=14.0, conf=0.7)]),
            Event(id="event_0003", t0=20.0, t1=30.0, anchors=[]),
        ],
        highlights=[
            KeyClip(id="hl_0001", t0=2.8, t1=3.8, source_event="event_0001", anchor_type="turn_head", anchor_t=3.0, conf=0.8),
            KeyClip(id="hl_0002", t0=13.8, t1=14.8, source_event="event_0002", anchor_type="stop_look", anchor_t=14.0, conf=0.7),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=2.5, t1=3.9, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=12.0, t1=13.0, type="SCENE_CHANGE", conf=0.7, source_event="event_0002"),
            ],
        ),
    )


def test_sweep_streaming_interventions_smoke(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "out"
    json_path.write_text(json.dumps(_build_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_streaming_interventions.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        "8",
        "--budgets",
        "20/50/4,60/200/12",
        "--trials",
        "3",
        "--seed",
        "0",
        "--query",
        "anchor=turn_head top_k=6",
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    assert (out_dir / "results_sweep.csv").exists()
    assert (out_dir / "best_config.yaml").exists()
    assert (out_dir / "best_report.md").exists()
    assert (out_dir / "snapshot.json").exists()

    for name in (
        "fig_objective_vs_latency.png",
        "fig_objective_vs_latency.pdf",
        "fig_pareto_frontier.png",
        "fig_pareto_frontier.pdf",
    ):
        assert (out_dir / "figures" / name).exists()

    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    inputs = snapshot.get("inputs", {})
    best = snapshot.get("best", {})
    assert int(inputs.get("trials", 0)) == 3
    assert str(inputs.get("search", "")) in {"random", "grid"}
    assert str(inputs.get("budgets", "")).strip()
    assert isinstance(inputs.get("queries"), list)
    assert str(best.get("cfg_hash", "")).strip()
