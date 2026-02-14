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
        video_id="stream_intervention_demo",
        meta={"duration_s": 32.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=10.0, anchors=[Anchor(type="turn_head", t=3.0, conf=0.8)]),
            Event(id="event_0002", t0=10.0, t1=20.0, anchors=[Anchor(type="stop_look", t=14.0, conf=0.7)]),
            Event(id="event_0003", t0=20.0, t1=32.0, anchors=[]),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=2.5,
                t1=3.8,
                source_event="event_0001",
                anchor_type="turn_head",
                anchor_t=3.0,
                conf=0.8,
                meta={"anchor_types": ["turn_head"]},
            ),
            KeyClip(
                id="hl_0002",
                t0=13.5,
                t1=15.0,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=14.0,
                conf=0.7,
                meta={"anchor_types": ["stop_look"]},
            ),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=2.0, t1=4.0, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
                Token(id="tok_0002", t0=12.0, t1=13.0, type="SCENE_CHANGE", conf=0.7, source_event="event_0002"),
            ],
        ),
    )


def test_streaming_intervention_policy_smoke(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "out"
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
        "--policy",
        "safety_latency_intervention",
        "--budgets",
        "20/50/4,60/200/12,120/400/24",
        "--max-trials",
        "5",
        "--query",
        "anchor=turn_head top_k=6",
        "--query",
        "token=TURN_HEAD top_k=6",
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "policy=safety_latency_intervention" in proc.stdout

    queries_csv = out_dir / "queries.csv"
    assert queries_csv.exists()
    lines = queries_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 1
    header = lines[0]
    for col in (
        "trial_idx",
        "action",
        "attribution",
        "budget_before",
        "budget_after",
        "config_before",
        "config_after",
        "success",
        "final_trial",
    ):
        assert col in header

    # At least one query should trigger intervention retries.
    # robust parse with csv module
    import csv

    with queries_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    trial_idx_values = [int(r.get("trial_idx", 0) or 0) for r in rows]
    assert max(trial_idx_values) > 1

    for name in (
        "fig_policy_interventions_over_queries.png",
        "fig_policy_interventions_over_queries.pdf",
        "fig_policy_intervention_breakdown.png",
        "fig_policy_intervention_breakdown.pdf",
    ):
        assert (out_dir / "figures" / name).exists()

    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    cfg = snapshot.get("inputs", {})
    assert cfg.get("budget_policy") == "safety_latency_intervention"
    assert int(cfg.get("max_trials_per_query", 0)) == 5
    assert isinstance(cfg.get("budgets"), list)
