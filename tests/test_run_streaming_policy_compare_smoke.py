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
        video_id="stream_policy_compare_demo",
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
                Token(id="tok_0002", t0=14.0, t1=16.0, type="ATTENTION_STOP_LOOK", conf=0.7, source_event="event_0002"),
            ],
        ),
    )


def test_run_streaming_policy_compare_smoke(tmp_path: Path) -> None:
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "out"
    json_path.write_text(json.dumps(_build_output().model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_streaming_policy_compare.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        "8",
        "--budgets",
        "20/50/4,60/200/12,120/400/24",
        "--max-trials",
        "5",
        "--intervention-cfg",
        str(ROOT / "configs" / "streaming_intervention_default.yaml"),
        "--query",
        "anchor=turn_head top_k=6",
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    compare_dir = out_dir / "compare"
    table_csv = compare_dir / "tables" / "table_streaming_policy_compare.csv"
    table_md = compare_dir / "tables" / "table_streaming_policy_compare.md"
    summary_json = compare_dir / "compare_summary.json"
    snapshot_json = compare_dir / "snapshot.json"
    readme = compare_dir / "README.md"
    commands = compare_dir / "commands.sh"

    assert table_csv.exists()
    assert table_md.exists()
    assert summary_json.exists()
    assert snapshot_json.exists()
    assert readme.exists()
    assert commands.exists()

    for name in (
        "fig_streaming_policy_compare_safety_latency.png",
        "fig_streaming_policy_compare_safety_latency.pdf",
        "fig_streaming_policy_compare_delta.png",
        "fig_streaming_policy_compare_delta.pdf",
    ):
        assert (compare_dir / "figures" / name).exists()

    header = table_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    expected_prefix = [
        "policy_a",
        "policy_b",
        "strict_success_rate_a",
        "strict_success_rate_b",
        "delta_strict_success_rate",
        "critical_fn_rate_a",
        "critical_fn_rate_b",
        "delta_critical_fn_rate",
        "latency_p95_e2e_ms_a",
        "latency_p95_e2e_ms_b",
        "delta_latency_p95_e2e_ms",
        "avg_trials_per_query_a",
        "avg_trials_per_query_b",
        "delta_avg_trials_per_query",
    ]
    assert header[: len(expected_prefix)] == expected_prefix

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary.get("policy_a")
    assert summary.get("policy_b")
    assert "delta" in summary
    snap = json.loads(snapshot_json.read_text(encoding="utf-8"))
    inputs = snap.get("inputs", {})
    assert str(inputs.get("intervention_cfg", "")).strip()
    assert str(inputs.get("intervention_cfg_hash", "")).strip()
