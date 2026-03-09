from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.schemas import Alternative, Anchor, DecisionPoint, Event, KeyClip, Output, Token, TokenCodec


def _build_output() -> Output:
    highlights = [
        KeyClip(id="hl_0001", t0=4.5, t1=5.5, source_event="event_0001", anchor_type="turn_head", anchor_t=5.0, conf=0.8, meta={"anchor_types": ["turn_head"]}),
        KeyClip(id="hl_0002", t0=9.5, t1=10.5, source_event="event_0001", anchor_type="stop_look", anchor_t=10.0, conf=0.75, meta={"anchor_types": ["stop_look"]}),
        KeyClip(id="hl_0003", t0=14.5, t1=15.5, source_event="event_0002", anchor_type="turn_head", anchor_t=15.0, conf=0.82, meta={"anchor_types": ["turn_head"]}),
        KeyClip(id="hl_0004", t0=19.5, t1=20.5, source_event="event_0002", anchor_type="stop_look", anchor_t=20.0, conf=0.74, meta={"anchor_types": ["stop_look"]}),
        KeyClip(id="hl_0005", t0=24.5, t1=25.5, source_event="event_0003", anchor_type="turn_head", anchor_t=25.0, conf=0.83, meta={"anchor_types": ["turn_head"]}),
        KeyClip(id="hl_0006", t0=29.5, t1=30.5, source_event="event_0003", anchor_type="stop_look", anchor_t=30.0, conf=0.76, meta={"anchor_types": ["stop_look"]}),
    ]
    tokens = [
        Token(id="tok_0001", t0=4.5, t1=5.5, type="ATTENTION_TURN_HEAD", conf=0.8, source_event="event_0001"),
        Token(id="tok_0002", t0=9.5, t1=10.5, type="ATTENTION_STOP_LOOK", conf=0.7, source_event="event_0001"),
        Token(id="tok_0003", t0=8.0, t1=8.8, type="SCENE_CHANGE", conf=0.72, source_event="event_0001"),
        Token(id="tok_0004", t0=14.5, t1=15.5, type="ATTENTION_TURN_HEAD", conf=0.81, source_event="event_0002"),
        Token(id="tok_0005", t0=18.0, t1=18.8, type="SCENE_CHANGE", conf=0.73, source_event="event_0002"),
        Token(id="tok_0006", t0=24.5, t1=25.5, type="ATTENTION_TURN_HEAD", conf=0.82, source_event="event_0003"),
        Token(id="tok_0007", t0=29.5, t1=30.5, type="ATTENTION_STOP_LOOK", conf=0.72, source_event="event_0003"),
        Token(id="tok_0008", t0=31.0, t1=33.0, type="MOTION_MOVING", conf=0.65, source_event="event_0004"),
    ]
    decisions = [
        DecisionPoint(
            id="dp_0001",
            t=5.0,
            t0=4.5,
            t1=5.5,
            source_event="event_0001",
            source_highlight="hl_0001",
            action={"type": "ATTENTION_TURN_HEAD"},
            trigger={"anchor_types": ["turn_head"]},
            alternatives=[Alternative(action_type="LOOK_FORWARD_ONLY", rationale="r", expected_outcome="e", conf=0.6)],
            conf=0.8,
        ),
        DecisionPoint(
            id="dp_0002",
            t=10.0,
            t0=9.5,
            t1=10.5,
            source_event="event_0001",
            source_highlight="hl_0002",
            action={"type": "ATTENTION_STOP_LOOK"},
            trigger={"anchor_types": ["stop_look"]},
            alternatives=[Alternative(action_type="CONTINUE_MOVING", rationale="r", expected_outcome="e", conf=0.6)],
            conf=0.75,
        ),
        DecisionPoint(
            id="dp_0003",
            t=25.0,
            t0=24.5,
            t1=25.5,
            source_event="event_0003",
            source_highlight="hl_0005",
            action={"type": "ATTENTION_TURN_HEAD"},
            trigger={"anchor_types": ["turn_head"]},
            alternatives=[Alternative(action_type="LOOK_FORWARD_ONLY", rationale="r", expected_outcome="e", conf=0.6)],
            conf=0.81,
        ),
    ]
    return Output(
        video_id="stream_budget_demo",
        meta={"duration_s": 40.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=12.0, anchors=[Anchor(type="turn_head", t=5.0, conf=0.8), Anchor(type="stop_look", t=10.0, conf=0.7)]),
            Event(id="event_0002", t0=12.0, t1=22.0, anchors=[Anchor(type="turn_head", t=15.0, conf=0.8), Anchor(type="stop_look", t=20.0, conf=0.7)]),
            Event(id="event_0003", t0=22.0, t1=32.0, anchors=[Anchor(type="turn_head", t=25.0, conf=0.82), Anchor(type="stop_look", t=30.0, conf=0.74)]),
            Event(id="event_0004", t0=32.0, t1=40.0, anchors=[]),
        ],
        highlights=highlights,
        decision_points=decisions,
        token_codec=TokenCodec(version="0.2", vocab=[], tokens=tokens),
    )


def test_streaming_budget_smoke_fixed_policy(tmp_path: Path) -> None:
    repo_root = ROOT
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "out"
    payload = _build_output().model_dump()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "streaming_budget_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        "10",
        "--budgets",
        "20/50/4,40/100/8,60/200/12",
        "--budget-policy",
        "fixed",
        "--fixed-budget",
        "40/100/8",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "policy=fixed" in proc.stdout
    assert (out_dir / "steps.csv").exists()
    assert (out_dir / "queries.csv").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "snapshot.json").exists()

    headers = (out_dir / "queries.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "chosen_budget_key" in headers
    assert "trials_count" in headers


def test_streaming_budget_smoke_safety_latency_policy(tmp_path: Path) -> None:
    repo_root = ROOT
    json_path = tmp_path / "demo.json"
    out_dir = tmp_path / "out_safety_latency"
    payload = _build_output().model_dump()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "streaming_budget_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        "8",
        "--budgets",
        "20/50/4,40/100/8,60/200/12",
        "--policy",
        "safety_latency",
        "--latency-cap-ms",
        "5",
        "--max-trials-per-query",
        "3",
        "--query",
        "anchor=turn_head top_k=6",
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "policy=safety_latency" in proc.stdout

    headers = (out_dir / "queries.csv").read_text(encoding="utf-8").splitlines()[0]
    for col in (
        "policy_name",
        "trial_index",
        "action",
        "safety_reason",
        "latency_e2e_ms",
        "final_trial",
    ):
        assert col in headers

    for fig in (
        "fig_policy_budget_over_queries.png",
        "fig_policy_budget_over_queries.pdf",
        "fig_policy_safety_vs_latency.png",
        "fig_policy_safety_vs_latency.pdf",
    ):
        assert (out_dir / "figures" / fig).exists()


def test_streaming_budget_smoke_safety_latency_chain_policy(tmp_path: Path) -> None:
    repo_root = ROOT
    json_path = tmp_path / "demo_chain.json"
    out_dir = tmp_path / "out_chain"
    payload = _build_output().model_dump()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "streaming_budget_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        "8",
        "--budgets",
        "20/50/4,40/100/8,60/200/12",
        "--policy",
        "safety_latency_chain",
        "--latency-cap-ms",
        "5",
        "--max-trials-per-query",
        "3",
        "--query",
        "lost_object=door which=last top_k=6 then token=SCENE_CHANGE which=last top_k=6 chain_derive=time+object chain_object_mode=hard",
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "policy=safety_latency_chain" in proc.stdout
    assert "chain_queries_total=" in proc.stdout
    assert "chain_success_rate=" in proc.stdout

    headers = (out_dir / "queries.csv").read_text(encoding="utf-8").splitlines()[0]
    for col in (
        "is_chain",
        "chain_success",
        "chain_fail_reason",
        "chain_step1_budget_seconds",
        "chain_step2_budget_seconds",
        "chain_backoff_strategy",
        "chain_backoff_level",
        "chain_backoff_used",
        "chain_backoff_exhausted",
        "chain_backoff_reason",
    ):
        assert col in headers

    for fig in (
        "fig_streaming_chain_success_vs_budget_seconds.png",
        "fig_streaming_chain_success_vs_budget_seconds.pdf",
        "fig_streaming_chain_failure_attribution.png",
        "fig_streaming_chain_failure_attribution.pdf",
    ):
        assert (out_dir / "figures" / fig).exists()
