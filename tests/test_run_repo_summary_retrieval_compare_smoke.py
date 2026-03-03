from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pov_compiler.schemas import Anchor, Event, EventV1, KeyClip, Output, Token, TokenCodec


def _make_output(video_id: str) -> Output:
    return Output(
        video_id=video_id,
        meta={"duration_s": 28.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=8.0, anchors=[Anchor(type="turn_head", t=3.0, conf=0.8)]),
            Event(id="event_0002", t0=8.0, t1=20.0, anchors=[Anchor(type="stop_look", t=11.0, conf=0.7)]),
            Event(id="event_0003", t0=20.0, t1=28.0, anchors=[]),
        ],
        events_v1=[
            EventV1(id="ev1_0001", t0=0.0, t1=8.0, place_segment_id="place_a", interaction_primary_object="door", interaction_score=0.8),
            EventV1(id="ev1_0002", t0=8.0, t1=20.0, place_segment_id="place_b", interaction_primary_object="cup", interaction_score=0.4),
            EventV1(id="ev1_0003", t0=20.0, t1=28.0, place_segment_id="place_b", interaction_primary_object="bag", interaction_score=0.3),
        ],
        highlights=[
            KeyClip(id="hl_0001", t0=2.6, t1=3.4, source_event="ev1_0001", anchor_type="turn_head", anchor_t=3.0, conf=0.8),
            KeyClip(id="hl_0002", t0=10.8, t1=11.3, source_event="ev1_0002", anchor_type="stop_look", anchor_t=11.0, conf=0.7),
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=1.0, t1=1.1, type="SCENE_CHANGE", conf=0.8, source_event="ev1_0001"),
                Token(id="tok_0002", t0=9.0, t1=9.1, type="SCENE_CHANGE", conf=0.8, source_event="ev1_0002"),
                Token(id="tok_0003", t0=22.0, t1=22.1, type="MOTION_STILL", conf=0.8, source_event="ev1_0003"),
            ],
        ),
    )


def test_run_repo_summary_retrieval_compare_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    output = _make_output("000a3525-6c98-4650-aaab-be7d2c7b9402")
    json_path = json_dir / "000a3525-6c98-4650-aaab-be7d2c7b9402_v03_decisions.json"
    json_path.write_text(json.dumps(output.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    uids = tmp_path / "uids.txt"
    uids.write_text("000a3525-6c98-4650-aaab-be7d2c7b9402\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_repo_summary_retrieval_compare.py"),
        "--pov-json-dir",
        str(json_dir),
        "--uids-file",
        str(uids),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,60/200/12",
        "--n",
        "2",
        "--top-k",
        "4",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_run_A=" in proc.stdout
    assert "saved_run_B=" in proc.stdout
    assert "saved_compare=" in proc.stdout

    compare = out_dir / "compare"
    assert (compare / "tables" / "table_repo_summary_retrieval_compare.csv").exists()
    assert (compare / "tables" / "table_repo_summary_retrieval_compare.md").exists()
    assert (compare / "figures" / "fig_repo_summary_retrieval_quality_vs_budget_seconds.png").exists()
    assert (compare / "figures" / "fig_repo_summary_retrieval_delta.png").exists()
    assert (compare / "compare_summary.json").exists()
    assert (compare / "snapshot.json").exists()
    assert (compare / "commands.sh").exists()
    assert (compare / "README.md").exists()

    summary = json.loads((compare / "compare_summary.json").read_text(encoding="utf-8"))
    assert "uids_total" in summary
    assert "budgets" in summary
    assert "per_budget" in summary
    if summary.get("per_budget"):
        first = summary["per_budget"][0]
        assert "stage0_hit_rate_a" in first
        assert "stage1_candidate_reduction_ratio_b" in first
