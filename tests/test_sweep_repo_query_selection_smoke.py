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


def _build_output(video_id: str) -> Output:
    return Output(
        video_id=video_id,
        meta={"duration_s": 18.0},
        events=[
            Event(id="event_0001", t0=0.0, t1=6.0, anchors=[Anchor(type="turn_head", t=2.0, conf=0.8)], meta={"label": "navigation"}),
            Event(id="event_0002", t0=6.0, t1=12.0, anchors=[Anchor(type="stop_look", t=8.0, conf=0.75)], meta={"label": "interaction-heavy"}),
            Event(id="event_0003", t0=12.0, t1=18.0, anchors=[Anchor(type="turn_head", t=14.0, conf=0.82)], meta={"label": "idle"}),
        ],
        highlights=[
            KeyClip(
                id="hl_0001",
                t0=7.6,
                t1=8.4,
                source_event="event_0002",
                anchor_type="stop_look",
                anchor_t=8.0,
                conf=0.82,
                meta={"anchor_types": ["stop_look"]},
            )
        ],
        token_codec=TokenCodec(
            version="0.2",
            vocab=[],
            tokens=[
                Token(id="tok_0001", t0=7.6, t1=8.4, type="ATTENTION_STOP_LOOK", conf=0.8, source_event="event_0002"),
                Token(id="tok_0002", t0=11.0, t1=11.6, type="SCENE_CHANGE", conf=0.75, source_event="event_0002"),
            ],
        ),
    )


def test_sweep_repo_query_selection_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    video_id = "00000000-0000-0000-0000-000000000099"
    output = _build_output(video_id)
    payload = output.model_dump() if hasattr(output, "model_dump") else output.dict()
    (json_dir / f"{video_id}_v03_decisions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    uids = tmp_path / "uids.txt"
    uids.write_text(video_id + "\n", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    qrows = [
        {
            "qid": "q_000001",
            "type": "decision",
            "query": "decision=ATTENTION_STOP_LOOK top_k=6",
            "top_k": 6,
            "relevant": {"events": ["event_0002"], "highlights": ["hl_0001"], "decisions": [], "tokens": ["tok_0001"]},
            "meta": {},
        },
        {
            "qid": "q_000002",
            "type": "token",
            "query": "token=SCENE_CHANGE top_k=6",
            "top_k": 6,
            "relevant": {"events": ["event_0002"], "highlights": [], "decisions": [], "tokens": ["tok_0002"]},
            "meta": {},
        },
    ]
    queries.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in qrows) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_repo_query_selection.py"),
        "--pov-json-dir",
        str(json_dir),
        "--uids-file",
        str(uids),
        "--queries-file",
        str(queries),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,60/200/12",
        "--policies",
        "baseline,query_aware",
        "--top-k",
        "6",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    metrics_csv = out_dir / "aggregate" / "metrics_by_policy_budget.csv"
    metrics_md = out_dir / "aggregate" / "metrics_by_policy_budget.md"
    assert metrics_csv.exists()
    assert metrics_md.exists()
    assert (out_dir / "best_report.md").exists()
    assert (out_dir / "snapshot.json").exists()
    for fig in (
        "fig_repo_query_selection_quality_vs_budget.png",
        "fig_repo_query_selection_quality_vs_budget.pdf",
        "fig_repo_query_selection_distractor_vs_budget.png",
        "fig_repo_query_selection_distractor_vs_budget.pdf",
        "fig_repo_query_selection_chunks_by_level.png",
        "fig_repo_query_selection_chunks_by_level.pdf",
    ):
        assert (out_dir / "figures" / fig).exists()

    header = metrics_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "mrr_strict" in header
    assert "top1_in_distractor_rate" in header
    assert "selected_chunks_count" in header
