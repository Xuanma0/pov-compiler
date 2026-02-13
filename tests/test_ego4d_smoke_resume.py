from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ego4d_smoke import plan_stage_actions


def _write_pipeline_json(path: Path) -> None:
    payload = {
        "video_id": "demo",
        "events": [],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resume_skips_completed_stages(tmp_path: Path) -> None:
    json_path = tmp_path / "json" / "v.json"
    index_prefix = tmp_path / "cache" / "v"
    queries_path = tmp_path / "eval" / "v" / "queries.jsonl"
    eval_dir = tmp_path / "eval" / "v"
    nlq_dir = tmp_path / "nlq" / "v"
    bye_dir = tmp_path / "bye" / "v"

    _write_pipeline_json(json_path)
    Path(f"{index_prefix}.index.npz").parent.mkdir(parents=True, exist_ok=True)
    Path(f"{index_prefix}.index.npz").write_bytes(b"ok")
    Path(f"{index_prefix}.index_meta.json").write_text("{}", encoding="utf-8")
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.write_text("{\"qid\":\"q1\"}\n", encoding="utf-8")
    (eval_dir / "report.md").write_text("# ok", encoding="utf-8")
    nlq_dir.mkdir(parents=True, exist_ok=True)
    (nlq_dir / "nlq_report.md").write_text("# ok", encoding="utf-8")

    actions = plan_stage_actions(
        json_path=json_path,
        index_prefix=index_prefix,
        queries_path=queries_path,
        eval_dir=eval_dir,
        nlq_dir=nlq_dir,
        bye_dir=bye_dir,
        run_eval=True,
        run_nlq=True,
        run_bye=False,
        resume=True,
    )
    assert actions == {
        "run_offline": False,
        "build_index": False,
        "gen_queries": False,
        "eval_cross": False,
        "eval_nlq": False,
        "run_bye": False,
    }


def test_no_resume_forces_all_stages(tmp_path: Path) -> None:
    json_path = tmp_path / "json" / "v.json"
    index_prefix = tmp_path / "cache" / "v"
    queries_path = tmp_path / "eval" / "v" / "queries.jsonl"
    eval_dir = tmp_path / "eval" / "v"
    nlq_dir = tmp_path / "nlq" / "v"
    bye_dir = tmp_path / "bye" / "v"

    actions = plan_stage_actions(
        json_path=json_path,
        index_prefix=index_prefix,
        queries_path=queries_path,
        eval_dir=eval_dir,
        nlq_dir=nlq_dir,
        bye_dir=bye_dir,
        run_eval=True,
        run_nlq=True,
        run_bye=False,
        resume=False,
    )
    assert actions == {
        "run_offline": True,
        "build_index": True,
        "gen_queries": True,
        "eval_cross": True,
        "eval_nlq": True,
        "run_bye": False,
    }
