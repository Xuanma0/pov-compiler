from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval import model_planner as model_planner_mod
from pov_compiler.retrieval.planner_schema import PlannerPlan
from pov_compiler.retrieval.query_planner import plan_with_backend


def test_planner_backend_auto_falls_back_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    plan, meta = plan_with_backend(
        "lost_object=door top_k=6",
        planner_backend="auto",
        planner_model_cfg={
            "provider": "openai_compat",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
    )
    assert plan.intent in {"mixed", "token", "decision", "anchor", "time"}
    assert str(meta.get("planner_backend_used")) == "heuristic"
    assert isinstance(meta.get("planner_plan"), dict)


def test_planner_backend_model_fake_returns_structured_plan() -> None:
    plan, meta = plan_with_backend(
        "anchor=turn_head top_k=6",
        planner_backend="model",
        planner_model_cfg={
            "provider": "fake",
            "model": "fake-planner-v1",
        },
        seed=7,
    )
    assert str(meta.get("planner_backend_used")) == "model"
    planner_plan = dict(meta.get("planner_plan", {}))
    assert planner_plan
    assert str(planner_plan.get("normalized_query", "")).strip()
    assert plan.candidates


def test_planner_backend_model_parse_failure_uses_heuristic(monkeypatch) -> None:
    def _fake_plan_query_with_model(*args, **kwargs):
        return PlannerPlan(), {
            "fallback_reason": "json_parse_failed",
            "parse_ok": False,
            "provider": "openai_compat",
            "model": "demo",
            "cache": {"hit": 0, "miss": 1, "write_fail": 0},
        }

    monkeypatch.setattr(model_planner_mod, "plan_query_with_model", _fake_plan_query_with_model)
    plan, meta = plan_with_backend(
        "decision=ATTENTION_TURN_HEAD top_k=6",
        planner_backend="model",
        planner_model_cfg={
            "provider": "openai_compat",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
    )
    assert plan.candidates
    assert str(meta.get("planner_backend_used")) == "heuristic"
    assert str(meta.get("planner_fallback_reason")) == "json_parse_failed"
