from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from pov_compiler.models import ModelClientConfig, get_model_cache_stats, make_client
from pov_compiler.models.client import redact_url
from pov_compiler.models.presets import normalize_provider
from pov_compiler.models.structured_output import generate_structured
from pov_compiler.retrieval.planner_schema import PlannerPlan
from pov_compiler.retrieval.query_parser import parse_query

_PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "retrieval_plan": {
            "type": "string",
            "enum": [
                "direct",
                "baseline",
                "summary_then_token",
                "summary_then_decision",
                "summary_then_token_then_decision",
                "summary_then_event",
            ],
        },
        "normalized_query": {"type": "string"},
        "constraints": {"type": "object"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "notes": {"type": "string"},
    },
    "required": ["retrieval_plan", "normalized_query", "constraints", "confidence"],
}


def _stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]


def _sanitize_constraints(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    allowed = {
        "place",
        "place_segment_id",
        "place_segment_ids",
        "which",
        "interaction_min",
        "contact_min",
        "interaction_object",
        "object_name",
        "lost_object",
        "object_last_seen",
        "need_object_match",
        "prefer_contact",
        "anchor_type",
        "decision_type",
        "token_type",
        "event_label",
        "after_scene_change",
        "chain_rel",
        "chain_window_s",
        "chain_top1_only",
        "chain_derive",
        "chain_place_mode",
        "chain_object_mode",
        "chain_time_mode",
        "chain_time_min_s",
        "chain_time_max_s",
        "chain_place_value",
        "chain_object_value",
        "time",
        "time_range",
        "top_k",
    }
    for key, value in dict(raw or {}).items():
        k = str(key).strip()
        if not k:
            continue
        if "key" in k.lower() or "token" in k.lower() and k.lower() in {"api_token", "auth_token"}:
            continue
        if k not in allowed:
            continue
        out[k] = value
    return out


def _build_signal_overview(signal_overview: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(signal_overview or {})
    keep_keys = {
        "has_place",
        "has_interaction",
        "has_lost_object",
        "has_summary_chunks",
        "has_decision_pool",
        "events_v1_count",
        "object_vocab_size",
    }
    return {k: src.get(k) for k in keep_keys if k in src}


def _fake_planner_plan(query_text: str, *, seed: int = 0) -> PlannerPlan:
    q = str(query_text or "").strip()
    parsed = parse_query(q)
    constraints: dict[str, Any] = {}
    if parsed.place is not None:
        constraints["place"] = parsed.place
    if parsed.which is not None:
        constraints["which"] = parsed.which
    if parsed.interaction_min is not None:
        constraints["interaction_min"] = float(parsed.interaction_min)
    if parsed.interaction_object:
        constraints["interaction_object"] = str(parsed.interaction_object)
        constraints.setdefault("object_name", str(parsed.interaction_object))
        constraints["need_object_match"] = True
    if parsed.object_name:
        constraints["object_name"] = str(parsed.object_name)
        constraints["need_object_match"] = True
    if parsed.lost_object:
        constraints["lost_object"] = str(parsed.lost_object)
        constraints["which"] = "last"
        constraints["prefer_contact"] = True
    if parsed.token_types:
        constraints["token_type"] = str(parsed.token_types[0])
    if parsed.decision_types:
        constraints["decision_type"] = str(parsed.decision_types[0])
    if parsed.anchor_types:
        constraints["anchor_type"] = str(parsed.anchor_types[0])
    if parsed.chain_rel:
        constraints["chain_rel"] = str(parsed.chain_rel)

    h = int(hashlib.sha256(f"{seed}|{q}".encode("utf-8")).hexdigest()[:8], 16)
    if parsed.decision_types or "decision=" in q.lower():
        retrieval_plan = "summary_then_decision"
    elif " then " in q.lower() and (parsed.token_types or parsed.anchor_types):
        retrieval_plan = "summary_then_token_then_decision" if (h % 3 == 0) else "summary_then_token"
    elif parsed.token_types or parsed.anchor_types or parsed.lost_object or parsed.object_name:
        retrieval_plan = "summary_then_token"
    else:
        retrieval_plan = "baseline" if (h % 5 == 0) else "summary_then_token"

    normalized = q
    if "top_k=" not in normalized:
        normalized = f"{normalized} top_k=6".strip()
    return PlannerPlan(
        retrieval_plan=retrieval_plan,
        normalized_query=normalized,
        constraints=_sanitize_constraints(constraints),
        confidence=0.72,
        notes="fake_planner_deterministic",
    )


def plan_query_with_model(
    query_text: str,
    *,
    planner_model_cfg: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    signal_overview: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[PlannerPlan, dict[str, Any]]:
    cfg_raw = dict(planner_model_cfg or {})
    provider = normalize_provider(str(cfg_raw.get("provider", "fake")))
    model_name = str(cfg_raw.get("model", "fake-planner-v1"))
    cfg = ModelClientConfig(
        provider=provider,
        model=model_name,
        api_mode=str(cfg_raw.get("api_mode", "auto")),
        base_url=cfg_raw.get("base_url"),
        base_url_env=str(cfg_raw.get("base_url_env", "")),
        api_key_env=str(cfg_raw.get("api_key_env", "")),
        timeout_s=int(cfg_raw.get("timeout_s", 60)),
        max_tokens=int(cfg_raw.get("max_tokens", 500)),
        temperature=float(cfg_raw.get("temperature", 0.1)),
        extra_headers=dict(cfg_raw.get("extra_headers", {})) if isinstance(cfg_raw.get("extra_headers", {}), dict) else {},
        extra=dict(cfg_raw.get("extra", {})) if isinstance(cfg_raw.get("extra", {}), dict) else {},
        model_cache_enabled=bool(cfg_raw.get("model_cache_enabled", True)),
        model_cache_dir=str(cfg_raw.get("model_cache_dir", "data/outputs/model_cache")),
        model_cache_max_entries=int(cfg_raw.get("model_cache_max_entries", 0)),
        model_cache_max_mb=int(cfg_raw.get("model_cache_max_mb", 0)),
        max_retries=int(cfg_raw.get("max_retries", 1)),
    )
    api_key_present = bool(os.environ.get(cfg.api_key_env, "")) if cfg.provider != "fake" else True
    meta: dict[str, Any] = {
        "provider": str(cfg.provider),
        "model": str(cfg.model),
        "base_url": redact_url(str(cfg.base_url or "")),
        "api_key_env": str(cfg.api_key_env),
        "api_key_present": bool(api_key_present),
        "fallback_reason": "",
        "used_mode": "",
        "api_mode_used": "",
        "parse_ok": False,
        "cache": {},
        "parse_report": {},
        "planner_namespace": "planner_v1",
    }

    if cfg.provider != "fake" and not api_key_present:
        meta["fallback_reason"] = "no_api_key"
        return PlannerPlan(), meta

    if cfg.provider == "fake":
        fake_plan = _fake_planner_plan(query_text, seed=seed)
        client = make_client(cfg)
        meta["cache"] = get_model_cache_stats(client)
        meta["used_mode"] = "fake_rule"
        meta["parse_ok"] = True
        return fake_plan, meta

    client = make_client(cfg)
    meta["cache"] = get_model_cache_stats(client)

    user_payload = {
        "cache_namespace": "planner_v1",
        "query": str(query_text),
        "budget": dict(budget or {}),
        "signal_overview": _build_signal_overview(signal_overview),
    }
    system_prompt = (
        "You are a query planner. Convert query into deterministic retrieval plan JSON. "
        "Return JSON only. Do not output credentials or secrets. "
        "Use retrieval_plan from allowed values and keep constraints minimal/valid."
    )
    user_prompt = json.dumps(user_payload, ensure_ascii=False, sort_keys=True)

    try:
        obj, _raw, out_meta = generate_structured(
            client,
            schema_name="planner_plan_v1",
            schema_json=_PLANNER_SCHEMA,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(cfg.temperature),
            max_tokens=int(cfg.max_tokens),
            timeout_s=int(cfg.timeout_s),
        )
        if isinstance(out_meta, dict):
            meta["used_mode"] = str(out_meta.get("used_mode", ""))
            meta["api_mode_used"] = str(out_meta.get("api_mode_used", ""))
            meta["parse_ok"] = bool(out_meta.get("parse_ok", False))
            if isinstance(out_meta.get("parse_report", {}), dict):
                meta["parse_report"] = dict(out_meta.get("parse_report", {}))
            if out_meta.get("error"):
                meta["fallback_reason"] = str(out_meta.get("error"))
        plan_obj = PlannerPlan.from_obj(obj if isinstance(obj, dict) else {})
        plan_obj.constraints = _sanitize_constraints(plan_obj.constraints)
        if not plan_obj.normalized_query:
            plan_obj.normalized_query = str(query_text).strip()
            if "top_k=" not in plan_obj.normalized_query:
                plan_obj.normalized_query = f"{plan_obj.normalized_query} top_k=6".strip()
        if not plan_obj.constraints:
            parsed = parse_query(str(plan_obj.normalized_query))
            if parsed.interaction_object:
                plan_obj.constraints["interaction_object"] = str(parsed.interaction_object)
            if parsed.object_name:
                plan_obj.constraints["object_name"] = str(parsed.object_name)
        if plan_obj.notes:
            plan_obj.notes = _stable_hash(plan_obj.notes)
        return plan_obj, meta
    except Exception as exc:
        meta["fallback_reason"] = f"model_exception:{exc.__class__.__name__}"
        return PlannerPlan(), meta
