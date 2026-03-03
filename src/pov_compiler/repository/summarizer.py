from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from pov_compiler.models import ModelClientConfig, make_client
from pov_compiler.models.structured_output import generate_structured
from pov_compiler.repository.schema import RepoChunk
from pov_compiler.repository.summary_schema import RepoSummaryV0


def _summary_schema_json() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "video_id": {"type": "string"},
            "t0_ms": {"type": "integer"},
            "t1_ms": {"type": "integer"},
            "goals": {"type": "array", "items": {"type": "string"}},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "t0_ms": {"type": ["integer", "null"]},
                        "t1_ms": {"type": ["integer", "null"]},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "entities": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}},
            },
            "places": {"type": "array", "items": {"type": "string"}},
            "interactions": {
                "type": "array",
                "items": {"type": "object", "properties": {"object": {"type": "string"}, "score": {"type": ["number", "null"]}}},
            },
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"type": {"type": "string"}, "description": {"type": "string"}, "confidence": {"type": ["number", "null"]}},
                },
            },
            "notes": {"type": ["string", "null"]},
            "provenance": {"type": "object"},
        },
        "required": ["video_id", "t0_ms", "t1_ms"],
    }


def _render_prompt(data: dict[str, Any]) -> str:
    try:
        from jinja2 import Template  # type: ignore

        tmpl = Template(
            (
                "Create a concise repository summary JSON.\n"
                "Video: {{ video_id }} | window=[{{ t0_ms }},{{ t1_ms }}]\n"
                "Facts (sorted):\n"
                "{% for f in facts %}- id={{ f.id }} level={{ f.level }} t0={{ f.t0_ms }} t1={{ f.t1_ms }} text={{ f.text }}\n{% endfor %}\n"
                "Return only JSON."
            )
        )
        return str(tmpl.render(**data))
    except Exception:
        return (
            "Create a concise repository summary JSON. "
            f"Video={data.get('video_id','')} window=[{data.get('t0_ms',0)},{data.get('t1_ms',0)}]. "
            f"Facts={json.dumps(data.get('facts', []), ensure_ascii=False)}. Return only JSON."
        )


def _heuristic_summary(
    *,
    chunks: list[RepoChunk],
    video_id: str,
    t0_ms: int,
    t1_ms: int,
    policy_name: str,
    model_meta: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(chunks, key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
    places: list[str] = []
    objects: dict[str, int] = {}
    decisions: list[dict[str, Any]] = []
    for chunk in ordered:
        meta = dict(chunk.meta or {})
        place = str(meta.get("place_segment_id", "")).strip()
        if place and place not in places:
            places.append(place)
        obj = str(meta.get("primary_object", "") or meta.get("interaction_primary_object", "")).strip().lower()
        if obj:
            objects[obj] = objects.get(obj, 0) + 1
        if str(chunk.level) == "decision":
            decisions.append(
                {
                    "type": str(meta.get("action_type", "")),
                    "description": str(chunk.text)[:120],
                    "confidence": float(chunk.score_fields.get("decision_conf", 0.0)),
                }
            )
    top_objects = sorted(objects.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    steps: list[dict[str, Any]] = []
    for c in ordered[:5]:
        steps.append(
            {
                "title": str(c.text)[:120],
                "t0_ms": int(c.t0_ms),
                "t1_ms": int(c.t1_ms),
                "evidence_ids": [str(x) for x in (c.source_ids or [])[:4]],
            }
        )
    goals = []
    if top_objects:
        goals.append(f"interact_with_{top_objects[0][0]}")
    if decisions:
        goals.append("make_decisions")
    if not goals:
        goals.append("summarize_timeline")
    interactions = [{"object": name, "score": float(min(1.0, count / max(1.0, len(ordered)))), "verb": "interact"} for name, count in top_objects]
    entities = [{"name": name, "type": "object", "attributes": {"count": count}} for name, count in top_objects]
    return {
        "video_id": str(video_id),
        "t0_ms": int(t0_ms),
        "t1_ms": int(t1_ms),
        "goals": goals,
        "steps": steps,
        "entities": entities,
        "places": places[:5],
        "interactions": interactions,
        "decisions": decisions[:6],
        "notes": "heuristic_summary_fallback",
        "provenance": {
            "input_chunk_ids": [str(c.id) for c in ordered],
            "policy": str(policy_name),
            "model": dict(model_meta),
        },
    }


def _summary_text(summary: dict[str, Any]) -> str:
    goals = [str(x) for x in summary.get("goals", []) if str(x).strip()]
    places = [str(x) for x in summary.get("places", []) if str(x).strip()]
    interactions = [str((item or {}).get("object", "")) for item in summary.get("interactions", []) if isinstance(item, dict)]
    decisions = [str((item or {}).get("type", "")) for item in summary.get("decisions", []) if isinstance(item, dict)]
    return (
        f"Summary [{int(summary.get('t0_ms',0))/1000:.1f}-{int(summary.get('t1_ms',0))/1000:.1f}s]; "
        f"goals={','.join(goals[:3]) or 'none'}; "
        f"places={','.join(places[:4]) or 'none'}; "
        f"objects={','.join([x for x in interactions if x][:4]) or 'none'}; "
        f"decisions={','.join([x for x in decisions if x][:4]) or 'none'}."
    )


def summarize_chunks_to_repo_summary(
    chunks: list[RepoChunk],
    *,
    video_id: str,
    policy_name: str,
    provider_cfg: dict[str, Any] | None,
    budget_hint: dict[str, Any] | None = None,
) -> RepoChunk:
    ordered = sorted(chunks, key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
    if not ordered:
        raise ValueError("summarize_chunks_to_repo_summary requires non-empty chunks")
    t0_ms = int(min(c.t0_ms for c in ordered))
    t1_ms = int(max(c.t1_ms for c in ordered))
    schema = _summary_schema_json()
    facts = [
        {
            "id": str(c.id),
            "level": str(c.level),
            "t0_ms": int(c.t0_ms),
            "t1_ms": int(c.t1_ms),
            "text": str(c.text)[:160],
            "tags": [str(x) for x in (c.tags or [])[:8]],
        }
        for c in ordered[:24]
    ]
    prompt_payload = {
        "video_id": str(video_id),
        "t0_ms": int(t0_ms),
        "t1_ms": int(t1_ms),
        "facts": facts,
        "budget_hint": dict(budget_hint or {}),
    }
    provider_cfg = dict(provider_cfg or {})
    provider = str(provider_cfg.get("provider", "fake")).strip().lower() or "fake"
    model = str(provider_cfg.get("model", "fake-summary-v0")).strip() or "fake-summary-v0"
    model_meta = {
        "provider": provider,
        "model": model,
        "mode": "heuristic",
        "parse_ok": False,
        "error": "",
    }

    summary_obj: dict[str, Any] | None = None
    raw_text = ""
    can_call = bool(provider_cfg.get("enabled", False))
    api_key_env = str(provider_cfg.get("api_key_env", "")).strip()
    if provider != "fake" and api_key_env and not os.environ.get(api_key_env):
        can_call = False
        model_meta["error"] = f"missing_env:{api_key_env}"
    if can_call:
        cfg = ModelClientConfig(
            provider=provider,
            model=model,
            base_url=provider_cfg.get("base_url"),
            base_url_env=str(provider_cfg.get("base_url_env", "")),
            api_key_env=api_key_env,
            timeout_s=int(provider_cfg.get("timeout_s", 60)),
            max_tokens=int(provider_cfg.get("max_tokens", 600)),
            temperature=float(provider_cfg.get("temperature", 0.2)),
            model_cache_enabled=bool(provider_cfg.get("model_cache_enabled", True)),
            model_cache_dir=str(provider_cfg.get("model_cache_dir", "data/outputs/model_cache")),
            model_cache_max_entries=int(provider_cfg.get("model_cache_max_entries", 0)),
            model_cache_max_mb=int(provider_cfg.get("model_cache_max_mb", 0)),
            extra_headers=dict(provider_cfg.get("extra_headers", {})) if isinstance(provider_cfg.get("extra_headers", {}), dict) else {},
            extra=dict(provider_cfg.get("extra", {})) if isinstance(provider_cfg.get("extra", {}), dict) else {},
            max_retries=int(provider_cfg.get("max_retries", 1)),
        )
        client = make_client(cfg)
        sys_prompt = (
            "You are a deterministic repository summarizer. "
            "Return one JSON object only, following the provided schema."
        )
        user_prompt = _render_prompt(prompt_payload)
        parsed, raw_text, meta = generate_structured(
            client,
            schema_name="repo_summary_v0",
            schema_json=schema,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            temperature=float(cfg.temperature),
            max_tokens=int(cfg.max_tokens),
            timeout_s=float(cfg.timeout_s),
        )
        model_meta.update(
            {
                "provider": provider,
                "model": model,
                "mode": str(meta.get("used_mode", "fallback_parse")),
                "parse_ok": bool(meta.get("parse_ok", False)),
                "schema_ok": meta.get("schema_ok"),
                "error": str(meta.get("error", "")),
            }
        )
        if isinstance(parsed, dict) and parsed:
            summary_obj = dict(parsed)
    if not summary_obj:
        summary_obj = _heuristic_summary(
            chunks=ordered,
            video_id=str(video_id),
            t0_ms=int(t0_ms),
            t1_ms=int(t1_ms),
            policy_name=str(policy_name),
            model_meta=model_meta,
        )
    summary_obj.setdefault("provenance", {})
    if isinstance(summary_obj["provenance"], dict):
        summary_obj["provenance"].setdefault("input_chunk_ids", [str(c.id) for c in ordered])
        summary_obj["provenance"].setdefault("policy", str(policy_name))
        summary_obj["provenance"]["model"] = dict(model_meta)
    try:
        validated = RepoSummaryV0.model_validate(summary_obj) if hasattr(RepoSummaryV0, "model_validate") else RepoSummaryV0.parse_obj(summary_obj)
    except Exception:
        fallback = _heuristic_summary(
            chunks=ordered,
            video_id=str(video_id),
            t0_ms=int(t0_ms),
            t1_ms=int(t1_ms),
            policy_name=str(policy_name),
            model_meta=model_meta,
        )
        validated = RepoSummaryV0.model_validate(fallback) if hasattr(RepoSummaryV0, "model_validate") else RepoSummaryV0.parse_obj(fallback)
    payload = validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    tags = ["summary", f"summary_policy:{policy_name}"]
    for place in payload.get("places", [])[:4]:
        p = str(place).strip().lower()
        if p:
            tags.append(f"place:{p}")
    for item in payload.get("interactions", [])[:4]:
        if not isinstance(item, dict):
            continue
        obj = str(item.get("object", "")).strip().lower()
        if obj:
            tags.append(f"obj:{obj}")
    return RepoChunk(
        id=f"repo_summary_{digest}",
        chunk_id=f"repo_summary_{digest}",
        level="summary",
        scale="summary",
        t0=float(t0_ms) / 1000.0,
        t1=float(t1_ms) / 1000.0,
        t0_ms=int(t0_ms),
        t1_ms=int(t1_ms),
        text=_summary_text(payload),
        importance=0.85 if bool(model_meta.get("parse_ok", False)) else 0.65,
        source_ids=[str(x) for x in payload.get("provenance", {}).get("input_chunk_ids", [])],
        tags=sorted(set(tags)),
        score_fields={
            "summary_salience": 0.85 if bool(model_meta.get("parse_ok", False)) else 0.65,
            "coverage_score": min(1.0, len(payload.get("steps", [])) / 6.0),
        },
        payload={"summary_v0": payload},
        meta={
            "summary_model_meta": model_meta,
            "summary_schema": "RepoSummaryV0",
            "summary_raw_len": len(raw_text or ""),
        },
    )
