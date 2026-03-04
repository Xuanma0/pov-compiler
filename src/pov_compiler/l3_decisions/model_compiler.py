from __future__ import annotations

import json
from typing import Any

from pov_compiler.models.client import ChatModelClient, ModelClientConfig
from pov_compiler.models.structured_output import generate_structured
from pov_compiler.schemas import Output

_DECISIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "decision_type": {"type": "string"},
                    "t0_ms": {"type": "number"},
                    "t1_ms": {"type": "number"},
                    "conf": {"type": "number"},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "span": {"type": "string"},
                        },
                    },
                },
                "required": ["decision_type", "t0_ms", "t1_ms", "conf"],
            },
        }
    },
    "required": ["decisions"],
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _extract_event_rows(output: Output, limit: int = 120) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    events_v1 = list(getattr(output, "events_v1", []) or [])
    for ev in events_v1[:limit]:
        rows.append(
            {
                "event_id": str(getattr(ev, "id", "")),
                "t0_ms": int(round(float(getattr(ev, "t0", 0.0)) * 1000.0)),
                "t1_ms": int(round(float(getattr(ev, "t1", 0.0)) * 1000.0)),
                "label": str(getattr(ev, "label", "")),
                "interaction_object": str(getattr(ev, "interaction_primary_object", "") or ""),
                "interaction_score": float(getattr(ev, "interaction_score", 0.0) or 0.0),
            }
        )
    if rows:
        return rows
    events = list(getattr(output, "events", []) or [])
    for ev in events[:limit]:
        rows.append(
            {
                "event_id": str(getattr(ev, "id", "")),
                "t0_ms": int(round(float(getattr(ev, "t0", 0.0)) * 1000.0)),
                "t1_ms": int(round(float(getattr(ev, "t1", 0.0)) * 1000.0)),
                "label": "",
            }
        )
    return rows


def _build_prompts(output: Output) -> tuple[str, str]:
    duration_ms = int(round(float(getattr(output, "meta", {}).get("duration_s", 0.0)) * 1000.0))
    payload = {
        "video_id": str(getattr(output, "video_id", "")),
        "duration_ms": duration_ms,
        "events": _extract_event_rows(output),
    }
    system = (
        "You are a strict JSON generator. Return ONLY a JSON object with key 'decisions'. "
        "Each item must contain decision_type, t0_ms, t1_ms, conf, evidence{event_id,span}."
    )
    user = json.dumps(payload, ensure_ascii=False)
    return system, user


def _validate_decisions(raw: Any, duration_ms: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise RuntimeError("model response must contain 'decisions' as a list")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        decision_type = str(item.get("decision_type", item.get("action_type", ""))).strip()
        if not decision_type:
            continue
        try:
            t0_ms = int(round(float(item.get("t0_ms", 0))))
            t1_ms = int(round(float(item.get("t1_ms", t0_ms + 1000))))
        except Exception:
            continue
        if duration_ms > 0:
            t0_ms = int(_clamp(t0_ms, 0.0, float(duration_ms)))
            t1_ms = int(_clamp(t1_ms, 0.0, float(duration_ms)))
        if t1_ms <= t0_ms:
            t1_ms = min(duration_ms if duration_ms > 0 else t0_ms + 1000, t0_ms + 1000)
        try:
            conf = float(item.get("conf", item.get("confidence", 0.5)))
        except Exception:
            conf = 0.5
        conf = _clamp(conf, 0.0, 1.0)
        evidence_obj = item.get("evidence", {})
        if not isinstance(evidence_obj, dict):
            evidence_obj = {}
        event_id = str(evidence_obj.get("event_id", "")).strip()
        span = str(evidence_obj.get("span", "")).strip()
        out.append(
            {
                "id": f"model_decision_{i:04d}",
                "decision_type": decision_type,
                "t0_ms": int(t0_ms),
                "t1_ms": int(t1_ms),
                "conf": float(conf),
                "evidence": {
                    "event_id": event_id,
                    "span": span,
                },
            }
        )
    if not out:
        raise RuntimeError("model returned zero valid decisions after validation")
    return out


def compile_decisions_with_model_and_meta(
    output: Output,
    client: ChatModelClient,
    cfg: ModelClientConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duration_ms = int(round(float(getattr(output, "meta", {}).get("duration_s", 0.0)) * 1000.0))
    system, user = _build_prompts(output)
    payload, _raw, parse_meta = generate_structured(
        client,
        schema_name="decisions_model_v1",
        schema_json=_DECISIONS_SCHEMA,
        system_prompt=system,
        user_prompt=user,
        temperature=float(cfg.temperature),
        max_tokens=int(cfg.max_tokens),
        timeout_s=float(cfg.timeout_s),
    )
    if not isinstance(payload, dict):
        raise RuntimeError("model client returned non-dict payload")
    decisions_raw = payload.get("decisions")
    if decisions_raw is None and isinstance(payload.get("decision_points"), list):
        decisions_raw = payload.get("decision_points")
    if decisions_raw is None:
        raise RuntimeError("model payload missing 'decisions' key")
    decisions = _validate_decisions(decisions_raw, duration_ms=duration_ms)
    meta = {
        "api_mode_used": str(parse_meta.get("api_mode_used", "")),
        "used_mode": str(parse_meta.get("used_mode", "")),
        "parse_ok": bool(parse_meta.get("parse_ok", False)),
        "parse_report": dict(parse_meta.get("parse_report", {})) if isinstance(parse_meta.get("parse_report", {}), dict) else {},
        "error": str(parse_meta.get("error", "")),
    }
    return decisions, meta


def compile_decisions_with_model(
    output: Output,
    client: ChatModelClient,
    cfg: ModelClientConfig,
) -> list[dict[str, Any]]:
    decisions, _meta = compile_decisions_with_model_and_meta(output=output, client=client, cfg=cfg)
    return decisions
