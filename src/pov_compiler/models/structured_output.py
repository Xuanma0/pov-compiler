from __future__ import annotations

import hashlib
import ast
import json
from typing import Any

from pov_compiler.models.capabilities import ModelCapabilities, infer_capabilities_static
from pov_compiler.models.client import extract_first_json_object, parse_json_from_text, redact_url
from pov_compiler.models.presets import normalize_provider

_ALLOWED_STRATEGIES = {"auto", "json_schema", "json_object", "tool", "prompted_json", "fallback_parse"}


def _provider_info(client: Any) -> tuple[str, str]:
    cfg = getattr(client, "cfg", None)
    if cfg is None:
        return "", ""
    provider = normalize_provider(str(getattr(cfg, "provider", "") or ""))
    model = str(getattr(cfg, "model", "") or "")
    return provider, model


def _validate_schema_best_effort(obj: dict[str, Any], schema_json: dict[str, Any]) -> tuple[bool, str]:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return True, ""
    try:
        jsonschema.validate(instance=obj, schema=schema_json)
        return True, ""
    except Exception as exc:  # pragma: no cover - schema errors are not deterministic across jsonschema versions
        return False, str(exc)


def _choose_strategy_auto(provider: str, caps: ModelCapabilities) -> str:
    p = normalize_provider(provider)
    if caps.supports_json_schema is True:
        return "json_schema"
    if p == "gemini" and caps.supports_json_object in {True, None}:
        return "json_object"
    if caps.supports_json_object is True:
        return "json_object"
    if caps.supports_tools is True:
        return "tool"
    return "prompted_json"


def _schema_hash(schema_json: dict[str, Any]) -> str:
    try:
        raw = json.dumps(schema_json, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = str(schema_json)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _safe_snippet(text: str, max_len: int = 200) -> str:
    out = str(text or "")[: max(8, int(max_len))]
    out = redact_url(out)
    low = out.lower()
    for token in ("authorization", "bearer ", "api_key", "sk-", "aiza", "key="):
        if token in low:
            out = out.replace(token, "***")
            low = out.lower()
    return out


def generate_structured(
    client: Any,
    *,
    schema_name: str,
    schema_json: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    strategy: str = "auto",
    strict: bool = True,
    capabilities: ModelCapabilities | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    provider, model = _provider_info(client)
    max_toks = int(max_tokens if max_tokens is not None else 800)
    timeout = int(timeout_s if timeout_s is not None else 60)
    strategy_req = str(strategy or "auto").strip().lower()
    if strategy_req not in _ALLOWED_STRATEGIES:
        strategy_req = "auto"
    caps = capabilities
    if caps is None:
        cfg = getattr(client, "cfg", None)
        if cfg is not None:
            caps = infer_capabilities_static(cfg)
        else:
            caps = infer_capabilities_static(provider, model=model)
    strategy_used = _choose_strategy_auto(provider, caps) if strategy_req == "auto" else strategy_req
    mode = "fallback_parse"
    meta: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "schema_name": str(schema_name),
        "strategy_requested": strategy_req,
        "strategy_used": strategy_used,
        "used_mode": mode,
        "api_mode_used": "",
        "parse_ok": False,
        "schema_ok": None,
        "error": "",
        "fallback_reason": "",
        "schema_hash": _schema_hash(schema_json),
        "capabilities": caps.to_dict() if isinstance(caps, ModelCapabilities) else {},
        "parse_report": {
            "raw_text_hash": "",
            "parse_ok": False,
            "error_type": "",
            "repair_used": False,
            "schema_version": "v1",
            "strategy_used": strategy_used,
            "prompt_hash": hashlib.sha256(f"{schema_name}|{system_prompt}|{user_prompt}".encode("utf-8")).hexdigest()[:12],
            "validator_error": "",
            "raw_snippet": "",
        },
    }
    kwargs: dict[str, Any] = {"structured_strategy": strategy_used}
    if strategy_used == "json_schema":
        mode = "structured"
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(schema_name),
                "schema": dict(schema_json or {}),
                "strict": bool(strict),
            },
        }
        if provider == "gemini":
            kwargs["json_mime"] = True
            kwargs["response_schema"] = dict(schema_json or {})
    elif strategy_used == "json_object":
        mode = "json_object"
        kwargs["response_format"] = {"type": "json_object"}
        if provider == "gemini":
            kwargs["json_mime"] = True
            kwargs["response_schema"] = dict(schema_json or {})
    elif strategy_used == "tool":
        mode = "tool"
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(schema_name),
                "schema": dict(schema_json or {}),
                "strict": bool(strict),
            },
        }
    elif strategy_used == "prompted_json":
        mode = "prompted_json"
    else:
        mode = "fallback_parse"
    meta["used_mode"] = mode

    raw_text = ""
    prompt_for_call = str(user_prompt)
    if strategy_used in {"prompted_json", "fallback_parse"}:
        prompt_for_call = (
            f"{str(user_prompt)}\n\n"
            f"Return ONLY JSON object matching schema_name={schema_name}. "
            f"Schema={json.dumps(schema_json, ensure_ascii=False, sort_keys=True)}"
        )

    def _call_once(call_prompt: str, call_kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if hasattr(client, "generate_text"):
            text_out, call_meta = client.generate_text(
                system=system_prompt,
                user=call_prompt,
                timeout_s=timeout,
                max_tokens=max_toks,
                temperature=float(temperature),
                **call_kwargs,
            )
            return str(text_out or ""), dict(call_meta or {}) if isinstance(call_meta, dict) else {}
        obj = client.complete_json(
            system=system_prompt,
            user=call_prompt,
            timeout_s=timeout,
            max_tokens=max_toks,
            temperature=float(temperature),
        )
        if not isinstance(obj, dict):
            raise RuntimeError("client.complete_json returned non-dict")
        return json.dumps(obj, ensure_ascii=False), {}

    parse_exc: Exception | None = None
    parsed: dict[str, Any] | None = None
    repair_used = False
    try:
        raw_text, call_meta = _call_once(prompt_for_call, dict(kwargs))
        if isinstance(call_meta, dict):
            for key in (
                "endpoint",
                "mode",
                "provider",
                "status_code",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "estimated_cost_usd",
            ):
                if key in call_meta and key not in meta:
                    val = call_meta[key]
                    if isinstance(val, str):
                        meta[key] = redact_url(val)
                    else:
                        meta[key] = val
            if "api_mode_used" in call_meta:
                meta["api_mode_used"] = str(call_meta.get("api_mode_used", ""))
            if "fallback_reason" in call_meta:
                meta["fallback_reason"] = str(call_meta.get("fallback_reason", ""))
            if "strategy_used" in call_meta and not meta.get("strategy_used"):
                meta["strategy_used"] = str(call_meta.get("strategy_used", ""))
    except Exception as exc:
        meta["error"] = str(exc)
        meta["parse_report"]["error_type"] = "generate_text_error"
        return {}, raw_text, meta

    meta["parse_report"]["raw_text_hash"] = hashlib.sha256(str(raw_text or "").encode("utf-8")).hexdigest()[:12]
    try:
        parsed = parse_json_from_text(raw_text)
    except Exception as exc:
        parse_exc = exc
        snippet = extract_first_json_object(raw_text or "")
        if not snippet:
            # last repair path: literal-eval for python-style dicts
            try:
                maybe_obj = ast.literal_eval(str(raw_text or "").strip())
                if isinstance(maybe_obj, dict):
                    parsed = dict(maybe_obj)
                    repair_used = True
                else:
                    meta["error"] = "json_parse_failed"
                    meta["parse_report"]["error_type"] = "json_parse_failed"
                    return {}, raw_text, meta
            except Exception:
                meta["error"] = "json_parse_failed"
                meta["parse_report"]["error_type"] = "json_parse_failed"
                parsed = None
        else:
            try:
                maybe = json.loads(snippet)
            except Exception:
                try:
                    maybe_lit = ast.literal_eval(snippet)
                    maybe = maybe_lit if isinstance(maybe_lit, dict) else {}
                except Exception as exc:
                    meta["error"] = f"json_parse_failed:{exc}"
                    meta["parse_report"]["error_type"] = "json_parse_failed"
                    return {}, raw_text, meta
            if not isinstance(maybe, dict):
                meta["error"] = "json_not_object"
                meta["parse_report"]["error_type"] = "json_not_object"
                return {}, raw_text, meta
            parsed = maybe
            repair_used = True

    if parsed is None:
        # one repair re-ask when parse failed and strategy wasn't already fallback-only
        if strategy_used not in {"fallback_parse"}:
            repair_used = True
            repair_prompt = (
                f"{str(user_prompt)}\n\n"
                "Your prior response was not valid JSON for the required schema. "
                "Return ONLY a valid JSON object, no markdown."
            )
            try:
                raw_text, repair_meta = _call_once(
                    repair_prompt,
                    {
                        "structured_strategy": "prompted_json_repair",
                        "response_format": {"type": "json_object"},
                        "json_mime": provider == "gemini",
                    },
                )
                if isinstance(repair_meta, dict):
                    if repair_meta.get("api_mode_used"):
                        meta["api_mode_used"] = str(repair_meta.get("api_mode_used", ""))
                    if repair_meta.get("fallback_reason"):
                        meta["fallback_reason"] = str(repair_meta.get("fallback_reason", ""))
                parsed = parse_json_from_text(raw_text)
                meta["strategy_used"] = "prompted_json_repair"
                meta["parse_report"]["strategy_used"] = "prompted_json_repair"
            except Exception as exc:
                meta["error"] = f"json_parse_failed:{exc}"
                meta["parse_report"]["error_type"] = "json_parse_failed"
                meta["parse_report"]["raw_snippet"] = _safe_snippet(raw_text)
                return {}, raw_text, meta
        else:
            meta["error"] = f"json_parse_failed:{parse_exc}" if parse_exc else "json_parse_failed"
            meta["parse_report"]["error_type"] = "json_parse_failed"
            meta["parse_report"]["raw_snippet"] = _safe_snippet(raw_text)
            return {}, raw_text, meta

    meta["parse_ok"] = True
    meta["parse_report"]["parse_ok"] = True
    meta["parse_report"]["repair_used"] = bool(repair_used)
    schema_ok, schema_err = _validate_schema_best_effort(parsed, schema_json)
    meta["schema_ok"] = bool(schema_ok)
    if not schema_ok and schema_err:
        meta["error"] = f"schema_validation_failed:{schema_err[:240]}"
        meta["parse_report"]["error_type"] = "schema_validation_failed"
        meta["parse_report"]["validator_error"] = str(schema_err[:240])
    elif repair_used:
        meta["parse_report"]["error_type"] = "repaired_parse"
    meta["parse_report"]["raw_snippet"] = _safe_snippet(raw_text)
    return parsed, raw_text, meta
