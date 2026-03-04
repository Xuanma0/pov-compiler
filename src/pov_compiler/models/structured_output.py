from __future__ import annotations

import json
import hashlib
import ast
from typing import Any

from pov_compiler.models.client import extract_first_json_object, parse_json_from_text, redact_url
from pov_compiler.models.presets import normalize_provider


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


def generate_structured(
    client: Any,
    *,
    schema_name: str,
    schema_json: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    provider, model = _provider_info(client)
    max_toks = int(max_tokens if max_tokens is not None else 800)
    timeout = int(timeout_s if timeout_s is not None else 60)
    mode = "fallback_parse"
    meta: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "schema_name": str(schema_name),
        "used_mode": mode,
        "api_mode_used": "",
        "parse_ok": False,
        "schema_ok": None,
        "error": "",
        "parse_report": {
            "raw_text_hash": "",
            "parse_ok": False,
            "error_type": "",
            "repair_used": False,
            "schema_version": "v1",
            "prompt_hash": hashlib.sha256(f"{schema_name}|{system_prompt}|{user_prompt}".encode("utf-8")).hexdigest()[:12],
        },
    }
    kwargs: dict[str, Any] = {}
    if provider in {"openai", "openai_compat"}:
        mode = "structured"
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(schema_name),
                "schema": dict(schema_json or {}),
            },
        }
    elif provider == "gemini":
        mode = "json_mime"
        kwargs["json_mime"] = True
    else:
        # OpenAI-compatible best-effort; providers may ignore this.
        mode = "structured"
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(schema_name),
                "schema": dict(schema_json or {}),
            },
        }
    meta["used_mode"] = mode

    raw_text = ""
    try:
        if hasattr(client, "generate_text"):
            raw_text, call_meta = client.generate_text(
                system=system_prompt,
                user=user_prompt,
                timeout_s=timeout,
                max_tokens=max_toks,
                temperature=float(temperature),
                **kwargs,
            )
            if isinstance(call_meta, dict):
                for key in ("endpoint", "mode", "provider"):
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
        else:
            # Fallback path for legacy clients.
            obj = client.complete_json(
                system=system_prompt,
                user=user_prompt,
                timeout_s=timeout,
                max_tokens=max_toks,
                temperature=float(temperature),
            )
            if not isinstance(obj, dict):
                raise RuntimeError("client.complete_json returned non-dict")
            raw_text = json.dumps(obj, ensure_ascii=False)
    except Exception as exc:
        meta["error"] = str(exc)
        meta["parse_report"]["error_type"] = "generate_text_error"
        return {}, raw_text, meta

    meta["parse_report"]["raw_text_hash"] = hashlib.sha256(str(raw_text or "").encode("utf-8")).hexdigest()[:12]
    repair_used = False
    try:
        parsed = parse_json_from_text(raw_text)
    except Exception:
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
                return {}, raw_text, meta
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

    meta["parse_ok"] = True
    meta["parse_report"]["parse_ok"] = True
    meta["parse_report"]["repair_used"] = bool(repair_used)
    schema_ok, schema_err = _validate_schema_best_effort(parsed, schema_json)
    meta["schema_ok"] = bool(schema_ok)
    if not schema_ok and schema_err:
        meta["error"] = f"schema_validation_failed:{schema_err}"
        meta["parse_report"]["error_type"] = "schema_validation_failed"
    elif repair_used:
        meta["parse_report"]["error_type"] = "repaired_parse"
    return parsed, raw_text, meta
