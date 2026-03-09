from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pov_compiler.models.client import ModelClientConfig, parse_json_from_text, redact_url
from pov_compiler.models.presets import get_preset, normalize_provider


@dataclass(slots=True)
class ModelCapabilities:
    provider: str
    model: str
    supports_json_schema: bool | None = None
    supports_json_object: bool | None = None
    supports_tools: bool | None = None
    supports_responses_api: bool | None = None
    supports_chat_api: bool | None = None
    source: str = "static"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cap_state(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if bool(value) else "false"


def capability_states(caps: ModelCapabilities) -> dict[str, str]:
    return {
        "json_schema": _cap_state(caps.supports_json_schema),
        "json_object": _cap_state(caps.supports_json_object),
        "tools": _cap_state(caps.supports_tools),
        "responses_api": _cap_state(caps.supports_responses_api),
        "chat_api": _cap_state(caps.supports_chat_api),
    }


def infer_capabilities_static(cfg: ModelClientConfig | dict[str, Any] | str, model: str | None = None) -> ModelCapabilities:
    if isinstance(cfg, ModelClientConfig):
        provider = normalize_provider(cfg.provider)
        model_name = str(cfg.model)
    elif isinstance(cfg, dict):
        provider = normalize_provider(str(cfg.get("provider", "")))
        model_name = str(cfg.get("model", model or ""))
    else:
        provider = normalize_provider(str(cfg))
        model_name = str(model or "")

    preset = get_preset(provider)
    caps = ModelCapabilities(
        provider=str(provider),
        model=str(model_name),
        supports_responses_api=bool(getattr(preset, "supports_responses", False)),
        supports_chat_api=True,
        source="static",
    )

    if provider in {"openai_compat"}:
        caps.supports_json_schema = True
        caps.supports_json_object = True
        caps.supports_tools = True
    elif provider in {"deepseek", "qwen", "qwen_intl", "glm"}:
        # OpenAI-compatible endpoints vary by vendor version; keep conservative.
        caps.supports_json_schema = None
        caps.supports_json_object = True
        caps.supports_tools = None
    elif provider == "gemini":
        caps.supports_json_schema = None
        caps.supports_json_object = True
        caps.supports_tools = False
        caps.supports_responses_api = False
    elif provider == "fake":
        caps.supports_json_schema = True
        caps.supports_json_object = True
        caps.supports_tools = True
        caps.supports_responses_api = False
        caps.supports_chat_api = True
    else:
        caps.supports_json_schema = None
        caps.supports_json_object = None
        caps.supports_tools = None
    return caps


def _cache_file(path_or_dir: str | Path) -> Path:
    p = Path(path_or_dir)
    if p.suffix.lower() == ".json":
        return p
    return p / "model_capabilities_cache.json"


def _cache_key(cfg: ModelClientConfig) -> str:
    base = str(cfg.base_url or "").strip()
    return f"{normalize_provider(cfg.provider)}|{cfg.model}|{redact_url(base)}|{cfg.api_mode}"


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _probe_json_schema(client: Any, *, timeout_s: int = 8) -> tuple[bool, str]:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    response_format = {"type": "json_schema", "json_schema": {"name": "probe_ok", "schema": schema, "strict": True}}
    try:
        text, _meta = client.generate_text(
            system="Return JSON only.",
            user='{"ok":true}',
            timeout_s=int(timeout_s),
            max_tokens=32,
            temperature=0.0,
            response_format=response_format,
            structured_strategy="json_schema",
            capability_probe=True,
        )
        payload = parse_json_from_text(text)
        return bool(payload.get("ok") is True), ""
    except Exception as exc:
        return False, str(exc)


def resolve_model_capabilities(
    cfg: ModelClientConfig,
    *,
    probe: bool = False,
    probe_timeout_s: int = 8,
    probe_cache_dir: str | Path = "data/outputs/model_capabilities",
) -> tuple[ModelCapabilities, dict[str, Any]]:
    static_caps = infer_capabilities_static(cfg)
    cache_path = _cache_file(probe_cache_dir)
    key = _cache_key(cfg)
    meta: dict[str, Any] = {
        "probe_used": False,
        "probe_cached": False,
        "probe_error": "",
        "cache_path": str(cache_path),
    }
    if not probe:
        return static_caps, meta

    cache = _load_cache(cache_path)
    cached = cache.get(key)
    if isinstance(cached, dict):
        try:
            caps = ModelCapabilities(
                provider=str(cached.get("provider", static_caps.provider)),
                model=str(cached.get("model", static_caps.model)),
                supports_json_schema=cached.get("supports_json_schema"),
                supports_json_object=cached.get("supports_json_object"),
                supports_tools=cached.get("supports_tools"),
                supports_responses_api=cached.get("supports_responses_api"),
                supports_chat_api=cached.get("supports_chat_api"),
                source="probe_cache",
            )
            meta["probe_cached"] = True
            return caps, meta
        except Exception:
            pass

    try:
        api_key_present = bool(cfg.provider == "fake" or cfg.get_api_key_or_raise())
    except Exception:
        api_key_present = False
    if not api_key_present:
        meta["probe_error"] = "missing_api_key"
        return static_caps, meta

    from pov_compiler.models import make_client

    client = make_client(cfg)
    ok, err = _probe_json_schema(client, timeout_s=int(probe_timeout_s))
    caps = ModelCapabilities(
        provider=static_caps.provider,
        model=static_caps.model,
        supports_json_schema=bool(ok),
        supports_json_object=static_caps.supports_json_object if static_caps.supports_json_object is not None else bool(ok),
        supports_tools=static_caps.supports_tools,
        supports_responses_api=static_caps.supports_responses_api,
        supports_chat_api=static_caps.supports_chat_api,
        source="probe",
    )
    meta["probe_used"] = True
    meta["probe_error"] = str(err or "")
    cache[key] = {
        **caps.to_dict(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _save_cache(cache_path, cache)
    except Exception:
        pass
    return caps, meta
