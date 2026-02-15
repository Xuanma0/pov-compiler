from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pov_compiler.models.cache import ModelCallCache, SCHEMA_VERSION


DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai_compat": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "qwen": "QWEN_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "glm": "GLM_API_KEY",
    "fake": "FAKE_MODEL_API_KEY",
}

DEFAULT_BASE_URL_ENV: dict[str, str] = {
    "openai_compat": "OPENAI_BASE_URL",
    "qwen": "QWEN_BASE_URL",
    "deepseek": "DEEPSEEK_BASE_URL",
    "glm": "GLM_BASE_URL",
}

_SENSITIVE_KEYWORDS = ("key", "token", "secret", "authorization", "bearer", "password")
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


class ChatModelClient(Protocol):
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class ModelClientConfig:
    provider: str
    model: str
    base_url: str | None = None
    base_url_env: str = ""
    api_key_env: str = ""
    timeout_s: int = 60
    max_tokens: int = 800
    temperature: float = 0.2
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    model_cache_enabled: bool = True
    model_cache_dir: str = "data/outputs/model_cache"
    model_cache_max_entries: int = 0
    model_cache_max_mb: int = 0

    def __post_init__(self) -> None:
        self.provider = str(self.provider or "").strip().lower()
        self.model = str(self.model or "").strip()
        if not self.provider:
            raise ValueError("provider is required")
        if not self.model:
            raise ValueError("model is required")
        if not self.api_key_env:
            self.api_key_env = DEFAULT_API_KEY_ENV.get(self.provider, "OPENAI_API_KEY")
        if not self.base_url:
            env_name = str(self.base_url_env or DEFAULT_BASE_URL_ENV.get(self.provider, ""))
            if env_name:
                env_val = os.environ.get(env_name, "").strip()
                if env_val:
                    self.base_url = env_val
        blocked_headers = {"authorization", "x-api-key", "api-key"}
        for key in list(self.extra_headers.keys()):
            if str(key).strip().lower() in blocked_headers:
                raise ValueError("extra_headers must not include secret-bearing headers; use api_key_env")

    def get_api_key_or_raise(self) -> str:
        if self.provider == "fake":
            return ""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"missing API key env '{self.api_key_env}' for provider={self.provider}; "
                f"set the environment variable and retry"
            )
        return key

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("api_key_env", None)
        payload["credential_env_hint_hash"] = _stable_hash(self.api_key_env)
        payload["extra_headers"] = self.redact_dict(payload.get("extra_headers", {}))
        payload["extra"] = self.redact_dict(payload.get("extra", {}))
        payload["base_url"] = redact_url(str(payload.get("base_url") or ""))
        return payload

    @staticmethod
    def redact_dict(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                key = str(k)
                key_norm = key.lower()
                if any(tok in key_norm for tok in _SENSITIVE_KEYWORDS):
                    out[key] = "***"
                    continue
                out[key] = ModelClientConfig.redact_dict(v)
            return out
        if isinstance(value, list):
            return [ModelClientConfig.redact_dict(v) for v in value]
        if isinstance(value, str):
            return redact_url(value)
        return value


@dataclass(slots=True)
class CachedModelClient:
    inner: ChatModelClient
    cfg: ModelClientConfig
    cache: ModelCallCache

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        request_payload = {
            "system": str(system),
            "user": str(user),
            "timeout_s": int(timeout_s),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        key = ModelCallCache.build_key(
            provider=self.cfg.provider,
            model=self.cfg.model,
            base_url=redact_url(str(self.cfg.base_url or "")),
            request_payload=request_payload,
            schema_version=SCHEMA_VERSION,
        )
        cached = self.cache.get(key)
        if isinstance(cached, dict):
            return cached
        response = self.inner.complete_json(
            system=system,
            user=user,
            timeout_s=int(timeout_s),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        if isinstance(response, dict):
            self.cache.set(key, response)
        return response


class ModelClientWithStats(Protocol):
    def get_model_cache_stats(self) -> dict[str, Any]:
        ...


def get_model_cache_stats(client: Any) -> dict[str, Any]:
    cache = getattr(client, "cache", None)
    if isinstance(cache, ModelCallCache):
        return {
            "enabled": True,
            "dir": str(cache.cache_dir),
            "schema_version": SCHEMA_VERSION,
            **cache.stats.to_dict(),
        }
    return {
        "enabled": False,
        "dir": "",
        "schema_version": SCHEMA_VERSION,
        "hit": 0,
        "miss": 0,
        "write_fail": 0,
        "hash_prefix": "",
    }


def maybe_wrap_with_cache(client: ChatModelClient, cfg: ModelClientConfig) -> ChatModelClient:
    if not bool(cfg.model_cache_enabled):
        return client
    cache = ModelCallCache(
        cache_dir=str(cfg.model_cache_dir),
        max_entries=int(cfg.model_cache_max_entries),
        max_mb=int(cfg.model_cache_max_mb),
    )
    return CachedModelClient(inner=client, cfg=cfg, cache=cache)


def _stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]


def extract_first_json_object(text: str) -> str | None:
    s = str(text or "")
    if not s:
        return None
    depth = 0
    start: int | None = None
    in_quote = False
    escaped = False
    for i, ch in enumerate(s):
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return s[start : i + 1]
    match = _JSON_OBJ_RE.search(s)
    return match.group(0) if match else None


def parse_json_from_text(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise RuntimeError("empty model response")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    snippet = extract_first_json_object(raw)
    if not snippet:
        raise RuntimeError("model response does not contain a JSON object")
    try:
        parsed = json.loads(snippet)
    except Exception as exc:
        raise RuntimeError(f"failed to parse JSON object from model response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("parsed model JSON is not an object")
    return parsed


def redact_url(url: str) -> str:
    value = str(url or "")
    if not value:
        return value
    try:
        parts = urlsplit(value)
        if not parts.query:
            return value
        q = parse_qsl(parts.query, keep_blank_values=True)
        redacted: list[tuple[str, str]] = []
        for k, v in q:
            kn = str(k).strip().lower()
            if any(tok in kn for tok in _SENSITIVE_KEYWORDS):
                redacted.append(("redacted", "***"))
            else:
                redacted.append((k, v))
        new_query = urlencode(redacted, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
    except Exception:
        return value
