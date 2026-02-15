from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai_compat": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "glm": "ZHIPU_API_KEY",
    "fake": "FAKE_MODEL_API_KEY",
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
    api_key_env: str = ""
    timeout_s: int = 60
    max_tokens: int = 800
    temperature: float = 0.2
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = str(self.provider or "").strip().lower()
        self.model = str(self.model or "").strip()
        if not self.provider:
            raise ValueError("provider is required")
        if not self.model:
            raise ValueError("model is required")
        if not self.api_key_env:
            self.api_key_env = DEFAULT_API_KEY_ENV.get(self.provider, "OPENAI_API_KEY")
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
                redacted.append((k, "***"))
            else:
                redacted.append((k, v))
        new_query = urlencode(redacted, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))
    except Exception:
        return value
