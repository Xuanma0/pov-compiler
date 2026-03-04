from __future__ import annotations

import json
import time
from typing import Any
from urllib.request import Request, urlopen

from pov_compiler.models.client import ModelClientConfig, parse_json_from_text, redact_url
from pov_compiler.models.cost import estimate_cost_usd
from pov_compiler.models.presets import get_preset, normalize_base_url, normalize_provider


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("choices missing in chat response")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("invalid choices[0] in chat response")
    message = first.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("message missing in chat response")
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def _extract_responses_text(payload: dict[str, Any]) -> str:
    out_text = payload.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text
    output = payload.get("output", [])
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    text = c.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
            text_alt = item.get("text")
            if isinstance(text_alt, str) and text_alt.strip():
                parts.append(text_alt)
        if parts:
            return "\n".join(parts).strip()
    raise RuntimeError("responses output text missing")


def _usage_from_chat(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    p = int(float(usage.get("prompt_tokens", 0) or 0))
    c = int(float(usage.get("completion_tokens", 0) or 0))
    t = int(float(usage.get("total_tokens", p + c) or (p + c)))
    return {"prompt_tokens": max(0, p), "completion_tokens": max(0, c), "total_tokens": max(0, t)}


def _usage_from_responses(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    p = int(float(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0))
    c = int(float(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0))
    t = int(float(usage.get("total_tokens", p + c) or (p + c)))
    return {"prompt_tokens": max(0, p), "completion_tokens": max(0, c), "total_tokens": max(0, t)}


class OpenAICompatClient:
    def __init__(self, cfg: ModelClientConfig):
        self.cfg = cfg
        self._last_call_meta: dict[str, Any] = {}

    def _base(self) -> str:
        provider = normalize_provider(self.cfg.provider)
        base = normalize_base_url(provider, self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        return base

    def _endpoint_chat(self) -> str:
        base = self._base()
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _endpoint_responses(self) -> str:
        base = self._base()
        if base.endswith("/responses"):
            return base
        return f"{base}/responses"

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        for k, v in self.cfg.extra_headers.items():
            headers[str(k)] = str(v)
        return headers

    def _post_json(
        self, endpoint: str, payload: dict[str, Any], timeout_s: int, headers: dict[str, str]
    ) -> tuple[dict[str, Any], int | None]:
        req = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=float(timeout_s)) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            status = int(getattr(resp, "status", 0) or 0)
        result = json.loads(text)
        if not isinstance(result, dict):
            raise RuntimeError("response is not a JSON object")
        return result, status

    def _resolve_mode_order(self, requested_mode: str) -> list[str]:
        mode = str(requested_mode or "auto").strip().lower()
        if mode not in {"auto", "responses", "chat"}:
            mode = "auto"
        provider = normalize_provider(self.cfg.provider)
        preset = get_preset(provider)
        if mode == "responses":
            return ["responses", "chat"]
        if mode == "chat":
            return ["chat"]
        # auto
        if bool(getattr(preset, "supports_responses", False)):
            return ["responses", "chat"]
        return ["chat", "responses"]

    def _call_chat(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        system: str,
        user: str,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
        structured_strategy: str = "",
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": str(system)},
                {"role": "user", "content": str(user)},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if isinstance(response_format, dict):
            payload["response_format"] = response_format
        t0 = time.perf_counter()
        result, status = self._post_json(endpoint, payload, int(timeout_s), headers)
        content = _extract_message_content(result)
        usage = _usage_from_chat(result)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        estimated = estimate_cost_usd(
            model=str(self.cfg.model),
            prompt_tokens=int(usage["prompt_tokens"]),
            completion_tokens=int(usage["completion_tokens"]),
            completion_text=str(content),
        )
        meta = {
            "api_mode_used": "chat",
            "endpoint": redact_url(endpoint),
            "status_code": int(status or 0),
            "latency_ms": latency_ms,
            **usage,
            "estimated_cost_usd": estimated,
            "strategy_used": str(structured_strategy or ""),
        }
        self._last_call_meta = dict(meta)
        return content, meta

    def _call_responses(
        self,
        *,
        endpoint: str,
        headers: dict[str, str],
        system: str,
        user: str,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
        structured_strategy: str = "",
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": str(system)}]},
                {"role": "user", "content": [{"type": "input_text", "text": str(user)}]},
            ],
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
        }
        if isinstance(response_format, dict):
            rf_type = str(response_format.get("type", "")).strip().lower()
            if rf_type == "json_schema":
                js = response_format.get("json_schema", {})
                if isinstance(js, dict):
                    payload["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": str(js.get("name", "schema")),
                            "schema": dict(js.get("schema", {})) if isinstance(js.get("schema", {}), dict) else {},
                        }
                    }
        t0 = time.perf_counter()
        result, status = self._post_json(endpoint, payload, int(timeout_s), headers)
        content = _extract_responses_text(result)
        usage = _usage_from_responses(result)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        estimated = estimate_cost_usd(
            model=str(self.cfg.model),
            prompt_tokens=int(usage["prompt_tokens"]),
            completion_tokens=int(usage["completion_tokens"]),
            completion_text=str(content),
        )
        meta = {
            "api_mode_used": "responses",
            "endpoint": redact_url(endpoint),
            "status_code": int(status or 0),
            "latency_ms": latency_ms,
            **usage,
            "estimated_cost_usd": estimated,
            "strategy_used": str(structured_strategy or ""),
        }
        self._last_call_meta = dict(meta)
        return content, meta

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        api_key = self.cfg.get_api_key_or_raise()
        headers = self._headers(api_key)
        response_format = kwargs.get("response_format")
        structured_strategy = str(kwargs.get("structured_strategy", ""))
        requested_mode = str(kwargs.get("api_mode", self.cfg.api_mode))
        mode_order = self._resolve_mode_order(requested_mode)
        tries = max(1, int(getattr(self.cfg, "max_retries", 1)) + 1)

        first_error: str = ""
        last_exc: Exception | None = None
        fallback_reason = ""

        for mode in mode_order:
            for _ in range(tries):
                try:
                    if mode == "responses":
                        text, meta = self._call_responses(
                            endpoint=self._endpoint_responses(),
                            headers=headers,
                            system=system,
                            user=user,
                            timeout_s=int(timeout_s),
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                            response_format=response_format if isinstance(response_format, dict) else None,
                            structured_strategy=structured_strategy,
                        )
                    else:
                        text, meta = self._call_chat(
                            endpoint=self._endpoint_chat(),
                            headers=headers,
                            system=system,
                            user=user,
                            timeout_s=int(timeout_s),
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                            response_format=response_format if isinstance(response_format, dict) else None,
                            structured_strategy=structured_strategy,
                        )
                    meta["provider"] = normalize_provider(self.cfg.provider)
                    if fallback_reason:
                        meta["fallback_reason"] = fallback_reason
                    self._last_call_meta = dict(meta)
                    return text, meta
                except Exception as exc:  # pragma: no cover - retry/fallback path
                    last_exc = exc
                    msg = f"{exc.__class__.__name__}"
                    if not first_error:
                        first_error = msg
                    fallback_reason = f"{mode}_failed:{msg}"
            if mode == "responses":
                # deterministic fallback to chat
                fallback_reason = fallback_reason or "responses_failed"

        safe_base = redact_url(self._base())
        error_label = fallback_reason or first_error or (last_exc.__class__.__name__ if last_exc else "unknown")
        raise RuntimeError(
            f"openai_compat call failed: {error_label} "
            f"(provider={self.cfg.provider}, base_url={safe_base}, model={self.cfg.model}, api_mode={self.cfg.api_mode})"
        ) from last_exc

    @property
    def last_call_meta(self) -> dict[str, Any]:
        return dict(self._last_call_meta or {})

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        text, _meta = self.generate_text(
            system=system,
            user=user,
            timeout_s=int(timeout_s),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        return parse_json_from_text(text)
