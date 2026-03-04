from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from pov_compiler.models.client import ModelClientConfig, parse_json_from_text, redact_url
from pov_compiler.models.cost import estimate_cost_usd


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("candidates missing in Gemini response")
    first = candidates[0]
    if not isinstance(first, dict):
        raise RuntimeError("invalid candidates[0] in Gemini response")
    content = first.get("content", {})
    if not isinstance(content, dict):
        raise RuntimeError("content missing in Gemini response")
    parts = content.get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("parts missing in Gemini response")
    p0 = parts[0]
    if isinstance(p0, dict):
        text = p0.get("text", "")
        if isinstance(text, str):
            return text
    return str(p0)


class GeminiClient:
    def __init__(self, cfg: ModelClientConfig):
        self.cfg = cfg
        self._last_call_meta: dict[str, Any] = {}

    def _endpoint(self, api_key: str) -> str:
        base = str(self.cfg.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        model = quote(self.cfg.model, safe="-_.")
        return f"{base}/v1beta/models/{model}:generateContent?key={api_key}"

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
        endpoint = self._endpoint(api_key)
        generation_cfg: dict[str, Any] = {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        }
        if bool(kwargs.get("json_mime", False)):
            generation_cfg["responseMimeType"] = "application/json"
        response_schema = kwargs.get("response_schema")
        if isinstance(response_schema, dict):
            generation_cfg["responseSchema"] = dict(response_schema)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{system}\n\n{user}"}],
                }
            ],
            "generationConfig": generation_cfg,
        }
        headers = {"Content-Type": "application/json"}
        for k, v in self.cfg.extra_headers.items():
            headers[str(k)] = str(v)
        tries = max(1, int(getattr(self.cfg, "max_retries", 1)) + 1)
        last_exc: Exception | None = None
        for _ in range(tries):
            try:
                t0 = time.perf_counter()
                req = Request(
                    endpoint,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urlopen(req, timeout=float(timeout_s)) as resp:
                    text = resp.read().decode("utf-8", errors="ignore")
                result = json.loads(text)
                if not isinstance(result, dict):
                    raise RuntimeError("response is not a JSON object")
                content = _extract_text(result)
                usage = result.get("usageMetadata", {}) if isinstance(result, dict) else {}
                if not isinstance(usage, dict):
                    usage = {}
                p = int(float(usage.get("promptTokenCount", usage.get("prompt_tokens", 0)) or 0))
                c = int(float(usage.get("candidatesTokenCount", usage.get("completion_tokens", 0)) or 0))
                t = int(float(usage.get("totalTokenCount", p + c) or (p + c)))
                latency_ms = int((time.perf_counter() - t0) * 1000)
                estimated = estimate_cost_usd(
                    model=str(self.cfg.model),
                    prompt_tokens=p,
                    completion_tokens=c,
                    completion_text=str(content),
                )
                meta = {
                    "mode": "gemini",
                    "endpoint": redact_url(endpoint),
                    "status_code": 200,
                    "latency_ms": latency_ms,
                    "prompt_tokens": max(0, p),
                    "completion_tokens": max(0, c),
                    "total_tokens": max(0, t),
                    "estimated_cost_usd": estimated,
                    "strategy_used": str(kwargs.get("structured_strategy", "")),
                }
                self._last_call_meta = dict(meta)
                return content, meta
            except Exception as exc:  # pragma: no cover - retried path is still deterministic
                last_exc = exc
        safe_endpoint = redact_url(endpoint)
        raise RuntimeError(
            f"gemini call failed: {last_exc} "
            f"(provider={self.cfg.provider}, endpoint={safe_endpoint}, model={self.cfg.model})"
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
