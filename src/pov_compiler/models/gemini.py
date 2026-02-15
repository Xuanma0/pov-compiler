from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from pov_compiler.models.client import ModelClientConfig, parse_json_from_text, redact_url


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

    def _endpoint(self, api_key: str) -> str:
        base = str(self.cfg.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        model = quote(self.cfg.model, safe="-_.")
        return f"{base}/v1beta/models/{model}:generateContent?key={api_key}"

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        api_key = self.cfg.get_api_key_or_raise()
        endpoint = self._endpoint(api_key)
        try:
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system}\n\n{user}"}],
                    }
                ],
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": int(max_tokens),
                },
            }
            headers = {"Content-Type": "application/json"}
            for k, v in self.cfg.extra_headers.items():
                headers[str(k)] = str(v)
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
            return parse_json_from_text(content)
        except Exception as exc:
            safe_endpoint = redact_url(endpoint)
            raise RuntimeError(
                f"gemini call failed: {exc} "
                f"(provider={self.cfg.provider}, endpoint={safe_endpoint}, model={self.cfg.model})"
            ) from exc
