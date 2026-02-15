from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

from pov_compiler.models.client import ModelClientConfig, parse_json_from_text, redact_url


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("choices missing in OpenAI-compatible response")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("invalid choices[0] in OpenAI-compatible response")
    message = first.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("message missing in OpenAI-compatible response")
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


class OpenAICompatClient:
    def __init__(self, cfg: ModelClientConfig):
        self.cfg = cfg

    def _endpoint(self) -> str:
        base = str(self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        timeout_s: int,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        endpoint = self._endpoint()
        try:
            api_key = self.cfg.get_api_key_or_raise()
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": str(system)},
                    {"role": "user", "content": str(user)},
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
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
            content = _extract_message_content(result)
            return parse_json_from_text(content)
        except Exception as exc:
            safe_endpoint = redact_url(endpoint)
            raise RuntimeError(
                f"openai_compat call failed: {exc} "
                f"(provider={self.cfg.provider}, base_url={safe_endpoint}, model={self.cfg.model})"
            ) from exc
