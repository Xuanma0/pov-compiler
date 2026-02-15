from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models import make_client
from pov_compiler.models.client import ModelClientConfig
from pov_compiler.models.gemini import GeminiClient
from pov_compiler.models.openai_compat import OpenAICompatClient


class _DummyResp:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def __enter__(self) -> "_DummyResp":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")


def test_openai_compat_client_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _DummyResp({"choices": [{"message": {"content": "{\"decisions\": []}"}}]})

    monkeypatch.setattr("pov_compiler.models.openai_compat.urlopen", _fake_urlopen)

    cfg = ModelClientConfig(
        provider="openai_compat",
        model="gpt-4o-mini",
        base_url="https://example.test/v1",
    )
    client = OpenAICompatClient(cfg)
    result = client.complete_json("sys", "usr", timeout_s=12, max_tokens=77, temperature=0.3)
    assert result == {"decisions": []}
    assert str(captured["url"]).endswith("/chat/completions")
    auth = str(captured["headers"].get("Authorization", ""))
    assert auth.startswith("Bearer ")
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["max_tokens"] == 77
    assert abs(float(captured["body"]["temperature"]) - 0.3) < 1e-6


def test_gemini_client_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFakeKeyButLongEnough12345")
    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _DummyResp({"candidates": [{"content": {"parts": [{"text": "{\"decisions\": []}"}]}}]})

    monkeypatch.setattr("pov_compiler.models.gemini.urlopen", _fake_urlopen)

    cfg = ModelClientConfig(
        provider="gemini",
        model="gemini-1.5-flash",
        base_url="https://generativelanguage.googleapis.com",
    )
    client = GeminiClient(cfg)
    result = client.complete_json("sys", "usr", timeout_s=10, max_tokens=88, temperature=0.4)
    assert result == {"decisions": []}
    assert ":generateContent" in str(captured["url"])
    assert "key=" in str(captured["url"])
    body = captured["body"]
    assert "contents" in body and isinstance(body["contents"], list)
    assert "generationConfig" in body and isinstance(body["generationConfig"], dict)
    assert int(body["generationConfig"]["maxOutputTokens"]) == 88


def test_make_client_routes_openai_compat_family() -> None:
    for provider in ("openai_compat", "qwen", "deepseek", "glm"):
        cfg = ModelClientConfig(provider=provider, model="demo-model", model_cache_enabled=False)
        client = make_client(cfg)
        assert client.__class__.__name__ == "OpenAICompatClient"
