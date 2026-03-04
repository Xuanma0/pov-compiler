from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models.capabilities import ModelCapabilities
from pov_compiler.models.client import ModelClientConfig
from pov_compiler.models.structured_output import generate_structured


class _AutoStrategyClient:
    def __init__(self, *, fail_json_schema: bool = False) -> None:
        self.cfg = ModelClientConfig(provider="fake", model="fake-structured", model_cache_enabled=False)
        self.fail_json_schema = bool(fail_json_schema)
        self.calls: list[str] = []

    def generate_text(self, system: str, user: str, *, timeout_s: int, max_tokens: int, temperature: float, **kwargs):  # type: ignore[no-untyped-def]
        strat = str(kwargs.get("structured_strategy", ""))
        self.calls.append(strat)
        if strat == "json_schema":
            if self.fail_json_schema:
                return "not a json response", {"strategy_used": "json_schema"}
            return '{"ok": true}', {"strategy_used": "json_schema"}
        return '{"ok": true}', {"strategy_used": strat or "prompted_json"}


def test_auto_strategy_prefers_json_schema_when_supported() -> None:
    client = _AutoStrategyClient()
    caps = ModelCapabilities(
        provider="fake",
        model="fake-structured",
        supports_json_schema=True,
        supports_json_object=True,
        supports_tools=True,
        supports_responses_api=False,
        supports_chat_api=True,
    )
    obj, _raw, meta = generate_structured(
        client,
        schema_name="probe",
        schema_json={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        system_prompt="sys",
        user_prompt="usr",
        strategy="auto",
        strict=True,
        capabilities=caps,
    )
    assert obj.get("ok") is True
    assert str(meta.get("strategy_used")) == "json_schema"
    assert client.calls and client.calls[0] == "json_schema"


def test_auto_strategy_repair_path_records_fallback() -> None:
    client = _AutoStrategyClient(fail_json_schema=True)
    caps = ModelCapabilities(
        provider="fake",
        model="fake-structured",
        supports_json_schema=True,
        supports_json_object=True,
        supports_tools=True,
        supports_responses_api=False,
        supports_chat_api=True,
    )
    obj, _raw, meta = generate_structured(
        client,
        schema_name="probe",
        schema_json={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        system_prompt="sys",
        user_prompt="usr",
        strategy="auto",
        strict=True,
        capabilities=caps,
    )
    assert obj.get("ok") is True
    parse_report = dict(meta.get("parse_report", {}))
    assert bool(parse_report.get("repair_used")) is True
    assert "prompted_json_repair" in str(meta.get("strategy_used", "")) or "prompted_json_repair" in str(
        parse_report.get("strategy_used", "")
    )
