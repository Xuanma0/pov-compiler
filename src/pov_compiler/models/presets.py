from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    name: str
    default_base_url: str
    default_api_key_env: str
    default_base_url_env: str
    default_api_mode: str = "auto"
    supports_responses: bool = False
    supports_chat_json_schema: bool = True
    default_headers: dict[str, str] = field(default_factory=dict)
    notes: str = ""


_PRESETS: dict[str, ProviderPreset] = {
    "openai_compat": ProviderPreset(
        name="openai_compat",
        default_base_url="https://api.openai.com/v1",
        default_api_key_env="OPENAI_API_KEY",
        default_base_url_env="OPENAI_BASE_URL",
        default_api_mode="auto",
        supports_responses=True,
        supports_chat_json_schema=True,
        notes="OpenAI and OpenAI-compatible default route.",
    ),
    "gemini": ProviderPreset(
        name="gemini",
        default_base_url="https://generativelanguage.googleapis.com",
        default_api_key_env="GEMINI_API_KEY",
        default_base_url_env="GEMINI_BASE_URL",
        default_api_mode="chat",
        supports_responses=False,
        supports_chat_json_schema=False,
    ),
    "qwen": ProviderPreset(
        name="qwen",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_api_key_env="DASHSCOPE_API_KEY",
        default_base_url_env="QWEN_BASE_URL",
        default_api_mode="auto",
        supports_responses=True,
        supports_chat_json_schema=True,
        notes="DashScope CN endpoint; set QWEN_REGION=intl to default intl endpoint.",
    ),
    "qwen_intl": ProviderPreset(
        name="qwen_intl",
        default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_api_key_env="DASHSCOPE_API_KEY",
        default_base_url_env="QWEN_INTL_BASE_URL",
        default_api_mode="auto",
        supports_responses=True,
        supports_chat_json_schema=True,
        notes="DashScope international endpoint.",
    ),
    "deepseek": ProviderPreset(
        name="deepseek",
        default_base_url="https://api.deepseek.com/v1",
        default_api_key_env="DEEPSEEK_API_KEY",
        default_base_url_env="DEEPSEEK_BASE_URL",
        default_api_mode="auto",
        supports_responses=True,
        supports_chat_json_schema=True,
        notes="DeepSeek OpenAI-compatible endpoint.",
    ),
    "glm": ProviderPreset(
        name="glm",
        default_base_url="https://api.z.ai/api/paas/v4/",
        default_api_key_env="ZAI_API_KEY",
        default_base_url_env="GLM_BASE_URL",
        default_api_mode="auto",
        supports_responses=True,
        supports_chat_json_schema=True,
        notes="Z.AI GLM OpenAI-compatible endpoint; coding endpoint can be set via --base-url override.",
    ),
    "fake": ProviderPreset(
        name="fake",
        default_base_url="",
        default_api_key_env="FAKE_MODEL_API_KEY",
        default_base_url_env="",
        default_api_mode="chat",
        supports_responses=False,
        supports_chat_json_schema=False,
    ),
}


def normalize_provider(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    if raw in {"openai", "openai-compatible", "openai_compatible"}:
        return "openai_compat"
    if raw in {"dashscope-intl", "qwen-intl", "qwen_intl"}:
        return "qwen_intl"
    return raw


def normalize_base_url(provider: str, base_url: str | None) -> str:
    p = normalize_provider(provider)
    value = str(base_url or "").strip()
    if not value:
        return value
    if p in {"openai_compat", "qwen", "qwen_intl", "deepseek", "glm"}:
        v = value.rstrip("/")
        if v.endswith("/chat/completions"):
            v = v[: -len("/chat/completions")]
        if v.endswith("/responses"):
            v = v[: -len("/responses")]
        if p == "deepseek" and not v.endswith("/v1"):
            v = f"{v}/v1"
        return v
    return value.rstrip("/")


def get_preset(provider: str) -> ProviderPreset:
    p = normalize_provider(provider)
    if p == "qwen":
        region = str(os.environ.get("QWEN_REGION", "")).strip().lower()
        if region in {"intl", "international", "global"}:
            p = "qwen_intl"
    if p in _PRESETS:
        return _PRESETS[p]
    raise RuntimeError(f"unsupported provider preset: {provider}")


def list_presets() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, item in _PRESETS.items():
        out[name] = {
            "name": item.name,
            "default_base_url": item.default_base_url,
            "default_api_key_env": item.default_api_key_env,
            "default_base_url_env": item.default_base_url_env,
            "default_headers": dict(item.default_headers),
            "notes": item.notes,
        }
    return out
