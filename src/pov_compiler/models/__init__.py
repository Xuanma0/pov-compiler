from __future__ import annotations

from pov_compiler.models.client import (
    ChatModelClient,
    ModelClientConfig,
    get_model_cache_stats,
    maybe_wrap_with_cache,
)
from pov_compiler.models.presets import get_preset, list_presets, normalize_provider
from pov_compiler.models.structured_output import generate_structured


def _apply_provider_defaults(cfg: ModelClientConfig) -> ModelClientConfig:
    provider = normalize_provider(cfg.provider)
    preset = get_preset(provider)
    cfg.provider = provider
    if not cfg.base_url:
        cfg.base_url = preset.default_base_url
    if not cfg.api_key_env:
        cfg.api_key_env = preset.default_api_key_env
    if not cfg.base_url_env:
        cfg.base_url_env = preset.default_base_url_env
    if preset.default_headers:
        merged = dict(preset.default_headers)
        merged.update({str(k): str(v) for k, v in cfg.extra_headers.items()})
        cfg.extra_headers = merged
    return cfg


def make_client(cfg: ModelClientConfig) -> ChatModelClient:
    cfg = _apply_provider_defaults(cfg)
    provider = normalize_provider(cfg.provider)
    if provider == "fake":
        from pov_compiler.models.fake import FakeModelClient

        return maybe_wrap_with_cache(FakeModelClient(cfg), cfg)
    if provider in {"openai_compat", "deepseek", "qwen", "glm"}:
        from pov_compiler.models.openai_compat import OpenAICompatClient

        return maybe_wrap_with_cache(OpenAICompatClient(cfg), cfg)
    if provider == "gemini":
        from pov_compiler.models.gemini import GeminiClient

        return maybe_wrap_with_cache(GeminiClient(cfg), cfg)
    raise RuntimeError(f"unsupported model provider: {provider}")


__all__ = [
    "ChatModelClient",
    "ModelClientConfig",
    "make_client",
    "get_model_cache_stats",
    "get_preset",
    "list_presets",
    "generate_structured",
]
