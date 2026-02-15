from __future__ import annotations

from pov_compiler.models.client import (
    ChatModelClient,
    ModelClientConfig,
    get_model_cache_stats,
    maybe_wrap_with_cache,
)


def _apply_provider_defaults(cfg: ModelClientConfig) -> ModelClientConfig:
    provider = str(cfg.provider).strip().lower()
    if provider == "deepseek" and not cfg.base_url:
        cfg.base_url = "https://api.deepseek.com/v1"
    elif provider == "qwen" and not cfg.base_url:
        cfg.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif provider == "glm" and not cfg.base_url:
        cfg.base_url = "https://open.bigmodel.cn/api/paas/v4"
    return cfg


def make_client(cfg: ModelClientConfig) -> ChatModelClient:
    cfg = _apply_provider_defaults(cfg)
    provider = str(cfg.provider).strip().lower()
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


__all__ = ["ChatModelClient", "ModelClientConfig", "make_client", "get_model_cache_stats"]
