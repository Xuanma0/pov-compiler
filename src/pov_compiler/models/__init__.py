from __future__ import annotations

from pov_compiler.models.client import ChatModelClient, ModelClientConfig


def _apply_provider_defaults(cfg: ModelClientConfig) -> ModelClientConfig:
    if cfg.provider == "deepseek" and not cfg.base_url:
        cfg.base_url = "https://api.deepseek.com/v1"
    elif cfg.provider == "qwen" and not cfg.base_url:
        cfg.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif cfg.provider == "glm" and not cfg.base_url:
        cfg.base_url = "https://open.bigmodel.cn/api/paas/v4"
    return cfg


def make_client(cfg: ModelClientConfig) -> ChatModelClient:
    cfg = _apply_provider_defaults(cfg)
    provider = str(cfg.provider).strip().lower()
    if provider == "fake":
        from pov_compiler.models.fake import FakeModelClient

        return FakeModelClient(cfg)
    if provider in {"openai_compat", "deepseek"}:
        from pov_compiler.models.openai_compat import OpenAICompatClient

        return OpenAICompatClient(cfg)
    if provider == "gemini":
        from pov_compiler.models.gemini import GeminiClient

        return GeminiClient(cfg)
    if provider in {"qwen", "glm"}:
        mode = str(cfg.extra.get("mode", "openai_compat")).strip().lower() if isinstance(cfg.extra, dict) else "openai_compat"
        if mode != "openai_compat":
            raise RuntimeError(
                f"provider '{provider}' native mode is not implemented in v1.33; "
                "use OpenAI-compatible mode with provider=openai_compat or set extra.mode=openai_compat"
            )
        from pov_compiler.models.openai_compat import OpenAICompatClient

        return OpenAICompatClient(cfg)
    raise RuntimeError(f"unsupported model provider: {provider}")


__all__ = ["ChatModelClient", "ModelClientConfig", "make_client"]
