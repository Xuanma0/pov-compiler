from __future__ import annotations

from typing import Any


def _to_int(value: Any) -> int:
    try:
        out = int(round(float(value)))
    except Exception:
        return 0
    return max(0, out)


def estimate_cost_usd(
    *,
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    prompt_text: str | None = None,
    completion_text: str | None = None,
) -> float | None:
    """Best-effort cost estimate via litellm; returns None when unavailable."""
    try:
        from litellm import completion_cost, token_counter  # type: ignore
    except Exception:
        return None

    p_tokens = _to_int(prompt_tokens)
    c_tokens = _to_int(completion_tokens)
    if p_tokens <= 0 and prompt_text is not None:
        try:
            p_tokens = _to_int(token_counter(model=str(model), text=str(prompt_text)))
        except Exception:
            p_tokens = 0
    if c_tokens <= 0 and completion_text is not None:
        try:
            c_tokens = _to_int(token_counter(model=str(model), text=str(completion_text)))
        except Exception:
            c_tokens = 0
    if p_tokens <= 0 and c_tokens <= 0:
        return None
    try:
        value = completion_cost(model=str(model), prompt_tokens=p_tokens, completion_tokens=c_tokens)
        return float(value)
    except Exception:
        return None
