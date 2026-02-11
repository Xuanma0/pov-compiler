from __future__ import annotations

from typing import Literal

from pov_compiler.schemas import Output, TokenCodec


Variant = Literal[
    "raw_events_only",
    "highlights_only",
    "highlights_plus_tokens",
    "highlights_plus_decisions",
    "full",
]


ALL_VARIANTS: list[Variant] = [
    "raw_events_only",
    "highlights_only",
    "highlights_plus_tokens",
    "highlights_plus_decisions",
    "full",
]


def apply_variant(output: Output, variant: str) -> Output:
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    cloned = output.model_copy(deep=True)
    if variant == "raw_events_only":
        cloned.highlights = []
        cloned.token_codec = TokenCodec(version="0.2", vocab=list(cloned.token_codec.vocab), tokens=[])
        cloned.decision_points = []
    elif variant == "highlights_only":
        cloned.token_codec = TokenCodec(version="0.2", vocab=list(cloned.token_codec.vocab), tokens=[])
        cloned.decision_points = []
    elif variant == "highlights_plus_tokens":
        cloned.decision_points = []
    elif variant == "highlights_plus_decisions":
        cloned.token_codec = TokenCodec(version="0.2", vocab=list(cloned.token_codec.vocab), tokens=[])
    elif variant == "full":
        pass

    return cloned
