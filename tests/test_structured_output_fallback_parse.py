from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models.client import ModelClientConfig
from pov_compiler.models.structured_output import generate_structured


class _EmbeddedJsonClient:
    def __init__(self) -> None:
        self.cfg = ModelClientConfig(provider="fake", model="fake-structured", model_cache_enabled=False)

    def generate_text(self, system: str, user: str, *, timeout_s: int, max_tokens: int, temperature: float, **kwargs):  # type: ignore[no-untyped-def]
        return "prefix... {\"video_id\":\"v1\",\"t0_ms\":0,\"t1_ms\":1000,\"goals\":[\"x\"]} ...suffix", {"mode": "fake"}


def test_structured_output_extracts_embedded_json() -> None:
    obj, raw, meta = generate_structured(
        _EmbeddedJsonClient(),
        schema_name="repo_summary_v0",
        schema_json={"type": "object"},
        system_prompt="sys",
        user_prompt="usr",
    )
    assert isinstance(obj, dict)
    assert obj.get("video_id") == "v1"
    assert "prefix" in raw
    assert bool(meta.get("parse_ok")) is True

