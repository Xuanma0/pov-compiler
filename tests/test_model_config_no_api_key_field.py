from __future__ import annotations

from dataclasses import fields
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models.client import ModelClientConfig


def test_model_client_config_has_no_api_key_field() -> None:
    names = {f.name for f in fields(ModelClientConfig)}
    assert "api_key" not in names
    cfg = ModelClientConfig(provider="openai_compat", model="gpt-4o-mini")
    assert cfg.api_key_env == "OPENAI_API_KEY"
