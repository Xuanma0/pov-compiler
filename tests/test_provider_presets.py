from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models.presets import get_preset, normalize_provider


def test_provider_presets_exist_and_nonempty() -> None:
    providers = ["openai", "openai_compat", "gemini", "deepseek", "qwen", "glm", "fake"]
    for p in providers:
        preset = get_preset(p)
        assert str(preset.name).strip() != ""
        if p != "fake":
            assert str(preset.default_api_key_env).strip().endswith("_API_KEY")
            assert str(preset.default_base_url).strip() != ""


def test_normalize_provider_alias_openai() -> None:
    assert normalize_provider("openai") == "openai_compat"
    assert normalize_provider("openai_compatible") == "openai_compat"


def test_configs_have_no_api_key_literal_field() -> None:
    cfg_dir = ROOT / "configs"
    for p in cfg_dir.glob("*.y*ml"):
        text = p.read_text(encoding="utf-8")
        assert re.search(r"(?mi)^\\s*api_key\\s*:", text) is None

