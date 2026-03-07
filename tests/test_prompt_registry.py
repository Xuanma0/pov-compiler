from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.prompts.registry import PromptRegistry, stable_prompt_hash, write_prompt_lock


def test_stable_prompt_hash_normalizes_newlines() -> None:
    assert stable_prompt_hash("line1\r\nline2\r\n") == stable_prompt_hash("line1\nline2\n")


def test_prompt_registry_builds_lock_from_repo_config(tmp_path: Path) -> None:
    registry_path = ROOT / "configs" / "prompts" / "registry_v1.yaml"
    registry = PromptRegistry.from_path(registry_path)
    lock = registry.build_prompt_lock(registry_path, "v1.42_main")

    assert lock["registry_id"] == "prompt_registry_v1"
    assert lock["profile"] == "v1.42_main"
    assert {item["task"] for item in lock["prompts"]} == {"decisions", "planner", "repository"}
    assert all(len(str(item["hash"])) == 64 for item in lock["prompts"])

    out_path = tmp_path / "prompt_lock.json"
    write_prompt_lock(lock, out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "v1.42_main"
    assert len(payload["prompts"]) == 3
