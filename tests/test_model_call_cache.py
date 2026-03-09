from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models import ModelClientConfig, get_model_cache_stats, make_client
from pov_compiler.models.fake import FakeModelClient


def test_model_call_cache_hits_and_misses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    orig = FakeModelClient.complete_json

    def _wrapped(self, system, user, *, timeout_s, max_tokens, temperature):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return orig(
            self,
            system,
            user,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    monkeypatch.setattr(FakeModelClient, "complete_json", _wrapped)
    cfg = ModelClientConfig(
        provider="fake",
        model="fake-cache-test",
        model_cache_enabled=True,
        model_cache_dir=str(tmp_path / "model_cache"),
    )
    client = make_client(cfg)
    a1 = client.complete_json("sys", "user", timeout_s=10, max_tokens=64, temperature=0.1)
    a2 = client.complete_json("sys", "user", timeout_s=10, max_tokens=64, temperature=0.1)
    b1 = client.complete_json("sys", "user-2", timeout_s=10, max_tokens=64, temperature=0.1)

    assert isinstance(a1, dict) and isinstance(a2, dict) and isinstance(b1, dict)
    assert calls["n"] == 2

    stats = get_model_cache_stats(client)
    assert bool(stats.get("enabled")) is True
    assert int(stats.get("hit", 0)) >= 1
    assert int(stats.get("miss", 0)) >= 2
