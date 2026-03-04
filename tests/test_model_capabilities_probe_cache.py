from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models.capabilities import resolve_model_capabilities
from pov_compiler.models.client import ModelClientConfig


def test_model_capabilities_probe_cache_hits(monkeypatch, tmp_path: Path) -> None:
    calls = {"n": 0}

    def _fake_probe(client, *, timeout_s: int = 8):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return True, ""

    monkeypatch.setattr("pov_compiler.models.capabilities._probe_json_schema", _fake_probe)
    cfg = ModelClientConfig(
        provider="fake",
        model="fake-probe-v1",
        model_cache_enabled=False,
    )
    cache_dir = tmp_path / "caps_cache"
    caps1, meta1 = resolve_model_capabilities(
        cfg,
        probe=True,
        probe_timeout_s=5,
        probe_cache_dir=cache_dir,
    )
    caps2, meta2 = resolve_model_capabilities(
        cfg,
        probe=True,
        probe_timeout_s=5,
        probe_cache_dir=cache_dir,
    )

    assert caps1.supports_json_schema is True
    assert caps2.supports_json_schema is True
    assert calls["n"] == 1
    assert bool(meta1.get("probe_used")) is True
    assert bool(meta2.get("probe_cached")) is True

    cache_file = cache_dir / "model_capabilities_cache.json"
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and payload
