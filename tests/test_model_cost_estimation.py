from __future__ import annotations

import builtins
import types
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.models.cost import estimate_cost_usd


def test_model_cost_estimation_with_litellm_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.SimpleNamespace()

    def _completion_cost(model: str, prompt_tokens: int, completion_tokens: int):  # type: ignore[no-untyped-def]
        return 0.001 * float(prompt_tokens + completion_tokens)

    def _token_counter(model: str, text: str):  # type: ignore[no-untyped-def]
        return len(str(text).split())

    fake_mod.completion_cost = _completion_cost
    fake_mod.token_counter = _token_counter
    monkeypatch.setitem(sys.modules, "litellm", fake_mod)

    value = estimate_cost_usd(model="fake-model", prompt_tokens=10, completion_tokens=5)
    assert value is not None
    assert float(value) == pytest.approx(0.015, rel=1e-6)


def test_model_cost_estimation_returns_none_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    orig_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "litellm":
            raise ImportError("no litellm")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    value = estimate_cost_usd(model="fake-model", prompt_tokens=10, completion_tokens=5)
    assert value is None
