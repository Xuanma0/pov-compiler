from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import test_fast as tf


def test_test_fast_xdist_enabled(monkeypatch, capsys) -> None:
    def _fake_run(cmd):
        return subprocess.CompletedProcess(
            args=list(cmd),
            returncode=0,
            stdout="passed\n",
            stderr="",
        )

    monkeypatch.setattr(tf, "_run", _fake_run)
    rc = tf.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "xdist_enabled=true" in out
    assert "-n auto" in out
    assert "returncode=0" in out


def test_test_fast_fallback_when_xdist_missing(monkeypatch, capsys) -> None:
    calls = {"n": 0}

    def _fake_run(cmd):
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(
                args=list(cmd),
                returncode=1,
                stdout="",
                stderr="ImportError: No module named xdist",
            )
        return subprocess.CompletedProcess(
            args=list(cmd),
            returncode=0,
            stdout="passed-fallback\n",
            stderr="",
        )

    monkeypatch.setattr(tf, "_run", _fake_run)
    rc = tf.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert calls["n"] == 2
    assert "xdist_enabled=false" in out
    assert "-m pytest -q" in out
    assert "-n auto" not in out.split("cmd=", 1)[1]
