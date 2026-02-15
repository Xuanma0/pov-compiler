from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.security_scan_secrets as scan_mod


def test_security_scan_detects_masked_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    tracked = repo / "sample.txt"
    tracked.write_text(
        "token = sk-ABCDEFGHIJKLMNOPQRSTUVWX123456\n"
        "url=https://example.test/path?api_key=THISSHOULDNEVERPRINT12345\n",
        encoding="utf-8",
    )

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="sample.txt\n", stderr="")

    monkeypatch.setattr(scan_mod.subprocess, "run", _fake_run)
    monkeypatch.chdir(repo)

    rc = scan_mod.main([])
    out = capsys.readouterr().out
    assert rc == 2
    assert "found_count=1" in out
    assert "sk-***" in out
    assert "api***345" in out or "api***" in out
    assert "ABCDEFGHIJKLMNOP" not in out
    assert "THISSHOULDNEVERPRINT" not in out
