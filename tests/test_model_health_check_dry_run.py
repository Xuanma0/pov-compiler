from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_model_health_check_dry_run_providers() -> None:
    providers = [("fake", "fake-decision-v1"), ("deepseek", "deepseek-chat"), ("qwen", "qwen-plus"), ("glm", "glm-5")]
    for provider, model in providers:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "model_health_check.py"),
            "--provider",
            provider,
            "--model",
            model,
            "--dry-run",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr or proc.stdout
        text = (proc.stdout or "").lower()
        assert "status=dry_run_ok" in text
        assert "provider=" in text
        assert "api_key_present=" in text
        assert "api_mode=" in text
        assert "api_mode_order=" in text
        banned = ["sk-", "bearer ", "authorization:", "api_key="]
        for token in banned:
            assert token not in text
