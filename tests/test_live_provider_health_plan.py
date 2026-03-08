from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_provider_health_plan(tmp_path: Path) -> None:
    out_dir = tmp_path / "provider_reachability"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_live_provider_health.py"),
            "--config",
            str(ROOT / "configs" / "providers" / "live_provider_health_v1.yaml"),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "proof_status=ok" in proc.stdout
    assert "real_call_status=ok" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("reachable") is True
    assert snapshot.get("proof_status") == "ok"
    assert snapshot.get("real_call_status") == "ok"
    assert snapshot.get("structured_output_supported") is True
