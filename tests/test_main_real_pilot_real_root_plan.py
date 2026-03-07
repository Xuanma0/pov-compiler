from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_real_pilot_real_root_plan(tmp_path: Path) -> None:
    out_dir = tmp_path / "v146_main_real_dry"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_main_real_benchmark.py"),
        "--manifest",
        str(ROOT / "configs" / "benchmarks" / "v1.46_main_real_pilot.yaml"),
        "--dry-collect",
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_provider_telemetry=skipped" in proc.stdout
    assert "gate_status=skipped" in proc.stdout
    payload = json.loads((out_dir / "manifest" / "dry_collect_snapshot.json").read_text(encoding="utf-8"))
    telemetry = payload.get("telemetry", {})
    assert payload.get("suite_id") == "v1.46_main_real_pilot"
    assert telemetry.get("enabled") is True
    assert telemetry.get("require_usage") is False
    assert telemetry.get("require_latency") is True
    assert telemetry.get("variants", {}).get("real", {}).get("provider") == "openai"
