from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_real_admission_plan(tmp_path: Path) -> None:
    out_dir = tmp_path / "v147_main_real_dry"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_main_real_benchmark.py"),
        "--manifest",
        str(ROOT / "configs" / "benchmarks" / "v1.47_main_real_admission.yaml"),
        "--dry-collect",
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_delta_audit=skipped" in proc.stdout
    assert "admission_status=skipped" in proc.stdout
    payload = json.loads((out_dir / "manifest" / "dry_collect_snapshot.json").read_text(encoding="utf-8"))
    assert payload.get("suite_id") == "v1.47_main_real_admission"
    assert payload.get("admission", {}).get("profile") == "main_real"
    assert payload.get("admission", {}).get("min_selected_uids") == 3
    assert payload.get("telemetry", {}).get("require_latency") is True
