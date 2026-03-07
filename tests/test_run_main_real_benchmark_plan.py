from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_main_real_benchmark_plan(tmp_path: Path) -> None:
    out_dir = tmp_path / "v144_main_real_dry"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_main_real_benchmark.py"),
        "--manifest",
        str(ROOT / "configs" / "benchmarks" / "v1.44_main_real.yaml"),
        "--dry-collect",
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_suite=" in proc.stdout
    assert "gate_status=skipped" in proc.stdout
    snapshot_path = out_dir / "manifest" / "dry_collect_snapshot.json"
    assert snapshot_path.exists()
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload.get("suite_id") == "v1.44_main_real"
    assert payload.get("paper_map", {}).get("paper_map_id") == "main_result_map_v1"
