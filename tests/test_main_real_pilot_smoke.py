from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_real_pilot_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "v145_fake_pilot"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_main_real_benchmark.py"),
        "--manifest",
        str(ROOT / "configs" / "benchmarks" / "v1.45_main_fake_pilot.yaml"),
        "--mode",
        "pilot",
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_result_diagnosis=" in proc.stdout
    assert "gate_status=ok" in proc.stdout
    assert (out_dir / "result_diagnosis" / "tables" / "table_result_diagnosis.csv").exists()
    assert (out_dir / "result_diagnosis" / "report.md").exists()
    assert (out_dir / "paper_ready" / "result_diagnosis" / "tables" / "table_result_diagnosis.csv").exists()
    assert (out_dir / "submission_pack" / "result_diagnosis" / "tables" / "table_result_diagnosis.csv").exists()
    health_snapshot = json.loads((out_dir / "result_health" / "snapshot.json").read_text(encoding="utf-8"))
    assert health_snapshot.get("diagnosis_available") is True
    assert str(health_snapshot.get("diagnosis_dir", "")).endswith("result_diagnosis")
