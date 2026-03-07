from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_real_small_pilot_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "v148_fake_pilot"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_main_real_benchmark.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.48_main_fake_pilot.yaml"),
            "--mode",
            "pilot",
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_admission_calibration=" in proc.stdout
    assert "saved_query_strength_audit=" in proc.stdout
    assert "calibration_status=" in proc.stdout
    assert (out_dir / "admission_calibration" / "tables" / "table_admission_calibration.csv").exists()
    assert (out_dir / "query_strength_audit" / "tables" / "table_query_strength_audit.csv").exists()
    assert (out_dir / "paper_ready" / "admission_calibration" / "snapshot.json").exists()
    assert (out_dir / "paper_ready" / "query_strength_audit" / "snapshot.json").exists()
    assert (out_dir / "submission_pack" / "admission_calibration" / "snapshot.json").exists()
    assert (out_dir / "submission_pack" / "query_strength_audit" / "snapshot.json").exists()
    health_snapshot = json.loads((out_dir / "result_health" / "snapshot.json").read_text(encoding="utf-8"))
    assert health_snapshot.get("admission_calibration_available") is True
    assert health_snapshot.get("query_strength_audit_available") is True
