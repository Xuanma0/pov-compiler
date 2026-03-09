from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.provider_telemetry import write_provider_telemetry_outputs


def test_provider_normalization_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_benchmark_suite.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.49_main_fake_pilot.yaml"),
            "--out_dir",
            str(suite_dir),
            "--mode",
            "collect-only",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    write_provider_telemetry_outputs(suite_dir=suite_dir, out_dir=suite_dir / "provider_telemetry")

    out_dir = tmp_path / "provider_normalization"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "normalize_provider_telemetry.py"),
            "--suite-dir",
            str(suite_dir),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "normalization_status=ok" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("normalization_status") == "ok"
    assert snapshot.get("real_call_status") == "simulated"
    table_text = (out_dir / "tables" / "table_provider_normalization.csv").read_text(encoding="utf-8")
    assert "prompt_tokens" in table_text
    assert "normalization_status" in table_text

