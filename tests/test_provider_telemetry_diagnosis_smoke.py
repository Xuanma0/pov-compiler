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


def test_provider_telemetry_diagnosis_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_benchmark_suite.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.46_main_fake_pilot.yaml"),
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
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_statistical_significance.py"),
            "--suite_dir",
            str(suite_dir),
            "--out_dir",
            str(suite_dir / "significance"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_result_health.py"),
            "--suite-dir",
            str(suite_dir),
            "--out_dir",
            str(suite_dir / "result_health"),
            "--gate-profile",
            "main_real",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    telemetry_out = write_provider_telemetry_outputs(suite_dir=suite_dir, out_dir=suite_dir / "provider_telemetry")
    summary_payload = json.loads(Path(telemetry_out["summary_json"]).read_text(encoding="utf-8"))
    assert summary_payload.get("availability") == "ok"
    assert summary_payload.get("calls_total") == 12
    assert summary_payload.get("model_cost_known_rate") == 1.0

    out_dir = tmp_path / "result_diagnosis"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_result_diagnosis.py"),
            "--suite-dir",
            str(suite_dir),
            "--out_dir",
            str(out_dir),
            "--provider-telemetry-dir",
            str(suite_dir / "provider_telemetry"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("provider_noise_summary", {}).get("availability") == "ok"
    assert snapshot.get("provider_noise_summary", {}).get("calls_total") == 12
    table_text = (out_dir / "tables" / "table_result_diagnosis.csv").read_text(encoding="utf-8")
    assert "variant_telemetry" in table_text
    assert "fake_openai" in table_text
