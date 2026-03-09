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


def test_query_strength_audit_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_benchmark_suite.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.48_main_fake_pilot.yaml"),
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
    write_provider_telemetry_outputs(suite_dir=suite_dir, out_dir=suite_dir / "provider_telemetry")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_result_diagnosis.py"),
            "--suite-dir",
            str(suite_dir),
            "--out_dir",
            str(suite_dir / "result_diagnosis"),
            "--provider-telemetry-dir",
            str(suite_dir / "provider_telemetry"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    out_dir = tmp_path / "query_strength_audit"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_query_strength_audit.py"),
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
    assert "query_strength_summary=" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("query_bank_id") == "core_real_v1"
    assert snapshot.get("main_recommendation")
    table_text = (out_dir / "tables" / "table_query_strength_audit.csv").read_text(encoding="utf-8")
    assert "query_group" in table_text
    assert "recommended_action" in table_text
    assert "repo_summary" in table_text
