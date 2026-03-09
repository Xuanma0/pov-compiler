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


def test_query_promotion_pack_smoke(tmp_path: Path) -> None:
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
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_query_strength_audit.py"),
            "--suite-dir",
            str(suite_dir),
            "--out_dir",
            str(suite_dir / "query_strength_audit"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    out_dir = tmp_path / "query_promotion_pack"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_query_promotion_pack.py"),
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
    assert "saved_promoted=" in proc.stdout
    summary = json.loads((out_dir / "query_pack" / "promotion_summary.json").read_text(encoding="utf-8"))
    assert summary.get("source_query_bank_id") == "core_real_v1"
    assert summary.get("promoted_count", 0) >= 1
    promoted_yaml = (out_dir / "query_pack" / "promoted_queries.yaml").read_text(encoding="utf-8")
    assert "query_id:" in promoted_yaml
    assert "decision" in promoted_yaml or "chain" in promoted_yaml

