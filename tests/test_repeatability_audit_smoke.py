from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repeatability_audit_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_main_real_benchmark.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.51_main_fake_repeat.yaml"),
            "--mode",
            "pilot",
            "--out_dir",
            str(suite_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    out_dir = tmp_path / "repeatability_audit"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_repeatability_audit.py"),
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
    assert "saved_table=" in proc.stdout
    assert "repeatability_status=" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("repeatability_status") in {"ok", "partial", "weak"}
    assert snapshot.get("repeat_runs_total", 0) >= 1
    table_text = (out_dir / "tables" / "table_repeatability_audit.csv").read_text(encoding="utf-8")
    assert "stability_flag" in table_text
    assert (out_dir / "figures" / "fig_repeatability_variance.png").exists()
