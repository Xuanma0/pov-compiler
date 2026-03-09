from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_manifest(manifest_name: str, out_dir: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_main_real_benchmark.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / manifest_name),
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


def test_query_bank_compare_smoke(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _run_manifest("v1.54_main_fake_v1.yaml", run_a)
    _run_manifest("v1.54_main_fake_v2.yaml", run_b)

    out_dir = tmp_path / "query_bank_compare"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_query_banks.py"),
            "--run_a",
            str(run_a),
            "--run_b",
            str(run_b),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    summary = json.loads((out_dir / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    assert summary.get("alignment_ok") is True
    assert summary.get("mismatch_reasons") == []
    assert summary.get("query_bank_a_id") == "core_real_v1"
    assert summary.get("query_bank_b_id") == "core_real_v2_candidate"
    assert (out_dir / "compare" / "tables" / "table_query_bank_compare.csv").exists()
    assert (out_dir / "compare" / "figures" / "fig_query_bank_delta.png").exists()
