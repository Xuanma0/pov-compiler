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


def test_query_bank_promotion_decision_smoke(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _run_manifest("v1.54_main_fake_v1.yaml", run_a)
    _run_manifest("v1.54_main_fake_v2.yaml", run_b)

    compare_dir = tmp_path / "compare_root"
    compare_proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_query_banks.py"),
            "--run_a",
            str(run_a),
            "--run_b",
            str(run_b),
            "--out_dir",
            str(compare_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert compare_proc.returncode == 0, compare_proc.stderr or compare_proc.stdout

    out_dir = tmp_path / "promotion_decision"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_query_bank_promotion_decision.py"),
            "--compare_dir",
            str(compare_dir),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    summary = dict(snapshot.get("promotion_decision_summary", {}))
    assert summary.get("promotion_decision") in {
        "promote_v2",
        "keep_v1",
        "expand_sample_first",
        "consider_sam3_next",
    }
    assert summary.get("recommended_next_step")
    assert (out_dir / "tables" / "table_query_bank_promotion_decision.csv").exists()
