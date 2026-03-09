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
            "main_real",
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_persistent_memory_main_compare_smoke(tmp_path: Path) -> None:
    run_a = tmp_path / "baseline"
    run_b = tmp_path / "persistent"
    _run_manifest("v1.58_main_fake_baseline.yaml", run_a)
    _run_manifest("v1.58_main_fake_persistent.yaml", run_b)

    out_dir = tmp_path / "compare_root"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_persistent_memory_main.py"),
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
    assert summary.get("persistent_memory_main_status") in {"improved", "no_change", "regressed"}
    assert (out_dir / "compare" / "tables" / "table_persistent_memory_main_compare.csv").exists()
    assert (out_dir / "compare" / "tables" / "table_persistent_memory_main_significance.csv").exists()
    assert (out_dir / "compare" / "figures" / "fig_persistent_memory_main_delta.png").exists()


def test_persistent_memory_main_compare_fails_fast_on_mismatch(tmp_path: Path) -> None:
    run_a = tmp_path / "baseline"
    run_b = tmp_path / "persistent"
    _run_manifest("v1.58_main_fake_baseline.yaml", run_a)
    _run_manifest("v1.58_main_fake_persistent.yaml", run_b)
    summary_b_path = run_b / "compare" / "compare_summary.json"
    summary_b = json.loads(summary_b_path.read_text(encoding="utf-8"))
    summary_b["uid_set_id"] = "mismatched_uid_set"
    summary_b_path.write_text(json.dumps(summary_b, ensure_ascii=False, indent=2), encoding="utf-8")

    out_dir = tmp_path / "compare_root"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_persistent_memory_main.py"),
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
    assert proc.returncode != 0
    summary = json.loads((out_dir / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    assert summary.get("alignment_ok") is False
    assert "uid_set_id_mismatch" in summary.get("mismatch_reasons", [])
