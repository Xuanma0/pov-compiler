from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_golden_real_sample_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_main_real_benchmark.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.50_main_fake_golden.yaml"),
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
    out_dir = tmp_path / "golden_real_sample"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_golden_real_sample.py"),
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
    assert "golden_sample_status=ok" in proc.stdout
    snapshot_path = out_dir / "snapshot.json"
    if not snapshot_path.exists():
        snapshot_path = out_dir / "golden_real_sample" / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot.get("golden_sample_status") == "ok"
    assert snapshot.get("source_provider_proof", {}).get("proof_status") == "ok"
    query_set_path = out_dir / "query_set.yaml"
    expected_outputs_path = out_dir / "expected_outputs.json"
    if not query_set_path.exists():
        query_set_path = out_dir / "golden_real_sample" / "query_set.yaml"
    if not expected_outputs_path.exists():
        expected_outputs_path = out_dir / "golden_real_sample" / "expected_outputs.json"
    assert query_set_path.exists()
    assert expected_outputs_path.exists()
