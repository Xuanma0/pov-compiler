from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sample_size_recommendation_smoke(tmp_path: Path) -> None:
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

    out_dir = tmp_path / "sample_size_recommendation"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "recommend_sample_size.py"),
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
    assert "sample_size_recommendation_status=" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("sample_size_recommendation_status") in {"ok", "range_only", "weak"}
    csv_text = (out_dir / "tables" / "table_sample_size_recommendation.csv").read_text(encoding="utf-8")
    assert "recommended_n_pairs_min" in csv_text
    assert "recommended_n_uids_min" in csv_text
