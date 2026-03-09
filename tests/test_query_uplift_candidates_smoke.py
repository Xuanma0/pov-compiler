from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_query_uplift_candidates_smoke(tmp_path: Path) -> None:
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

    out_dir = tmp_path / "query_uplift_candidates"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_query_uplift_candidates.py"),
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
    assert "saved_candidate=" in proc.stdout
    summary = json.loads(
        (out_dir / "query_uplift_candidates" / "uplift_summary.json").read_text(encoding="utf-8")
    )
    assert summary.get("source_query_bank_id") == "core_real_v1"
    assert summary.get("candidate_count", 0) + summary.get("analysis_only_count", 0) + summary.get("drop_count", 0) >= 1
    assert (out_dir / "query_uplift_candidates" / "candidate_queries.yaml").exists()
    assert (out_dir / "query_uplift_candidates" / "analysis_only_queries.yaml").exists()
    assert (out_dir / "query_uplift_candidates" / "drop_queries.yaml").exists()
