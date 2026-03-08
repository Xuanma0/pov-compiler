from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_uplift_smoke(tmp_path: Path) -> None:
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
    submission_pack = suite_dir / "submission_pack"
    assert (submission_pack / "repeatability_audit" / "snapshot.json").exists()
    assert (submission_pack / "sample_size_recommendation" / "snapshot.json").exists()
    assert (submission_pack / "query_uplift_candidates" / "query_uplift_candidates" / "uplift_summary.json").exists()
    readme = (submission_pack / "README.md").read_text(encoding="utf-8")
    assert "Repeatability Audit" in readme
    assert "Sample Size Recommendation" in readme
    assert "Query Uplift Candidates" in readme
    snapshot = json.loads((submission_pack / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("repeatability_audit_dir")
    assert snapshot.get("sample_size_recommendation_dir")
    assert snapshot.get("query_uplift_candidates_dir")
