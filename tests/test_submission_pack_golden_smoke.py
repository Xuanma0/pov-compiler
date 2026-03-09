from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_golden_smoke(tmp_path: Path) -> None:
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
    submission_pack = suite_dir / "submission_pack"
    assert (submission_pack / "provider_reachability" / "summary.json").exists()
    assert (submission_pack / "golden_real_sample" / "snapshot.json").exists()
    readme = (submission_pack / "README.md").read_text(encoding="utf-8")
    assert "Provider Reachability" in readme
    assert "Golden Real Sample" in readme
    snapshot = json.loads((submission_pack / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("provider_reachability_dir")
    assert snapshot.get("golden_real_sample_dir")
