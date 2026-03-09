from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_fake_suite(out_dir: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_signal_uplift_pilot.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.53_query_bank_fake.yaml"),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_query_bank_rewrite_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    _run_fake_suite(suite_dir)
    out_dir = tmp_path / "rewrite"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rewrite_query_bank.py"),
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
    summary = json.loads((out_dir / "query_bank_rewrite" / "rewrite_summary.json").read_text(encoding="utf-8"))
    assert summary.get("source_query_bank_id") == "core_real_v1"
    assert summary.get("candidate_count") == 6
    assert summary.get("analysis_only_count") == 2
    candidate_yaml = (out_dir / "query_bank_rewrite" / "core_real_v2_candidate.yaml").read_text(encoding="utf-8")
    assert "query_bank_id: core_real_v2_candidate" in candidate_yaml
    assert "group: lost_object" in candidate_yaml
    assert "group: chain" in candidate_yaml
    assert "interaction_object=" in candidate_yaml
    assert "lost_object=" in candidate_yaml
