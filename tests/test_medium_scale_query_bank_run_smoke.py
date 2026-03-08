from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_manifest(manifest_name: str, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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


def test_medium_scale_query_bank_run_smoke(tmp_path: Path) -> None:
    out_v1 = tmp_path / "v1"
    out_v2 = tmp_path / "v2"
    proc_v1 = _run_manifest("v1.54_main_fake_v1.yaml", out_v1)
    proc_v2 = _run_manifest("v1.54_main_fake_v2.yaml", out_v2)
    assert proc_v1.returncode == 0, proc_v1.stderr or proc_v1.stdout
    assert proc_v2.returncode == 0, proc_v2.stderr or proc_v2.stdout

    summary_v1 = json.loads((out_v1 / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    summary_v2 = json.loads((out_v2 / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    assert summary_v1.get("query_bank_id") == "core_real_v1"
    assert summary_v2.get("query_bank_id") == "core_real_v2_candidate"
    assert summary_v1.get("run_signature_hash")
    assert summary_v1.get("run_signature_hash") == summary_v2.get("run_signature_hash")
    assert summary_v1.get("compare_pair_id") == "v1_vs_v2_medium_fake"
    assert summary_v2.get("compare_pair_id") == "v1_vs_v2_medium_fake"
    assert summary_v1.get("source_query_bank_id") == "core_real_v1"
    assert summary_v2.get("source_query_bank_id") == "core_real_v2_candidate"
