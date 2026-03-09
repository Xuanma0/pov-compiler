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
            "main_real",
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_large_sample_persistent_memory_run_smoke(tmp_path: Path) -> None:
    out_baseline = tmp_path / "baseline"
    out_persistent = tmp_path / "persistent"
    proc_baseline = _run_manifest("v1.58_main_fake_baseline.yaml", out_baseline)
    proc_persistent = _run_manifest("v1.58_main_fake_persistent.yaml", out_persistent)
    assert proc_baseline.returncode == 0, proc_baseline.stderr or proc_baseline.stdout
    assert proc_persistent.returncode == 0, proc_persistent.stderr or proc_persistent.stdout

    summary_baseline = json.loads((out_baseline / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    summary_persistent = json.loads((out_persistent / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    assert summary_baseline.get("query_bank_id") == "persistent_object_memory_core_v1"
    assert summary_persistent.get("query_bank_id") == "persistent_object_memory_core_v1"
    assert summary_baseline.get("query_bank_hash") == summary_persistent.get("query_bank_hash")
    assert summary_baseline.get("compare_pair_id") == "persistent_memory_main_fake"
    assert summary_persistent.get("compare_pair_id") == "persistent_memory_main_fake"
    assert summary_baseline.get("uid_set_id") == summary_persistent.get("uid_set_id")
    assert summary_baseline.get("sample_signature_hash") == summary_persistent.get("sample_signature_hash")
    assert summary_baseline.get("paired_contract_hash") == summary_persistent.get("paired_contract_hash")
    assert summary_baseline.get("object_memory_logic_variant") == "persistence_v1"
    assert summary_persistent.get("object_memory_logic_variant") == "persistent_v2"
    assert summary_baseline.get("run_signature_hash")
    assert summary_persistent.get("run_signature_hash")
