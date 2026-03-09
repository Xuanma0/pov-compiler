from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_freeze_benchmark_run_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    (suite_dir / "compare" / "tables").mkdir(parents=True, exist_ok=True)
    (suite_dir / "compare" / "figures").mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest" / "query_banks").mkdir(parents=True, exist_ok=True)

    (suite_dir / "compare" / "tables" / "table_main_results.csv").write_text("task,budget_key\nnlq,20/50/4\n", encoding="utf-8")
    (suite_dir / "compare" / "tables" / "table_main_results.md").write_text("# main\n", encoding="utf-8")
    (suite_dir / "compare" / "figures" / "fig_main_budget_frontier.png").write_bytes(b"PNG")
    (suite_dir / "compare" / "figures" / "fig_main_budget_frontier.pdf").write_bytes(b"PDF")
    (suite_dir / "compare" / "compare_summary.json").write_text(json.dumps({"suite_id": "suite_freeze"}), encoding="utf-8")
    (suite_dir / "compare" / "snapshot.json").write_text(json.dumps({"suite_id": "suite_freeze"}), encoding="utf-8")
    (suite_dir / "manifest" / "experiment_manifest.yaml").write_text("suite_id: suite_freeze\n", encoding="utf-8")
    (suite_dir / "manifest" / "query_bank_lock.json").write_text(
        json.dumps({"primary": {"query_bank_id": "core_real_v1", "query_bank_hash": "abc123"}}),
        encoding="utf-8",
    )
    (suite_dir / "manifest" / "query_banks" / "core_real_v1.yaml").write_text("query_bank_id: core_real_v1\n", encoding="utf-8")

    out_dir = tmp_path / "freeze"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "freeze_benchmark_run.py"),
        "--suite-dir",
        str(suite_dir),
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_freeze_manifest=" in proc.stdout
    assert (out_dir / "freeze_manifest.json").exists()
    assert (out_dir / "artifacts_sha256.csv").exists()
    payload = json.loads((out_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert payload.get("suite_id") == "suite_freeze"
    assert payload.get("query_bank_id") == "core_real_v1"
