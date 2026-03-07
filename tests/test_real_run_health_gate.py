from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_run_health_gate_failure(tmp_path: Path) -> None:
    compare_source = tmp_path / "compare_source"
    signal_dir = compare_source / "signal_selection"
    signal_dir.mkdir(parents=True, exist_ok=True)
    (signal_dir / "selected_uids.txt").write_text("u1\n", encoding="utf-8")
    (signal_dir / "coverage.csv").write_text(
        "uid,coverage_score,missing_place,missing_interaction,missing_lost_object\nu1,1.0,1,1,1\n",
        encoding="utf-8",
    )
    (signal_dir / "snapshot.json").write_text(json.dumps({"selection_mode": "auto_signal_cache"}), encoding="utf-8")

    suite_dir = tmp_path / "suite"
    (suite_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (suite_dir / "ledger").mkdir(parents=True, exist_ok=True)
    (suite_dir / "compare" / "tables").mkdir(parents=True, exist_ok=True)
    (suite_dir / "significance" / "tables").mkdir(parents=True, exist_ok=True)

    (suite_dir / "manifest" / "experiment_manifest.yaml").write_text(
        "\n".join(
            [
                "suite_id: suite_gate_fail",
                'suite_version: "1.44"',
                "selection:",
                f"  compare_dir: {compare_source}",
                "  mode: auto_signal_cache",
                "health_gate_profile: main_real",
            ]
        ),
        encoding="utf-8",
    )
    (suite_dir / "ledger" / "runs.jsonl").write_text(json.dumps({"task": "nlq", "status": "ok"}) + "\n", encoding="utf-8")
    (suite_dir / "compare" / "compare_summary.json").write_text(
        json.dumps({"suite_id": "suite_gate_fail", "compare_dir": str(compare_source)}),
        encoding="utf-8",
    )
    (suite_dir / "compare" / "snapshot.json").write_text(json.dumps({"suite_id": "suite_gate_fail"}), encoding="utf-8")
    (suite_dir / "compare" / "tables" / "table_main_results.csv").write_text(
        "\n".join(
            [
                "task,budget_key,budget_seconds,primary_metric,label_a,label_b,value_a,value_b,delta,n_rows_a,n_rows_b,status",
                "nlq,20/50/4,20,nlq_full_hit_at_k_strict,stub,real,0.4,0.4,0.0,1,1,ok",
            ]
        ),
        encoding="utf-8",
    )
    (suite_dir / "compare" / "tables" / "table_significance.csv").write_text(
        "\n".join(
            [
                "task,budget_key,budget_seconds,primary_metric,baseline_label,treatment_label,n_pairs,status",
                "nlq,20/50/4,20,nlq_full_hit_at_k_strict,stub,real,0,insufficient_pairs",
            ]
        ),
        encoding="utf-8",
    )
    (suite_dir / "significance" / "tables" / "table_significance_main.csv").write_text(
        "\n".join(
            [
                "task,budget_key,budget_seconds,primary_metric,baseline_label,treatment_label,n_pairs,mean_delta,status",
                "nlq,20/50/4,20,nlq_full_hit_at_k_strict,stub,real,0,0.0,insufficient_pairs",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "health"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_result_health.py"),
        "--suite-dir",
        str(suite_dir),
        "--out_dir",
        str(out_dir),
        "--gate-profile",
        "main_real",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "gate_status=fail" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("gate", {}).get("gate_status") == "fail"
    assert snapshot.get("gate", {}).get("gate_fail_reasons")
