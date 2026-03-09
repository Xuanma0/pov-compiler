from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_export_paper_ready_mainline_panels_smoke(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    _write_csv(
        compare_dir / "nlq_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "nlq_full_hit_at_k_strict": 0.40},
            {"budget_key": "40/100/8", "budget_seconds": 40, "nlq_full_hit_at_k_strict": 0.55},
        ],
    )
    _write_csv(
        compare_dir / "nlq_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "nlq_full_hit_at_k_strict": 0.45},
            {"budget_key": "40/100/8", "budget_seconds": 40, "nlq_full_hit_at_k_strict": 0.60},
        ],
    )
    _write_csv(
        compare_dir / "streaming_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.50, "e2e_ms_p95": 12.0},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.60, "e2e_ms_p95": 13.0},
        ],
    )
    _write_csv(
        compare_dir / "streaming_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.58, "e2e_ms_p95": 12.5},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.68, "e2e_ms_p95": 13.5},
        ],
    )

    suite_dir = tmp_path / "suite"
    (suite_dir / "manifest" / "query_banks").mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest" / "experiment_manifest.yaml").write_text(
        "suite_id: v1.59_mainline_cleanup\n",
        encoding="utf-8",
    )
    (suite_dir / "manifest" / "manifest_resolved.json").write_text("{}", encoding="utf-8")
    (suite_dir / "manifest" / "prompt_lock.json").write_text("{}", encoding="utf-8")
    (suite_dir / "manifest" / "query_bank_lock.json").write_text(
        json.dumps({"query_bank_id": "persistent_object_memory_core_v1"}),
        encoding="utf-8",
    )
    (suite_dir / "manifest" / "query_banks" / "persistent_object_memory_core_v1.yaml").write_text(
        "query_bank_id: persistent_object_memory_core_v1\n",
        encoding="utf-8",
    )
    (suite_dir / "ledger").mkdir(parents=True, exist_ok=True)
    (suite_dir / "ledger" / "results_long.csv").write_text(
        "variant,budget_key,metric,value\nA,20/50/4,mrr,0.5\n",
        encoding="utf-8",
    )
    (suite_dir / "ledger" / "runs.jsonl").write_text("{}\n", encoding="utf-8")

    significance_dir = tmp_path / "significance"
    (significance_dir / "tables").mkdir(parents=True, exist_ok=True)
    (significance_dir / "tables" / "table_significance_main.csv").write_text(
        "metric,p_value\nmrr,0.03\n",
        encoding="utf-8",
    )
    (significance_dir / "tables" / "table_confidence_intervals.csv").write_text(
        "metric,ci_low,ci_high\nmrr,0.01,0.08\n",
        encoding="utf-8",
    )
    (significance_dir / "report.md").write_text("# significance\n", encoding="utf-8")

    result_health_dir = tmp_path / "result_health"
    (result_health_dir / "tables").mkdir(parents=True, exist_ok=True)
    (result_health_dir / "figures").mkdir(parents=True, exist_ok=True)
    (result_health_dir / "tables" / "table_result_health.csv").write_text(
        "task,health_status\nmain,ok\n",
        encoding="utf-8",
    )
    (result_health_dir / "figures" / "fig_result_health_breakdown.png").write_bytes(b"PNG")
    (result_health_dir / "report.md").write_text("# result health\n", encoding="utf-8")
    (result_health_dir / "snapshot.json").write_text(
        json.dumps({"gate": {"status": "ok"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    persistent_compare_dir = tmp_path / "persistent_memory_main_compare"
    (persistent_compare_dir / "tables").mkdir(parents=True, exist_ok=True)
    (persistent_compare_dir / "figures").mkdir(parents=True, exist_ok=True)
    (persistent_compare_dir / "tables" / "table_persistent_memory_main_compare.csv").write_text(
        "query_bank_a_id,query_bank_b_id,mean_delta_mrr_strict\n"
        "persistent_object_memory_core_v1,persistent_object_memory_core_v1,0.05\n",
        encoding="utf-8",
    )
    (persistent_compare_dir / "tables" / "table_persistent_memory_main_compare.md").write_text(
        "# compare\n",
        encoding="utf-8",
    )
    (persistent_compare_dir / "tables" / "table_persistent_memory_main_significance.csv").write_text(
        "metric,p_value\nmrr_strict,0.03\n",
        encoding="utf-8",
    )
    (persistent_compare_dir / "tables" / "table_persistent_memory_main_significance.md").write_text(
        "# sig\n",
        encoding="utf-8",
    )
    for stem in (
        "fig_persistent_memory_main_delta",
        "fig_persistent_memory_main_health",
        "fig_persistent_memory_main_query_strength",
    ):
        (persistent_compare_dir / "figures" / f"{stem}.png").write_bytes(b"PNG")
        (persistent_compare_dir / "figures" / f"{stem}.pdf").write_bytes(b"PDF")
    (persistent_compare_dir / "compare_summary.json").write_text(
        json.dumps(
            {
                "persistent_memory_main_status": "improved",
                "query_bank_a_id": "persistent_object_memory_core_v1",
                "query_bank_b_id": "persistent_object_memory_core_v1",
                "alignment_ok": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (persistent_compare_dir / "snapshot.json").write_text(
        json.dumps({"persistent_memory_main_status": "improved"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (persistent_compare_dir / "README.md").write_text("# persistent main compare\n", encoding="utf-8")

    persistent_decision_dir = tmp_path / "persistent_memory_main_decision"
    (persistent_decision_dir / "tables").mkdir(parents=True, exist_ok=True)
    (persistent_decision_dir / "tables" / "table_persistent_memory_main_decision.csv").write_text(
        "promotion_decision,promotion_ready\npromote_persistent_memory_to_mainline,true\n",
        encoding="utf-8",
    )
    (persistent_decision_dir / "tables" / "table_persistent_memory_main_decision.md").write_text(
        "# decision\n",
        encoding="utf-8",
    )
    (persistent_decision_dir / "report.md").write_text("# persistent main decision\n", encoding="utf-8")
    (persistent_decision_dir / "snapshot.json").write_text(
        json.dumps(
            {"promotion_decision_summary": {"promotion_decision": "promote_persistent_memory_to_mainline"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cleanup_dir = tmp_path / "mainline_admission_cleanup"
    (cleanup_dir / "tables").mkdir(parents=True, exist_ok=True)
    (cleanup_dir / "tables" / "table_mainline_admission_cleanup.csv").write_text(
        "mainline_admission_cleanup_status\nexplained\n",
        encoding="utf-8",
    )
    (cleanup_dir / "tables" / "table_mainline_admission_cleanup.md").write_text("# cleanup\n", encoding="utf-8")
    (cleanup_dir / "report.md").write_text("# cleanup report\n", encoding="utf-8")
    (cleanup_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "mainline_admission_cleanup_summary": {
                    "mainline_admission_cleanup_status": "explained",
                    "promotion_vs_admission_consistency_status": "consistent_different_layers",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sample_dir = tmp_path / "sample_contract"
    (sample_dir / "tables").mkdir(parents=True, exist_ok=True)
    (sample_dir / "tables" / "table_sample_contract_summary.csv").write_text(
        "sample_contract_status,large_sample_claim_status\nborderline,supported_with_caveat\n",
        encoding="utf-8",
    )
    (sample_dir / "tables" / "table_sample_contract_summary.md").write_text("# sample\n", encoding="utf-8")
    (sample_dir / "report.md").write_text("# sample contract\n", encoding="utf-8")
    (sample_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "sample_contract_summary": {
                    "sample_contract_status": "borderline",
                    "large_sample_claim_status": "supported_with_caveat",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "paper_ready_min"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_paper_ready.py"),
            "--compare_dir",
            str(compare_dir),
            "--out_dir",
            str(out_dir),
            "--suite-dir",
            str(suite_dir),
            "--significance-dir",
            str(significance_dir),
            "--result-health-dir",
            str(result_health_dir),
            "--persistent-memory-main-compare-dir",
            str(persistent_compare_dir),
            "--persistent-memory-main-decision-dir",
            str(persistent_decision_dir),
            "--mainline-admission-cleanup-dir",
            str(cleanup_dir),
            "--sample-contract-dir",
            str(sample_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_table_panel=" in proc.stdout
    assert (out_dir / "tables" / "table_budget_panel.csv").exists()
    assert (out_dir / "tables" / "table_budget_panel_delta.csv").exists()
    assert (out_dir / "persistent_memory_main_compare" / "compare_summary.json").exists()
    assert (out_dir / "persistent_memory_main_decision" / "report.md").exists()
    assert (out_dir / "mainline_admission_cleanup" / "report.md").exists()
    assert (out_dir / "sample_contract" / "report.md").exists()
    assert (out_dir / "figures" / "fig_persistent_memory_main_delta.png").exists()
    report_text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Persistent Memory Main Compare" in report_text
    assert "## Persistent Memory Main Decision" in report_text
    assert "## Mainline Admission Cleanup" in report_text
    assert "## Sample Contract" in report_text
    assert "## Mainline Reading Order" in report_text
