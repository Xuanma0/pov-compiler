from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_benchmark_suite_runner_smoke(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    _write_csv(
        compare_dir / "nlq_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "nlq_full_hit_at_k_strict": 0.40,
                "safety_reason_budget_insufficient_rate": 0.12,
                "safety_reason_evidence_missing_rate": 0.05,
                "safety_reason_constraints_over_filtered_rate": 0.03,
                "safety_reason_retrieval_distractor_rate": 0.02,
                "safety_reason_other_rate": 0.01,
            },
            {
                "budget_key": "40/100/8",
                "budget_seconds": 40,
                "nlq_full_hit_at_k_strict": 0.58,
                "safety_reason_budget_insufficient_rate": 0.08,
                "safety_reason_evidence_missing_rate": 0.04,
                "safety_reason_constraints_over_filtered_rate": 0.02,
                "safety_reason_retrieval_distractor_rate": 0.02,
                "safety_reason_other_rate": 0.01,
            },
        ],
    )
    _write_csv(
        compare_dir / "nlq_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "nlq_full_hit_at_k_strict": 0.46,
                "safety_reason_budget_insufficient_rate": 0.10,
                "safety_reason_evidence_missing_rate": 0.04,
                "safety_reason_constraints_over_filtered_rate": 0.03,
                "safety_reason_retrieval_distractor_rate": 0.02,
                "safety_reason_other_rate": 0.01,
            },
            {
                "budget_key": "40/100/8",
                "budget_seconds": 40,
                "nlq_full_hit_at_k_strict": 0.64,
                "safety_reason_budget_insufficient_rate": 0.06,
                "safety_reason_evidence_missing_rate": 0.03,
                "safety_reason_constraints_over_filtered_rate": 0.02,
                "safety_reason_retrieval_distractor_rate": 0.01,
                "safety_reason_other_rate": 0.01,
            },
        ],
    )
    _write_csv(
        compare_dir / "streaming_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.50, "e2e_ms_p50": 8.0},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.61, "e2e_ms_p50": 9.0},
        ],
    )
    _write_csv(
        compare_dir / "streaming_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.57, "e2e_ms_p50": 8.6},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.69, "e2e_ms_p50": 9.4},
        ],
    )
    _write_csv(
        compare_dir / "bye_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "qualityScore": 0.48},
            {"budget_key": "40/100/8", "budget_seconds": 40, "qualityScore": 0.55},
        ],
    )
    _write_csv(
        compare_dir / "bye_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "qualityScore": 0.53},
            {"budget_key": "40/100/8", "budget_seconds": 40, "qualityScore": 0.62},
        ],
    )
    (compare_dir / "signal_selection").mkdir(parents=True, exist_ok=True)
    (compare_dir / "signal_selection" / "selected_uids.txt").write_text("u1\nu2\n", encoding="utf-8")
    (compare_dir / "signal_selection" / "coverage.csv").write_text(
        "uid,coverage_score,missing_place,missing_interaction,missing_lost_object\nu1,3.0,0,0,0\nu2,2.0,0,1,0\n",
        encoding="utf-8",
    )
    (compare_dir / "signal_selection" / "snapshot.json").write_text('{"selection_mode":"auto_signal_cache"}', encoding="utf-8")

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "suite_id: suite_smoke",
                'suite_version: "1.42"',
                "seed: 5",
                "health_gate_profile: main_real",
                "paper_map: configs/paper/main_result_map_v1.yaml",
                f"output_root: {tmp_path / 'main_result_root'}",
                "selection:",
                f"  compare_dir: {compare_dir}",
                "  tasks: [nlq, streaming, bye]",
                "  labels:",
                "    a: stub",
                "    b: real",
                "  mode: auto_signal_cache",
                "  signal_min_score: 1.5",
                "  top_k_uids: 2",
                "  signal_selection_dir: signal_selection",
                "budgets:",
                "  points:",
                "    - key: 20/50/4",
                "    - key: 40/100/8",
                "queries:",
                "  query_bank: configs/queries/core_real_v1.yaml",
                "  auxiliary_banks:",
                "    - configs/queries/core_chain_v1.yaml",
                "    - configs/queries/core_lost_object_v1.yaml",
                "  groups: [decision, chain, lost_object, repo_summary]",
                "  top_k: 6",
                "metrics:",
                "  primary:",
                "    nlq: nlq_full_hit_at_k_strict",
                '    streaming: "hit@k_strict"',
                "    bye: qualityScore",
                "  failure:",
                "    nlq:",
                "      - safety_reason_budget_insufficient_rate",
                "      - safety_reason_evidence_missing_rate",
                "      - safety_reason_constraints_over_filtered_rate",
                "      - safety_reason_retrieval_distractor_rate",
                "      - safety_reason_other_rate",
                "prompts:",
                "  registry: configs/prompts/registry_v1.yaml",
                "  profile: v1.42_main",
                "  lock_required: true",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "suite_out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_benchmark_suite.py"),
        "--manifest",
        str(manifest_path),
        "--out_dir",
        str(out_dir),
        "--mode",
        "collect-only",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_results_long=" in proc.stdout
    assert "saved_query_bank_lock=" in proc.stdout

    assert (out_dir / "manifest" / "experiment_manifest.yaml").exists()
    assert (out_dir / "manifest" / "manifest_resolved.json").exists()
    assert (out_dir / "manifest" / "prompt_lock.json").exists()
    assert (out_dir / "manifest" / "query_bank_lock.json").exists()
    assert (out_dir / "manifest" / "query_banks" / "core_real_v1.yaml").exists()
    assert (out_dir / "ledger" / "results_long.csv").exists()
    assert (out_dir / "ledger" / "runs.jsonl").exists()
    assert (out_dir / "compare" / "tables" / "table_main_results.csv").exists()
    assert (out_dir / "compare" / "tables" / "table_main_results.md").exists()
    assert (out_dir / "compare" / "tables" / "table_significance.csv").exists()
    assert (out_dir / "compare" / "tables" / "table_significance.md").exists()
    assert (out_dir / "compare" / "tables" / "table_failure_attribution.csv").exists()
    assert (out_dir / "compare" / "tables" / "table_failure_attribution.md").exists()
    assert (out_dir / "compare" / "figures" / "fig_main_budget_frontier.png").exists()
    assert (out_dir / "compare" / "figures" / "fig_main_budget_frontier.pdf").exists()
    assert (out_dir / "compare" / "figures" / "fig_main_failure_attribution.png").exists()
    assert (out_dir / "compare" / "figures" / "fig_main_variant_delta.png").exists()
    assert (out_dir / "compare" / "compare_summary.json").exists()
    assert (out_dir / "compare" / "snapshot.json").exists()
    assert (out_dir / "compare" / "commands.sh").exists()
    assert (out_dir / "compare" / "README.md").exists()
    assert (out_dir / "significance" / "tables" / "table_significance_main.csv").exists()
    assert "insufficient_pairs" in (out_dir / "compare" / "tables" / "table_significance.csv").read_text(encoding="utf-8")
    compare_summary = (out_dir / "compare" / "compare_summary.json").read_text(encoding="utf-8")
    assert "query_bank_id" in compare_summary
    assert "query_bank_hash" in compare_summary
    assert "selection_mode" in compare_summary
    assert "manifest_hash" in compare_summary
    assert "paper_map" in compare_summary
    assert "health_gate_profile" in compare_summary
    compare_snapshot = (out_dir / "compare" / "snapshot.json").read_text(encoding="utf-8")
    assert "query_bank_hash" in compare_snapshot
    assert "manifest_hash" in compare_snapshot
