from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_export_paper_ready_smoke(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    # NLQ stub/real
    _write_csv(
        compare_dir / "nlq_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "nlq_full_hit_at_k_strict": 0.4,
                "safety_count_granularity": "row=(variant,budget,query)",
                "safety_critical_fn_rate": 0.30,
                "safety_reason_budget_insufficient_rate": 0.15,
                "safety_reason_evidence_missing_rate": 0.05,
                "safety_reason_constraints_over_filtered_rate": 0.04,
                "safety_reason_retrieval_distractor_rate": 0.03,
                "safety_reason_other_rate": 0.03,
            },
            {
                "budget_key": "40/100/8",
                "budget_seconds": 40,
                "nlq_full_hit_at_k_strict": 0.6,
                "safety_count_granularity": "row=(variant,budget,query)",
                "safety_critical_fn_rate": 0.20,
                "safety_reason_budget_insufficient_rate": 0.08,
                "safety_reason_evidence_missing_rate": 0.04,
                "safety_reason_constraints_over_filtered_rate": 0.03,
                "safety_reason_retrieval_distractor_rate": 0.03,
                "safety_reason_other_rate": 0.02,
            },
        ],
    )
    _write_csv(
        compare_dir / "nlq_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "nlq_full_hit_at_k_strict": 0.45,
                "safety_count_granularity": "row=(variant,budget,query)",
                "safety_critical_fn_rate": 0.27,
                "safety_reason_budget_insufficient_rate": 0.12,
                "safety_reason_evidence_missing_rate": 0.05,
                "safety_reason_constraints_over_filtered_rate": 0.05,
                "safety_reason_retrieval_distractor_rate": 0.03,
                "safety_reason_other_rate": 0.02,
            },
            {
                "budget_key": "40/100/8",
                "budget_seconds": 40,
                "nlq_full_hit_at_k_strict": 0.67,
                "safety_count_granularity": "row=(variant,budget,query)",
                "safety_critical_fn_rate": 0.15,
                "safety_reason_budget_insufficient_rate": 0.06,
                "safety_reason_evidence_missing_rate": 0.03,
                "safety_reason_constraints_over_filtered_rate": 0.03,
                "safety_reason_retrieval_distractor_rate": 0.02,
                "safety_reason_other_rate": 0.01,
            },
        ],
    )
    # Streaming stub/real
    _write_csv(
        compare_dir / "streaming_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.5, "e2e_ms_p50": 8.0, "e2e_ms_p95": 12.0},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.62, "e2e_ms_p50": 9.0, "e2e_ms_p95": 13.0},
        ],
    )
    _write_csv(
        compare_dir / "streaming_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.55, "e2e_ms_p50": 8.5, "e2e_ms_p95": 12.5},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.7, "e2e_ms_p50": 9.5, "e2e_ms_p95": 13.5},
        ],
    )
    # BYE intentionally missing: should be handled gracefully.
    (compare_dir / "budget_recommend" / "stub").mkdir(parents=True, exist_ok=True)
    (compare_dir / "budget_recommend" / "real").mkdir(parents=True, exist_ok=True)
    (compare_dir / "budget_recommend" / "stub" / "recommend_summary.json").write_text(
        json.dumps({"top1_budget_key": "40/100/8"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "budget_recommend" / "real" / "recommend_summary.json").write_text(
        json.dumps({"top1_budget_key": "40/100/8"}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional streaming policy compare input.
    (compare_dir / "stream_policy_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_policy_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_policy_cmp" / "tables" / "table_streaming_policy_compare.csv").write_text(
        "policy_a,policy_b,strict_success_rate_a,strict_success_rate_b\nsafety_latency,safety_latency_intervention,0.50,0.62\n",
        encoding="utf-8",
    )
    (compare_dir / "stream_policy_cmp" / "tables" / "table_streaming_policy_compare.md").write_text(
        "# compare\n",
        encoding="utf-8",
    )
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_safety_latency.png").write_bytes(b"PNG")
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_safety_latency.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_delta.png").write_bytes(b"PNG")
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_chain_success.png").write_bytes(b"PNG")
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_chain_success.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_chain_delta.png").write_bytes(b"PNG")
    (compare_dir / "stream_policy_cmp" / "figures" / "fig_streaming_policy_compare_chain_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_policy_cmp" / "compare_summary.json").write_text(
        json.dumps({"policy_a": "safety_latency", "policy_b": "safety_latency_intervention"}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional streaming repo compare input.
    (compare_dir / "stream_repo_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_repo_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_repo_cmp" / "tables" / "table_streaming_repo_compare.csv").write_text(
        "policy_a,policy_b,a_use_repo,b_use_repo,strict_hit_at_k_rate_a,strict_hit_at_k_rate_b\nsafety_latency,safety_latency,false,true,0.52,0.63\n",
        encoding="utf-8",
    )
    (compare_dir / "stream_repo_cmp" / "tables" / "table_streaming_repo_compare.md").write_text(
        "# repo compare\n",
        encoding="utf-8",
    )
    (compare_dir / "stream_repo_cmp" / "figures" / "fig_streaming_repo_compare_safety_latency.png").write_bytes(b"PNG")
    (compare_dir / "stream_repo_cmp" / "figures" / "fig_streaming_repo_compare_safety_latency.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_repo_cmp" / "figures" / "fig_streaming_repo_compare_delta.png").write_bytes(b"PNG")
    (compare_dir / "stream_repo_cmp" / "figures" / "fig_streaming_repo_compare_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_repo_cmp" / "compare_summary.json").write_text(
        json.dumps({"policy_a": "safety_latency", "policy_b": "safety_latency", "b_use_repo": True}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional streaming intervention sweep input.
    (compare_dir / "stream_intervention_sweep" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_intervention_sweep" / "best_config.yaml").write_text("name: best\nw_safety: 1.1\n", encoding="utf-8")
    (compare_dir / "stream_intervention_sweep" / "best_report.md").write_text("# best\n", encoding="utf-8")
    (compare_dir / "stream_intervention_sweep" / "snapshot.json").write_text(
        json.dumps(
            {
                "best": {"cfg_name": "best", "cfg_hash": "abc123", "objective": 0.6},
                "default": {"objective": 0.4},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (compare_dir / "stream_intervention_sweep" / "figures" / "fig_objective_vs_latency.png").write_bytes(b"PNG")
    (compare_dir / "stream_intervention_sweep" / "figures" / "fig_objective_vs_latency.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_intervention_sweep" / "figures" / "fig_pareto_frontier.png").write_bytes(b"PNG")
    (compare_dir / "stream_intervention_sweep" / "figures" / "fig_pareto_frontier.pdf").write_bytes(b"PDF")
    # Optional streaming codec sweep input.
    (compare_dir / "stream_codec_sweep" / "aggregate").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_codec_sweep" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_codec_sweep" / "aggregate" / "metrics_by_k.csv").write_text(
        "codec_k,hit_at_k_strict,objective_combo\n4,0.55,0.40\n8,0.61,0.47\n",
        encoding="utf-8",
    )
    (compare_dir / "stream_codec_sweep" / "aggregate" / "metrics_by_k.md").write_text("# codec\n", encoding="utf-8")
    (compare_dir / "stream_codec_sweep" / "snapshot.json").write_text(
        json.dumps({"k_list": [4, 8], "policy": "safety_latency_intervention"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "stream_codec_sweep" / "figures" / "fig_streaming_quality_vs_k.png").write_bytes(b"PNG")
    (compare_dir / "stream_codec_sweep" / "figures" / "fig_streaming_quality_vs_k.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_codec_sweep" / "figures" / "fig_streaming_safety_vs_k.png").write_bytes(b"PNG")
    (compare_dir / "stream_codec_sweep" / "figures" / "fig_streaming_safety_vs_k.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_codec_sweep" / "figures" / "fig_streaming_latency_vs_k.png").write_bytes(b"PNG")
    (compare_dir / "stream_codec_sweep" / "figures" / "fig_streaming_latency_vs_k.pdf").write_bytes(b"PDF")
    # Optional reranker sweep input.
    (compare_dir / "reranker_sweep" / "aggregate").mkdir(parents=True, exist_ok=True)
    (compare_dir / "reranker_sweep" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "reranker_sweep" / "aggregate" / "metrics_by_weights.csv").write_text(
        "weights_id,cfg_name,objective,mrr_strict,top1_in_distractor_rate,critical_fn_rate\n1,default,0.12,0.20,0.30,0.25\n",
        encoding="utf-8",
    )
    (compare_dir / "reranker_sweep" / "aggregate" / "metrics_by_weights.md").write_text("# sweep\n", encoding="utf-8")
    (compare_dir / "reranker_sweep" / "best_weights.yaml").write_text("name: best\nw_trigger: 0.9\n", encoding="utf-8")
    (compare_dir / "reranker_sweep" / "best_report.md").write_text("# best reranker\n", encoding="utf-8")
    (compare_dir / "reranker_sweep" / "snapshot.json").write_text(
        json.dumps({"best": {"cfg_name": "best", "cfg_hash": "def456", "objective": 0.2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "reranker_sweep" / "figures" / "fig_objective_vs_weights_id.png").write_bytes(b"PNG")
    (compare_dir / "reranker_sweep" / "figures" / "fig_objective_vs_weights_id.pdf").write_bytes(b"PDF")
    (compare_dir / "reranker_sweep" / "figures" / "fig_tradeoff_strict_vs_distractor.png").write_bytes(b"PNG")
    (compare_dir / "reranker_sweep" / "figures" / "fig_tradeoff_strict_vs_distractor.pdf").write_bytes(b"PDF")
    # Optional repo policy sweep input.
    (compare_dir / "repo_policy_sweep" / "aggregate").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_policy_sweep" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_policy_sweep" / "aggregate" / "metrics_by_setting.csv").write_text(
        "setting,budget_key,quality_proxy,objective\nfixed_interval|budgeted_topk,20/50/4,0.42,0.31\n",
        encoding="utf-8",
    )
    (compare_dir / "repo_policy_sweep" / "aggregate" / "metrics_by_setting.md").write_text("# repo policy\n", encoding="utf-8")
    (compare_dir / "repo_policy_sweep" / "best_report.md").write_text("# best repo policy\n", encoding="utf-8")
    (compare_dir / "repo_policy_sweep" / "snapshot.json").write_text(
        json.dumps({"best": {"setting": "fixed_interval|budgeted_topk"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "repo_policy_sweep" / "figures" / "fig_repo_quality_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "repo_policy_sweep" / "figures" / "fig_repo_quality_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_policy_sweep" / "figures" / "fig_repo_size_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "repo_policy_sweep" / "figures" / "fig_repo_size_vs_budget_seconds.pdf").write_bytes(b"PDF")
    # Optional repo query selection sweep input.
    (compare_dir / "repo_query_selection_sweep" / "aggregate").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_query_selection_sweep" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_query_selection_sweep" / "aggregate" / "metrics_by_policy_budget.csv").write_text(
        "policy,budget_key,mrr_strict,top1_in_distractor_rate\nbudgeted_topk,20/50/4,0.30,0.40\nquery_aware,20/50/4,0.42,0.28\n",
        encoding="utf-8",
    )
    (compare_dir / "repo_query_selection_sweep" / "aggregate" / "metrics_by_policy_budget.md").write_text("# repo query\n", encoding="utf-8")
    (compare_dir / "repo_query_selection_sweep" / "best_report.md").write_text("# best repo query\n", encoding="utf-8")
    (compare_dir / "repo_query_selection_sweep" / "snapshot.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "best": {"policy": "query_aware", "mrr_strict": 0.42, "top1_in_distractor_rate": 0.28},
                    "baseline_best": {"policy": "budgeted_topk", "mrr_strict": 0.30, "top1_in_distractor_rate": 0.40},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (compare_dir / "repo_query_selection_sweep" / "figures" / "fig_repo_query_selection_quality_vs_budget.png").write_bytes(b"PNG")
    (compare_dir / "repo_query_selection_sweep" / "figures" / "fig_repo_query_selection_quality_vs_budget.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_query_selection_sweep" / "figures" / "fig_repo_query_selection_distractor_vs_budget.png").write_bytes(b"PNG")
    (compare_dir / "repo_query_selection_sweep" / "figures" / "fig_repo_query_selection_distractor_vs_budget.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_query_selection_sweep" / "figures" / "fig_repo_query_selection_chunks_by_level.png").write_bytes(b"PNG")
    (compare_dir / "repo_query_selection_sweep" / "figures" / "fig_repo_query_selection_chunks_by_level.pdf").write_bytes(b"PDF")
    # Optional component attribution compare input.
    (compare_dir / "component_attr_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "component_attr_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "component_attr_cmp" / "tables" / "table_component_attribution.csv").write_text(
        "setting,strict_value,critical_fn_rate,latency_p95_ms\nA,0.50,0.30,10.0\nD,0.62,0.24,11.0\n",
        encoding="utf-8",
    )
    (compare_dir / "component_attr_cmp" / "tables" / "table_component_attribution.md").write_text(
        "# component attribution\n",
        encoding="utf-8",
    )
    (compare_dir / "component_attr_cmp" / "figures" / "fig_component_attribution_delta.png").write_bytes(b"PNG")
    (compare_dir / "component_attr_cmp" / "figures" / "fig_component_attribution_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "component_attr_cmp" / "figures" / "fig_component_attribution_tradeoff.png").write_bytes(b"PNG")
    (compare_dir / "component_attr_cmp" / "figures" / "fig_component_attribution_tradeoff.pdf").write_bytes(b"PDF")
    (compare_dir / "component_attr_cmp" / "compare_summary.json").write_text(
        json.dumps({"summary": {"delta_D_vs_A": 0.12}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "component_attr_cmp" / "snapshot.json").write_text(
        json.dumps({"inputs": {"selected_uids": 2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional BYE report compare input.
    (compare_dir / "bye_report_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "bye_report_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "bye_report_cmp" / "tables" / "table_bye_report_compare.csv").write_text(
        "uid,status_a,status_b,bye_primary_score_a,bye_primary_score_b,delta_bye_primary_score\nu001,ok,ok,0.7,0.8,0.1\n",
        encoding="utf-8",
    )
    (compare_dir / "bye_report_cmp" / "tables" / "table_bye_report_compare.md").write_text(
        "# bye report compare\n",
        encoding="utf-8",
    )
    (compare_dir / "bye_report_cmp" / "figures" / "fig_bye_critical_fn_delta.png").write_bytes(b"PNG")
    (compare_dir / "bye_report_cmp" / "figures" / "fig_bye_critical_fn_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "bye_report_cmp" / "figures" / "fig_bye_latency_delta.png").write_bytes(b"PNG")
    (compare_dir / "bye_report_cmp" / "figures" / "fig_bye_latency_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "bye_report_cmp" / "compare_summary.json").write_text(
        json.dumps({"uids_total": 1, "labels": {"a": "stub", "b": "real"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional chain NLQ panel input.
    (compare_dir / "chain_nlq").mkdir(parents=True, exist_ok=True)
    (compare_dir / "chain_nlq" / "table_chain_summary.csv").write_text(
        "budget_key,budget_seconds,chain_hit_at_k_strict,chain_mrr,chain_success_rate\n20/50/4,20,0.5,0.45,0.5\n",
        encoding="utf-8",
    )
    (compare_dir / "chain_nlq" / "table_chain_summary.md").write_text("# chain\n", encoding="utf-8")
    (compare_dir / "chain_nlq" / "table_chain_failure_attribution.csv").write_text(
        "variant,chain_derive,budget_max_total_s,chain_fail_step1_no_hit_rate\nfull,time+place,20,0.1\n",
        encoding="utf-8",
    )
    (compare_dir / "chain_nlq" / "table_chain_failure_attribution.md").write_text("# chain failure\n", encoding="utf-8")
    (compare_dir / "chain_nlq" / "fig_chain_success_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "chain_nlq" / "fig_chain_success_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_nlq" / "fig_chain_failure_attribution_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "chain_nlq" / "fig_chain_failure_attribution_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_nlq" / "fig_chain_success_vs_derive.png").write_bytes(b"PNG")
    (compare_dir / "chain_nlq" / "fig_chain_success_vs_derive.pdf").write_bytes(b"PDF")
    # Optional chain repo compare input.
    (compare_dir / "chain_repo_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "chain_repo_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "chain_repo_cmp" / "tables" / "table_chain_repo_compare.csv").write_text(
        "uid,status_a,status_b,budget_seconds,chain_success_rate_a,chain_success_rate_b,delta_chain_success_rate\nu001,ok,ok,20,0.20,0.35,0.15\n",
        encoding="utf-8",
    )
    (compare_dir / "chain_repo_cmp" / "tables" / "table_chain_repo_compare.md").write_text(
        "# chain repo compare\n",
        encoding="utf-8",
    )
    (compare_dir / "chain_repo_cmp" / "figures" / "fig_chain_repo_compare_success_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "chain_repo_cmp" / "figures" / "fig_chain_repo_compare_success_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_repo_cmp" / "figures" / "fig_chain_repo_compare_delta.png").write_bytes(b"PNG")
    (compare_dir / "chain_repo_cmp" / "figures" / "fig_chain_repo_compare_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_repo_cmp" / "compare_summary.json").write_text(
        json.dumps({"uids_total": 1, "budgets_matched": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "chain_repo_cmp" / "snapshot.json").write_text(
        json.dumps({"selection": {"uids_found": 1}}, ensure_ascii=False),
        encoding="utf-8",
    )

    out_dir = tmp_path / "paper_ready"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_paper_ready.py"),
        "--compare_dir",
        str(compare_dir),
        "--out_dir",
        str(out_dir),
        "--format",
        "md+csv",
        "--streaming-policy-compare-dir",
        str(compare_dir / "stream_policy_cmp"),
        "--streaming-repo-compare-dir",
        str(compare_dir / "stream_repo_cmp"),
        "--streaming-intervention-sweep-dir",
        str(compare_dir / "stream_intervention_sweep"),
        "--streaming-codec-sweep-dir",
        str(compare_dir / "stream_codec_sweep"),
        "--reranker-sweep-dir",
        str(compare_dir / "reranker_sweep"),
        "--repo-policy-sweep-dir",
        str(compare_dir / "repo_policy_sweep"),
        "--repo-query-selection-sweep-dir",
        str(compare_dir / "repo_query_selection_sweep"),
        "--component-attribution-dir",
        str(compare_dir / "component_attr_cmp"),
        "--bye-report-compare-dir",
        str(compare_dir / "bye_report_cmp"),
        "--chain-nlq-dir",
        str(compare_dir / "chain_nlq"),
        "--chain-repo-compare-dir",
        str(compare_dir / "chain_repo_cmp"),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_table_panel=" in proc.stdout
    assert "saved_table_delta=" in proc.stdout

    panel_csv = out_dir / "tables" / "table_budget_panel.csv"
    panel_md = out_dir / "tables" / "table_budget_panel.md"
    delta_csv = out_dir / "tables" / "table_budget_panel_delta.csv"
    delta_md = out_dir / "tables" / "table_budget_panel_delta.md"
    report_md = out_dir / "report.md"
    snapshot_json = out_dir / "snapshot.json"
    assert panel_csv.exists()
    assert panel_md.exists()
    assert delta_csv.exists()
    assert delta_md.exists()
    assert report_md.exists()
    assert snapshot_json.exists()
    assert (out_dir / "figures" / "fig_budget_primary_vs_seconds_panel.png").exists()
    assert (out_dir / "figures" / "fig_budget_primary_delta_vs_seconds_panel.png").exists()
    assert (out_dir / "figures" / "fig_budget_latency_vs_seconds_streaming.png").exists()
    assert (out_dir / "figures" / "fig_nlq_critical_fn_rate_vs_seconds.png").exists()
    assert (out_dir / "figures" / "fig_nlq_failure_attribution_vs_seconds.png").exists()
    assert (out_dir / "tables" / "table_streaming_policy_compare.csv").exists()
    assert (out_dir / "tables" / "table_streaming_policy_compare.md").exists()
    assert (out_dir / "figures" / "fig_streaming_policy_compare_safety_latency.png").exists()
    assert (out_dir / "figures" / "fig_streaming_policy_compare_delta.png").exists()
    assert (out_dir / "figures" / "fig_streaming_policy_compare_chain_success.png").exists()
    assert (out_dir / "figures" / "fig_streaming_policy_compare_chain_delta.png").exists()
    assert (out_dir / "tables" / "table_streaming_repo_compare.csv").exists()
    assert (out_dir / "tables" / "table_streaming_repo_compare.md").exists()
    assert (out_dir / "figures" / "fig_streaming_repo_compare_safety_latency.png").exists()
    assert (out_dir / "figures" / "fig_streaming_repo_compare_delta.png").exists()
    assert (out_dir / "streaming_intervention_sweep" / "best_config.yaml").exists()
    assert (out_dir / "streaming_intervention_sweep" / "best_report.md").exists()
    assert (out_dir / "figures" / "fig_objective_vs_latency.png").exists()
    assert (out_dir / "figures" / "fig_pareto_frontier.png").exists()
    assert (out_dir / "streaming_codec_sweep" / "metrics_by_k.csv").exists()
    assert (out_dir / "figures" / "fig_streaming_quality_vs_k.png").exists()
    assert (out_dir / "figures" / "fig_streaming_safety_vs_k.png").exists()
    assert (out_dir / "figures" / "fig_streaming_latency_vs_k.png").exists()
    assert (out_dir / "reranker_sweep" / "metrics_by_weights.csv").exists()
    assert (out_dir / "reranker_sweep" / "best_weights.yaml").exists()
    assert (out_dir / "figures" / "fig_objective_vs_weights_id.png").exists()
    assert (out_dir / "figures" / "fig_tradeoff_strict_vs_distractor.png").exists()
    assert (out_dir / "repo_policy" / "metrics_by_setting.csv").exists()
    assert (out_dir / "repo_policy" / "best_report.md").exists()
    assert (out_dir / "figures" / "fig_repo_quality_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_size_vs_budget_seconds.png").exists()
    assert (out_dir / "repo_query_selection" / "metrics_by_policy_budget.csv").exists()
    assert (out_dir / "repo_query_selection" / "best_report.md").exists()
    assert (out_dir / "figures" / "fig_repo_query_selection_quality_vs_budget.png").exists()
    assert (out_dir / "figures" / "fig_repo_query_selection_distractor_vs_budget.png").exists()
    assert (out_dir / "figures" / "fig_repo_query_selection_chunks_by_level.png").exists()
    assert (out_dir / "component_attribution" / "table_component_attribution.csv").exists()
    assert (out_dir / "component_attribution" / "table_component_attribution.md").exists()
    assert (out_dir / "figures" / "fig_component_attribution_delta.png").exists()
    assert (out_dir / "figures" / "fig_component_attribution_tradeoff.png").exists()
    assert (out_dir / "bye_report" / "table_bye_report_compare.csv").exists()
    assert (out_dir / "bye_report" / "table_bye_report_compare.md").exists()
    assert (out_dir / "figures" / "fig_bye_critical_fn_delta.png").exists()
    assert (out_dir / "figures" / "fig_bye_latency_delta.png").exists()
    assert (out_dir / "chain_nlq_panel" / "table_chain_summary.csv").exists()
    assert (out_dir / "chain_nlq_panel" / "table_chain_summary.md").exists()
    assert (out_dir / "chain_nlq_panel" / "table_chain_failure_attribution.csv").exists()
    assert (out_dir / "chain_nlq_panel" / "table_chain_failure_attribution.md").exists()
    assert (out_dir / "figures" / "fig_chain_success_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_chain_failure_attribution_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_chain_success_vs_derive.png").exists()
    assert (out_dir / "chain_repo_compare" / "table_chain_repo_compare.csv").exists()
    assert (out_dir / "chain_repo_compare" / "table_chain_repo_compare.md").exists()
    assert (out_dir / "figures" / "fig_chain_repo_compare_success_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_chain_repo_compare_delta.png").exists()

    header = panel_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "task" in header
    assert "budget_seconds" in header
    assert "primary_a" in header
    assert "primary_b" in header
    assert "delta_primary" in header
    assert "safety_critical_fn_rate_a" in header
