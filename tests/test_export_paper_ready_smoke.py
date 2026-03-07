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
    # Optional repo summary sweep input.
    (compare_dir / "repo_summary_sweep" / "aggregate").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_summary_sweep" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_summary_sweep" / "aggregate" / "metrics_by_policy_budget.csv").write_text(
        "policy,budget_key,repo_quality_proxy\nbaseline,20/50/4,0.3\nsummary_v0,20/50/4,0.5\n",
        encoding="utf-8",
    )
    (compare_dir / "repo_summary_sweep" / "aggregate" / "metrics_by_policy_budget.md").write_text("# repo summary\n", encoding="utf-8")
    (compare_dir / "repo_summary_sweep" / "snapshot.json").write_text(
        json.dumps({"outputs": {"rows": 2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "repo_summary_sweep" / "figures" / "fig_repo_summary_quality_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "repo_summary_sweep" / "figures" / "fig_repo_summary_quality_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_summary_sweep" / "figures" / "fig_repo_summary_size_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "repo_summary_sweep" / "figures" / "fig_repo_summary_size_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_summary_sweep" / "figures" / "fig_repo_summary_delta_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "repo_summary_sweep" / "figures" / "fig_repo_summary_delta_vs_budget_seconds.pdf").write_bytes(b"PDF")
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
    # Optional decisions backend compare input.
    (compare_dir / "decisions_backend_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "decisions_backend_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "decisions_backend_cmp" / "tables" / "table_decisions_backend_compare.csv").write_text(
        "uid,status_stub,status_real,delta_mrr_strict\nu001,ok,ok,0.12\n",
        encoding="utf-8",
    )
    (compare_dir / "decisions_backend_cmp" / "tables" / "table_decisions_backend_compare.md").write_text(
        "# decisions backend compare\n",
        encoding="utf-8",
    )
    (compare_dir / "decisions_backend_cmp" / "figures" / "fig_decisions_backend_delta.png").write_bytes(b"PNG")
    (compare_dir / "decisions_backend_cmp" / "figures" / "fig_decisions_backend_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "decisions_backend_cmp" / "figures" / "fig_decisions_backend_tradeoff.png").write_bytes(b"PNG")
    (compare_dir / "decisions_backend_cmp" / "figures" / "fig_decisions_backend_tradeoff.pdf").write_bytes(b"PDF")
    (compare_dir / "decisions_backend_cmp" / "compare_summary.json").write_text(
        json.dumps({"uids_total": 1, "delta_stats": {"mrr_strict": {"mean": 0.12}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional planner backend compare input.
    (compare_dir / "planner_backend_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "planner_backend_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "planner_backend_cmp" / "tables" / "table_planner_backend_compare.csv").write_text(
        "uid,budget_key,planner_a,planner_b,delta_mrr_strict\nu001,20/50/4,heuristic,model,0.08\n",
        encoding="utf-8",
    )
    (compare_dir / "planner_backend_cmp" / "tables" / "table_planner_backend_compare.md").write_text(
        "# planner backend compare\n",
        encoding="utf-8",
    )
    (compare_dir / "planner_backend_cmp" / "figures" / "fig_planner_backend_delta.png").write_bytes(b"PNG")
    (compare_dir / "planner_backend_cmp" / "figures" / "fig_planner_backend_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "planner_backend_cmp" / "figures" / "fig_planner_backend_tradeoff.png").write_bytes(b"PNG")
    (compare_dir / "planner_backend_cmp" / "figures" / "fig_planner_backend_tradeoff.pdf").write_bytes(b"PDF")
    (compare_dir / "planner_backend_cmp" / "compare_summary.json").write_text(
        json.dumps({"uids_total": 1, "planner_a": "heuristic", "planner_b": "model"}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional model stack compare input.
    (compare_dir / "model_stack_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "model_stack_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "model_stack_cmp" / "tables" / "table_model_stack_compare.csv").write_text(
        "budget_key,budget_seconds,variant_code,mrr_strict,delta_mrr_vs_A\n20/50/4,20,A,0.2,0.0\n20/50/4,20,B,0.3,0.1\n",
        encoding="utf-8",
    )
    (compare_dir / "model_stack_cmp" / "tables" / "table_model_stack_compare.md").write_text(
        "# model stack compare\n",
        encoding="utf-8",
    )
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_stack_delta.png").write_bytes(b"PNG")
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_stack_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_stack_tradeoff.png").write_bytes(b"PNG")
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_stack_tradeoff.pdf").write_bytes(b"PDF")
    (compare_dir / "model_stack_cmp" / "tables" / "table_model_cost_compare.csv").write_text(
        "budget_key,budget_seconds,variant_code,model_cost_usd_total,structured_parse_fail_rate\n20/50/4,20,B,0.001,0.0\n",
        encoding="utf-8",
    )
    (compare_dir / "model_stack_cmp" / "tables" / "table_model_cost_compare.md").write_text(
        "# model cost compare\n",
        encoding="utf-8",
    )
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_cost_vs_quality.png").write_bytes(b"PNG")
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_cost_vs_quality.pdf").write_bytes(b"PDF")
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_parse_fail_rate.png").write_bytes(b"PNG")
    (compare_dir / "model_stack_cmp" / "figures" / "fig_model_parse_fail_rate.pdf").write_bytes(b"PDF")
    (compare_dir / "model_stack_cmp" / "compare_summary.json").write_text(
        json.dumps({"uids_total": 1, "variants": ["A", "B", "C", "D"]}, ensure_ascii=False),
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
    # Optional chain attribution panel input.
    (compare_dir / "chain_attr" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "chain_attr" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "chain_attr" / "tables" / "table_chain_attribution.csv").write_text(
        "budget_key,budget_seconds,variant_code,chain_success_rate,delta_success\n20/50/4,20,A,0.3,0.0\n20/50/4,20,B,0.4,0.1\n",
        encoding="utf-8",
    )
    (compare_dir / "chain_attr" / "tables" / "table_chain_attribution.md").write_text("# chain attribution\n", encoding="utf-8")
    (compare_dir / "chain_attr" / "tables" / "table_chain_failure_breakdown.csv").write_text(
        "budget_key,variant_code,chain_fail_constraints_over_filtered_rate\n20/50/4,A,0.2\n",
        encoding="utf-8",
    )
    (compare_dir / "chain_attr" / "tables" / "table_chain_failure_breakdown.md").write_text("# chain failure\n", encoding="utf-8")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_success_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_success_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_delta_success_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_delta_success_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_failure_attribution_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_failure_attribution_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_tradeoff.png").write_bytes(b"PNG")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_tradeoff.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_backoff_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "chain_attr" / "figures" / "fig_chain_attribution_backoff_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "chain_attr" / "compare_summary.json").write_text(
        json.dumps({"budgets_matched": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "chain_attr" / "snapshot.json").write_text(
        json.dumps({"selection": {"uids_found": 1}}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional repo summary retrieval compare input.
    (compare_dir / "repo_summary_retrieval_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_summary_retrieval_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "repo_summary_retrieval_cmp" / "tables" / "table_repo_summary_retrieval_compare.csv").write_text(
        "uid,budget_key,budget_seconds,mrr_strict_a,mrr_strict_b,delta_mrr_strict\nu001,20/50/4,20,0.20,0.35,0.15\n",
        encoding="utf-8",
    )
    (compare_dir / "repo_summary_retrieval_cmp" / "tables" / "table_repo_summary_retrieval_compare.md").write_text(
        "# repo summary retrieval compare\n",
        encoding="utf-8",
    )
    (compare_dir / "repo_summary_retrieval_cmp" / "figures" / "fig_repo_summary_retrieval_quality_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "repo_summary_retrieval_cmp" / "figures" / "fig_repo_summary_retrieval_quality_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_summary_retrieval_cmp" / "figures" / "fig_repo_summary_retrieval_delta.png").write_bytes(b"PNG")
    (compare_dir / "repo_summary_retrieval_cmp" / "figures" / "fig_repo_summary_retrieval_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_summary_retrieval_cmp" / "figures" / "fig_repo_summary_retrieval_candidate_scale.png").write_bytes(b"PNG")
    (compare_dir / "repo_summary_retrieval_cmp" / "figures" / "fig_repo_summary_retrieval_candidate_scale.pdf").write_bytes(b"PDF")
    (compare_dir / "repo_summary_retrieval_cmp" / "compare_summary.json").write_text(
        json.dumps({"uids_total": 1, "budgets_matched": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "repo_summary_retrieval_cmp" / "snapshot.json").write_text(
        json.dumps({"selection": {"uids_found": 1}}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional streaming chain backoff compare input.
    (compare_dir / "stream_chain_backoff_cmp" / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_chain_backoff_cmp" / "figures").mkdir(parents=True, exist_ok=True)
    (compare_dir / "stream_chain_backoff_cmp" / "tables" / "table_streaming_chain_backoff_compare.csv").write_text(
        "strategy,budget_seconds,chain_success_rate,chain_backoff_mean_level\nstrict,20,0.2,0.0\nladder,20,0.4,1.0\n",
        encoding="utf-8",
    )
    (compare_dir / "stream_chain_backoff_cmp" / "tables" / "table_streaming_chain_backoff_compare.md").write_text(
        "# chain backoff compare\n",
        encoding="utf-8",
    )
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_success_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_success_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_latency_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_latency_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds.png").write_bytes(b"PNG")
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_delta.png").write_bytes(b"PNG")
    (compare_dir / "stream_chain_backoff_cmp" / "figures" / "fig_streaming_chain_backoff_delta.pdf").write_bytes(b"PDF")
    (compare_dir / "stream_chain_backoff_cmp" / "compare_summary.json").write_text(
        json.dumps({"policies": ["strict", "ladder", "adaptive"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    # Optional signal selection input.
    (compare_dir / "signal_selection").mkdir(parents=True, exist_ok=True)
    (compare_dir / "signal_selection" / "coverage.md").write_text("# coverage\n", encoding="utf-8")
    (compare_dir / "signal_selection" / "selection_report.md").write_text("# selection\n", encoding="utf-8")
    (compare_dir / "signal_selection" / "selected_uids.txt").write_text("u1\nu2\n", encoding="utf-8")
    (compare_dir / "signal_selection" / "coverage.csv").write_text("uid,coverage_score\nu1,3\n", encoding="utf-8")
    (compare_dir / "signal_selection" / "snapshot.json").write_text(json.dumps({"rows": 1}), encoding="utf-8")

    significance_dir = tmp_path / "significance"
    (significance_dir / "tables").mkdir(parents=True, exist_ok=True)
    (significance_dir / "figures").mkdir(parents=True, exist_ok=True)
    (significance_dir / "tables" / "table_significance_main.csv").write_text(
        "task,budget_key,budget_seconds,mean_delta,status\nnlq,20/50/4,20,0.10,ok\n",
        encoding="utf-8",
    )
    (significance_dir / "tables" / "table_significance_main.md").write_text("# significance\n", encoding="utf-8")
    (significance_dir / "tables" / "table_confidence_intervals.csv").write_text(
        "task,budget_key,ci_low,ci_high,status\nnlq,20/50/4,0.02,0.18,ok\n",
        encoding="utf-8",
    )
    (significance_dir / "tables" / "table_confidence_intervals.md").write_text("# ci\n", encoding="utf-8")
    (significance_dir / "figures" / "fig_significance_delta_vs_budget_seconds.png").write_bytes(b"PNG")
    (significance_dir / "figures" / "fig_significance_delta_vs_budget_seconds.pdf").write_bytes(b"PDF")
    (significance_dir / "figures" / "fig_effect_size_forest.png").write_bytes(b"PNG")
    (significance_dir / "figures" / "fig_effect_size_forest.pdf").write_bytes(b"PDF")
    (significance_dir / "report.md").write_text("# significance report\n", encoding="utf-8")
    (significance_dir / "snapshot.json").write_text(json.dumps({"rows": 1}), encoding="utf-8")

    suite_dir = tmp_path / "suite"
    (suite_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (suite_dir / "ledger").mkdir(parents=True, exist_ok=True)
    (suite_dir / "compare").mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest" / "experiment_manifest.yaml").write_text("suite_id: v1.42_main\n", encoding="utf-8")
    (suite_dir / "manifest" / "manifest_resolved.json").write_text(json.dumps({"suite_id": "v1.42_main"}), encoding="utf-8")
    (suite_dir / "manifest" / "prompt_lock.json").write_text(
        json.dumps({"profile": "v1.42_main", "prompts": [{"task": "decisions", "hash": "abc"}]}),
        encoding="utf-8",
    )
    (suite_dir / "manifest" / "query_banks").mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest" / "query_bank_lock.json").write_text(
        json.dumps(
            {
                "primary": {
                    "query_bank_id": "core_real_v1",
                    "query_bank_version": "1",
                    "query_bank_hash": "hash123",
                }
            }
        ),
        encoding="utf-8",
    )
    (suite_dir / "manifest" / "query_banks" / "core_real_v1.yaml").write_text("query_bank_id: core_real_v1\n", encoding="utf-8")
    (suite_dir / "ledger" / "results_long.csv").write_text("task,budget_key\nnlq,20/50/4\n", encoding="utf-8")
    (suite_dir / "ledger" / "runs.jsonl").write_text('{"task":"nlq","status":"ok"}\n', encoding="utf-8")
    (suite_dir / "compare" / "commands.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (suite_dir / "compare" / "compare_summary.json").write_text(json.dumps({"suite_id": "v1.42_main"}), encoding="utf-8")
    (suite_dir / "compare" / "snapshot.json").write_text(json.dumps({"suite_id": "v1.42_main"}), encoding="utf-8")
    (suite_dir / "compare" / "README.md").write_text("# suite compare\n", encoding="utf-8")
    (suite_dir / "compare" / "tables").mkdir(parents=True, exist_ok=True)
    (suite_dir / "compare" / "figures").mkdir(parents=True, exist_ok=True)
    (suite_dir / "compare" / "tables" / "table_main_results.csv").write_text("task,budget_key\nnlq,20/50/4\n", encoding="utf-8")
    (suite_dir / "compare" / "tables" / "table_main_results.md").write_text("# main results\n", encoding="utf-8")
    (suite_dir / "compare" / "figures" / "fig_main_budget_frontier.png").write_bytes(b"PNG")
    (suite_dir / "admission_control").mkdir(parents=True, exist_ok=True)
    (suite_dir / "admission_control" / "report.md").write_text("# admission report\n", encoding="utf-8")
    (suite_dir / "admission_control" / "snapshot.json").write_text(
        json.dumps(
            {
                "admission_status": "partial",
                "admission_fail_reasons": ["selected_uids_count<3 (2)"],
                "admission_metrics": {"selected_uids_count": 2},
            }
        ),
        encoding="utf-8",
    )

    result_health_dir = tmp_path / "result_health"
    (result_health_dir / "tables").mkdir(parents=True, exist_ok=True)
    (result_health_dir / "figures").mkdir(parents=True, exist_ok=True)
    (result_health_dir / "tables" / "table_result_health.csv").write_text(
        "task,missing_metric_rate,no_data_reason_breakdown\nnlq,0.0,\"{\"\"ok\"\":1}\"\n",
        encoding="utf-8",
    )
    (result_health_dir / "tables" / "table_result_health.md").write_text("# health\n", encoding="utf-8")
    (result_health_dir / "figures" / "fig_result_health_breakdown.png").write_bytes(b"PNG")
    (result_health_dir / "figures" / "fig_result_health_breakdown.pdf").write_bytes(b"PDF")
    (result_health_dir / "snapshot.json").write_text(
        json.dumps({"overall_no_data_reason_counts": {"ok": 1}, "rows_total": 1}),
        encoding="utf-8",
    )

    result_diagnosis_dir = tmp_path / "result_diagnosis"
    (result_diagnosis_dir / "tables").mkdir(parents=True, exist_ok=True)
    (result_diagnosis_dir / "figures").mkdir(parents=True, exist_ok=True)
    (result_diagnosis_dir / "tables" / "table_result_diagnosis.csv").write_text(
        "task,near_zero_delta_rate,diagnosis_recommendations\nnlq,0.0,\"[\"\"keep_current_query_bank\"\"]\"\n",
        encoding="utf-8",
    )
    (result_diagnosis_dir / "tables" / "table_result_diagnosis.md").write_text("# diagnosis\n", encoding="utf-8")
    (result_diagnosis_dir / "figures" / "fig_result_diagnosis_breakdown.png").write_bytes(b"PNG")
    (result_diagnosis_dir / "figures" / "fig_result_diagnosis_breakdown.pdf").write_bytes(b"PDF")
    (result_diagnosis_dir / "report.md").write_text("# diagnosis report\n", encoding="utf-8")
    (result_diagnosis_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "provider_noise_summary": {"availability": "unavailable"},
                "diagnosis_recommendations": ["inspect diagnosis"],
            }
        ),
        encoding="utf-8",
    )

    delta_audit_dir = tmp_path / "delta_audit"
    (delta_audit_dir / "tables").mkdir(parents=True, exist_ok=True)
    (delta_audit_dir / "figures").mkdir(parents=True, exist_ok=True)
    (delta_audit_dir / "tables" / "table_delta_audit.csv").write_text(
        "variant,task,metric,budget,delta_value,recommended_action\nstub->real,nlq,nlq_full_hit_at_k_strict,20/50/4,0.0,strengthen_query_bank\n",
        encoding="utf-8",
    )
    (delta_audit_dir / "tables" / "table_delta_audit.md").write_text("# delta audit\n", encoding="utf-8")
    (delta_audit_dir / "figures" / "fig_delta_audit_breakdown.png").write_bytes(b"PNG")
    (delta_audit_dir / "figures" / "fig_delta_audit_breakdown.pdf").write_bytes(b"PDF")
    (delta_audit_dir / "report.md").write_text("# delta audit report\n", encoding="utf-8")
    (delta_audit_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "main_recommendation": "strengthen_query_bank",
                "recommended_action_counts": {"strengthen_query_bank": 1},
            }
        ),
        encoding="utf-8",
    )

    admission_calibration_dir = tmp_path / "admission_calibration"
    (admission_calibration_dir / "tables").mkdir(parents=True, exist_ok=True)
    (admission_calibration_dir / "tables" / "table_admission_calibration.csv").write_text(
        "admission_profile,calibration_status,recommended_min_selected_uids\nmain_real,partial,3\n",
        encoding="utf-8",
    )
    (admission_calibration_dir / "tables" / "table_admission_calibration.md").write_text(
        "# admission calibration\n",
        encoding="utf-8",
    )
    (admission_calibration_dir / "report.md").write_text("# admission calibration report\n", encoding="utf-8")
    (admission_calibration_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "calibration_status": "partial",
                "calibration_confidence": "medium",
                "recommended_profile": {"min_selected_uids": 3},
                "calibration_basis": {"available_runs": 36, "query_groups": 4},
            }
        ),
        encoding="utf-8",
    )

    provider_telemetry_dir = tmp_path / "provider_telemetry"
    provider_telemetry_dir.mkdir(parents=True, exist_ok=True)
    (provider_telemetry_dir / "by_variant.csv").write_text(
        "\n".join(
            [
                "variant_label,provider,model,api_mode_used,usage_present,cost_known,model_cost_usd_total,model_cost_usd_mean_per_query,latency_p50_ms,latency_p95_ms,structured_parse_fail_rate,planner_fallback_rate,calls_total,calls_with_usage,calls_with_cost,availability,availability_reason,telemetry_source_paths",
                'stub,fake,fake-stub-v1,fixture_json,True,True,0.0,0.0,8.0,12.0,0.0,0.0,6,6,6,ok,telemetry_available,"[]"',
                'real,fake_openai,fake-gpt-4.1-mini,response_json,True,True,0.018,0.0015,84.0,133.0,0.02,0.04,6,6,6,ok,telemetry_available,"[]"',
            ]
        ),
        encoding="utf-8",
    )

    query_strength_audit_dir = tmp_path / "query_strength_audit"
    (query_strength_audit_dir / "tables").mkdir(parents=True, exist_ok=True)
    (query_strength_audit_dir / "figures").mkdir(parents=True, exist_ok=True)
    (query_strength_audit_dir / "tables" / "table_query_strength_audit.csv").write_text(
        "query_group,query_type,coverage_rate,signal_support_rate,nonzero_delta_rate,significance_available_rate,weak_query_flag,recommended_action\nchain,hard_pseudo_chain,1.0,0.83,1.0,1.0,False,promote_to_core_query_bank\nrepo_summary,summary_first,1.0,0.92,0.0,0.0,True,keep_for_analysis_only\n",
        encoding="utf-8",
    )
    (query_strength_audit_dir / "tables" / "table_query_strength_audit.md").write_text("# query strength\n", encoding="utf-8")
    (query_strength_audit_dir / "figures" / "fig_query_strength_breakdown.png").write_bytes(b"PNG")
    (query_strength_audit_dir / "figures" / "fig_query_strength_breakdown.pdf").write_bytes(b"PDF")
    (query_strength_audit_dir / "report.md").write_text("# query strength report\n", encoding="utf-8")
    (query_strength_audit_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "main_recommendation": "promote_to_core_query_bank",
                "recommended_action_counts": {
                    "keep_for_analysis_only": 1,
                    "promote_to_core_query_bank": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (provider_telemetry_dir / "summary.json").write_text(
        json.dumps(
            {
                "availability": "ok",
                "usage_present_rate": 1.0,
                "model_cost_known_rate": 1.0,
                "model_latency_p95_ms_mean": 72.5,
                "structured_parse_fail_rate_mean": 0.01,
                "planner_fallback_rate": 0.02,
                "calls_total": 12,
                "calls_with_usage": 12,
                "calls_with_cost": 12,
                "telemetry_source_paths": ["fixture_manifest"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider_normalization_dir = tmp_path / "provider_normalization"
    (provider_normalization_dir / "tables").mkdir(parents=True, exist_ok=True)
    (provider_normalization_dir / "tables" / "table_provider_normalization.csv").write_text(
        "variant_label,provider,model,api_mode_used,usage_present,prompt_tokens,completion_tokens,total_tokens,cost_known,model_cost_usd_total,model_cost_usd_mean_per_query,latency_p50_ms,latency_p95_ms,structured_parse_fail_rate,planner_fallback_rate,calls_total,calls_with_usage,calls_with_cost,normalization_status,missing_fields\n"
        "stub,fake,fake-stub-v1,fixture_json,True,0,0,0,True,0.0,0.0,8.0,12.0,0.0,0.0,6,6,6,ok,[]\n",
        encoding="utf-8",
    )
    (provider_normalization_dir / "tables" / "table_provider_normalization.md").write_text(
        "# provider normalization\n",
        encoding="utf-8",
    )
    (provider_normalization_dir / "report.md").write_text("# provider normalization report\n", encoding="utf-8")
    (provider_normalization_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "normalization_status": "ok",
                "real_call_status": "simulated",
                "missing_fields_union": [],
            }
        ),
        encoding="utf-8",
    )
    query_promotion_pack_dir = tmp_path / "query_promotion_pack"
    (query_promotion_pack_dir / "query_pack").mkdir(parents=True, exist_ok=True)
    (query_promotion_pack_dir / "query_pack" / "promoted_queries.yaml").write_text(
        "queries:\n  - query_id: chain_scene_to_object\n",
        encoding="utf-8",
    )
    (query_promotion_pack_dir / "query_pack" / "analysis_only_queries.yaml").write_text(
        "queries:\n  - query_id: repo_summary_place\n",
        encoding="utf-8",
    )
    (query_promotion_pack_dir / "query_pack" / "promotion_summary.json").write_text(
        json.dumps(
            {
                "promoted_count": 1,
                "analysis_only_count": 1,
                "dropped_count": 0,
                "source_query_bank_id": "core_real_v1",
                "source_query_bank_hash": "abc123",
                "promotion_criteria": {"recommended_action": "promote_to_core_query_bank"},
                "promotion_confidence": "high",
            }
        ),
        encoding="utf-8",
    )
    (query_promotion_pack_dir / "report.md").write_text("# query promotion pack report\n", encoding="utf-8")

    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    (freeze_dir / "freeze_manifest.json").write_text(
        json.dumps({"suite_id": "v1.42_main", "query_bank_id": "core_real_v1", "artifact_count": 5}),
        encoding="utf-8",
    )
    (freeze_dir / "artifacts_sha256.csv").write_text(
        "artifact_group,relpath,sha256,size_bytes\ncompare_meta,compare/compare_summary.json,abc,10\n",
        encoding="utf-8",
    )

    paper_map = tmp_path / "main_result_map_v1.yaml"
    paper_map.write_text(
        "\n".join(
            [
                "paper_map_id: main_result_map_v1",
                'paper_map_version: "1"',
                "description: smoke map",
                "entries:",
                "  - canonical_id: Table 1",
                "    kind: table",
                "    title: Main Results",
                "    sources:",
                "      - compare/tables/table_main_results.csv",
                "      - compare/tables/table_main_results.md",
                "  - canonical_id: Figure 2",
                "    kind: figure",
                "    title: Budget Frontier",
                "    sources:",
                "      - compare/figures/fig_main_budget_frontier.png",
                "  - canonical_id: Figure 4",
                "    kind: figure",
                "    title: Result Health",
                "    sources:",
                "      - result_health/figures/fig_result_health_breakdown.png",
            ]
        ),
        encoding="utf-8",
    )

    prompt_root = tmp_path / "prompt_assets"
    (prompt_root / "prompts" / "decisions").mkdir(parents=True, exist_ok=True)
    (prompt_root / "prompts" / "planner").mkdir(parents=True, exist_ok=True)
    (prompt_root / "prompts" / "repository").mkdir(parents=True, exist_ok=True)
    (prompt_root / "prompts" / "decisions" / "model_decision_v1.txt").write_text("decision prompt\n", encoding="utf-8")
    (prompt_root / "prompts" / "planner" / "model_planner_v1.txt").write_text("planner prompt\n", encoding="utf-8")
    (prompt_root / "prompts" / "repository" / "repo_summary_v1.txt").write_text("repo prompt\n", encoding="utf-8")
    prompt_registry = prompt_root / "registry_v1.yaml"
    prompt_registry.write_text(
        "\n".join(
            [
                "registry_id: prompt_registry_v1",
                'version: "1"',
                "entries:",
                "  - prompt_id: model_decision_v1",
                "    task: decisions",
                "    version: v1",
                "    path: prompts/decisions/model_decision_v1.txt",
                "  - prompt_id: model_planner_v1",
                "    task: planner",
                "    version: v1",
                "    path: prompts/planner/model_planner_v1.txt",
                "  - prompt_id: repo_summary_v1",
                "    task: repository",
                "    version: v1",
                "    path: prompts/repository/repo_summary_v1.txt",
                "profiles:",
                "  v1.42_main:",
                "    prompts:",
                "      decisions: model_decision_v1",
                "      planner: model_planner_v1",
                "      repository: repo_summary_v1",
            ]
        ),
        encoding="utf-8",
    )
    prompt_lock = prompt_root / "prompt_lock.json"
    prompt_lock.write_text(json.dumps({"profile": "v1.42_main", "prompts": [{"task": "planner"}]}), encoding="utf-8")

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
        "--streaming-chain-backoff-compare-dir",
        str(compare_dir / "stream_chain_backoff_cmp"),
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
        "--repo-summary-sweep-dir",
        str(compare_dir / "repo_summary_sweep"),
        "--repo-summary-retrieval-compare-dir",
        str(compare_dir / "repo_summary_retrieval_cmp"),
        "--repo-query-selection-sweep-dir",
        str(compare_dir / "repo_query_selection_sweep"),
        "--component-attribution-dir",
        str(compare_dir / "component_attr_cmp"),
        "--bye-report-compare-dir",
        str(compare_dir / "bye_report_cmp"),
        "--decisions-backend-compare-dir",
        str(compare_dir / "decisions_backend_cmp"),
        "--planner-backend-compare-dir",
        str(compare_dir / "planner_backend_cmp"),
        "--model-stack-compare-dir",
        str(compare_dir / "model_stack_cmp"),
        "--model-cost-compare-dir",
        str(compare_dir / "model_stack_cmp"),
        "--chain-nlq-dir",
        str(compare_dir / "chain_nlq"),
        "--chain-repo-compare-dir",
        str(compare_dir / "chain_repo_cmp"),
        "--chain-attribution-dir",
        str(compare_dir / "chain_attr"),
        "--signal-selection-dir",
        str(compare_dir / "signal_selection"),
        "--suite-dir",
        str(suite_dir),
        "--significance-dir",
        str(significance_dir),
        "--result-health-dir",
        str(result_health_dir),
        "--result-diagnosis-dir",
        str(result_diagnosis_dir),
        "--delta-audit-dir",
        str(delta_audit_dir),
        "--provider-telemetry-dir",
        str(provider_telemetry_dir),
        "--provider-normalization-dir",
        str(provider_normalization_dir),
        "--admission-calibration-dir",
        str(admission_calibration_dir),
        "--query-strength-audit-dir",
        str(query_strength_audit_dir),
        "--query-promotion-pack-dir",
        str(query_promotion_pack_dir),
        "--benchmark-freeze-dir",
        str(freeze_dir),
        "--paper-map",
        str(paper_map),
        "--prompt-registry",
        str(prompt_registry),
        "--prompt-lock",
        str(prompt_lock),
        "--export-submission-pack",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_table_panel=" in proc.stdout
    assert "saved_table_delta=" in proc.stdout
    assert "saved_submission_pack=" in proc.stdout

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
    report_text = report_md.read_text(encoding="utf-8")
    assert "## Main Result Contract" in report_text
    assert "## Canonical Paper Map" in report_text
    assert "## Result Health" in report_text
    assert "## Admission Control" in report_text
    assert "## Admission Calibration" in report_text
    assert "## Result Diagnosis" in report_text
    assert "## Delta Audit" in report_text
    assert "## Query Strength Audit" in report_text
    assert "## Provider Telemetry" in report_text
    assert "## Provider Normalization" in report_text
    assert "## Query Promotion Pack" in report_text
    assert "## Benchmark Freeze" in report_text
    assert (out_dir / "canonical" / "tables" / "Table_1.csv").exists()
    assert (out_dir / "canonical" / "tables" / "Table_1.md").exists()
    assert (out_dir / "canonical" / "figures" / "Figure_2.png").exists()
    assert (out_dir / "canonical" / "figures" / "Figure_4.png").exists()
    assert (out_dir / "canonical" / "table_paper_artifact_map.csv").exists()
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
    assert (out_dir / "tables" / "table_streaming_chain_backoff_compare.csv").exists()
    assert (out_dir / "tables" / "table_streaming_chain_backoff_compare.md").exists()
    assert (out_dir / "figures" / "fig_streaming_chain_backoff_success_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_streaming_chain_backoff_latency_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_streaming_chain_backoff_delta.png").exists()
    assert (out_dir / "selection" / "coverage.md").exists()
    assert (out_dir / "selection" / "selection_report.md").exists()
    assert (out_dir / "selection" / "selected_uids.txt").exists()
    assert (out_dir / "significance" / "table_significance_main.csv").exists()
    assert (out_dir / "significance" / "table_confidence_intervals.csv").exists()
    assert (out_dir / "significance" / "report.md").exists()
    assert (out_dir / "result_health" / "tables" / "table_result_health.csv").exists()
    assert (out_dir / "result_health" / "figures" / "fig_result_health_breakdown.png").exists()
    assert (out_dir / "admission_control" / "snapshot.json").exists()
    assert (out_dir / "admission_calibration" / "snapshot.json").exists()
    assert (out_dir / "result_diagnosis" / "tables" / "table_result_diagnosis.csv").exists()
    assert (out_dir / "result_diagnosis" / "report.md").exists()
    assert (out_dir / "delta_audit" / "tables" / "table_delta_audit.csv").exists()
    assert (out_dir / "delta_audit" / "report.md").exists()
    assert (out_dir / "query_strength_audit" / "tables" / "table_query_strength_audit.csv").exists()
    assert (out_dir / "query_strength_audit" / "report.md").exists()
    assert (out_dir / "provider_normalization" / "tables" / "table_provider_normalization.csv").exists()
    assert (out_dir / "provider_normalization" / "snapshot.json").exists()
    assert (out_dir / "query_promotion_pack" / "query_pack" / "promotion_summary.json").exists()
    assert (out_dir / "query_promotion_pack" / "report.md").exists()
    assert (out_dir / "figures" / "fig_result_diagnosis_breakdown.png").exists()
    assert (out_dir / "figures" / "fig_delta_audit_breakdown.png").exists()
    assert (out_dir / "figures" / "fig_query_strength_breakdown.png").exists()
    assert (out_dir / "provider_telemetry" / "summary.json").exists()
    assert (out_dir / "provider_telemetry" / "by_variant.csv").exists()
    assert (out_dir / "freeze" / "freeze_manifest.json").exists()
    assert (out_dir / "freeze" / "artifacts_sha256.csv").exists()
    assert (out_dir / "manifest" / "experiment_manifest.yaml").exists()
    assert (out_dir / "manifest" / "manifest_resolved.json").exists()
    assert (out_dir / "manifest" / "prompt_lock.json").exists()
    assert (out_dir / "manifest" / "query_bank_lock.json").exists()
    assert (out_dir / "manifest" / "query_banks" / "core_real_v1.yaml").exists()
    assert (out_dir / "provenance" / "results_long.csv").exists()
    assert (out_dir / "provenance" / "runs.jsonl").exists()
    assert (out_dir / "prompts" / "registry_v1.yaml").exists()
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
    assert (out_dir / "repo_summary" / "metrics_by_policy_budget.csv").exists()
    assert (out_dir / "repo_summary" / "metrics_by_policy_budget.md").exists()
    assert (out_dir / "figures" / "fig_repo_summary_quality_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_summary_size_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_summary_delta_vs_budget_seconds.png").exists()
    assert (out_dir / "repo_summary_retrieval" / "table_repo_summary_retrieval_compare.csv").exists()
    assert (out_dir / "repo_summary_retrieval" / "table_repo_summary_retrieval_compare.md").exists()
    assert (out_dir / "figures" / "fig_repo_summary_retrieval_quality_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_repo_summary_retrieval_delta.png").exists()
    assert (out_dir / "figures" / "fig_repo_summary_retrieval_candidate_scale.png").exists()
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
    assert (out_dir / "decisions_backend" / "table_decisions_backend_compare.csv").exists()
    assert (out_dir / "decisions_backend" / "table_decisions_backend_compare.md").exists()
    assert (out_dir / "figures" / "fig_decisions_backend_delta.png").exists()
    assert (out_dir / "figures" / "fig_decisions_backend_tradeoff.png").exists()
    assert (out_dir / "planner_backend" / "table_planner_backend_compare.csv").exists()
    assert (out_dir / "planner_backend" / "table_planner_backend_compare.md").exists()
    assert (out_dir / "figures" / "fig_planner_backend_delta.png").exists()
    assert (out_dir / "figures" / "fig_planner_backend_tradeoff.png").exists()
    assert (out_dir / "model_stack" / "table_model_stack_compare.csv").exists()
    assert (out_dir / "model_stack" / "table_model_stack_compare.md").exists()
    assert (out_dir / "model_stack" / "table_model_cost_compare.csv").exists()
    assert (out_dir / "model_stack" / "table_model_cost_compare.md").exists()
    assert (out_dir / "figures" / "fig_model_stack_delta.png").exists()
    assert (out_dir / "figures" / "fig_model_stack_tradeoff.png").exists()
    assert (out_dir / "figures" / "fig_model_cost_vs_quality.png").exists()
    assert (out_dir / "figures" / "fig_model_parse_fail_rate.png").exists()
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
    assert (out_dir / "chain_attribution" / "table_chain_attribution.csv").exists()
    assert (out_dir / "chain_attribution" / "table_chain_attribution.md").exists()
    assert (out_dir / "chain_attribution" / "table_chain_failure_breakdown.csv").exists()
    assert (out_dir / "chain_attribution" / "table_chain_failure_breakdown.md").exists()
    assert (out_dir / "figures" / "fig_chain_attribution_success_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_chain_attribution_delta_success_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_chain_attribution_backoff_vs_budget_seconds.png").exists()

    header = panel_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "task" in header
    assert "budget_seconds" in header
    assert "primary_a" in header
    assert "primary_b" in header
    assert "delta_primary" in header
    assert "safety_critical_fn_rate_a" in header

    submission_pack = out_dir / "submission_pack"
    assert (submission_pack / "README.md").exists()
    assert (submission_pack / "snapshot.json").exists()
    assert (submission_pack / "paper_ready" / "tables" / "table_budget_panel.csv").exists()
    assert (submission_pack / "paper_ready" / "figures" / "fig_budget_primary_vs_seconds_panel.png").exists()
    assert (submission_pack / "significance" / "tables" / "table_significance_main.csv").exists()
    assert (submission_pack / "significance" / "report.md").exists()
    assert (submission_pack / "result_health" / "tables" / "table_result_health.csv").exists()
    assert (submission_pack / "admission_control" / "snapshot.json").exists()
    assert (submission_pack / "admission_calibration" / "snapshot.json").exists()
    assert (submission_pack / "result_diagnosis" / "tables" / "table_result_diagnosis.csv").exists()
    assert (submission_pack / "delta_audit" / "tables" / "table_delta_audit.csv").exists()
    assert (submission_pack / "query_strength_audit" / "tables" / "table_query_strength_audit.csv").exists()
    assert (submission_pack / "provider_telemetry" / "summary.json").exists()
    assert (submission_pack / "provider_normalization" / "snapshot.json").exists()
    assert (submission_pack / "query_promotion_pack" / "query_pack" / "promotion_summary.json").exists()
    assert (submission_pack / "freeze" / "freeze_manifest.json").exists()
    assert (submission_pack / "manifest" / "experiment_manifest.yaml").exists()
    assert (submission_pack / "manifest" / "prompt_lock.json").exists()
    assert (submission_pack / "manifest" / "query_bank_lock.json").exists()
    assert (submission_pack / "manifest" / "query_banks" / "core_real_v1.yaml").exists()
    assert (submission_pack / "manifest" / "main_result_map_v1.yaml").exists()
    assert (submission_pack / "compare" / "tables" / "table_main_results.csv").exists()
    assert (submission_pack / "provenance" / "results_long.csv").exists()
    assert (submission_pack / "provenance" / "commands.sh").exists()
    assert (submission_pack / "prompts" / "registry_v1.yaml").exists()
    assert (submission_pack / "prompts" / "decisions" / "model_decision_v1.txt").exists()
    assert (submission_pack / "prompts" / "planner" / "model_planner_v1.txt").exists()
    assert (submission_pack / "prompts" / "repository" / "repo_summary_v1.txt").exists()
    assert (submission_pack / "paper_ready" / "canonical" / "tables" / "Table_1.csv").exists()
    submission_readme = (submission_pack / "README.md").read_text(encoding="utf-8")
    assert "Table 1" in submission_readme
    assert "Figure 2" in submission_readme
    assert "Admission First" in submission_readme
    assert "Calibration First" in submission_readme
    assert "Diagnosis First" in submission_readme
    assert "Delta Audit" in submission_readme
    assert "Query Strength Audit" in submission_readme
    assert "Provider Noise" in submission_readme
    assert "Normalized Telemetry" in submission_readme
    assert "Query Promotion" in submission_readme
