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
    (compare_dir / "stream_policy_cmp" / "compare_summary.json").write_text(
        json.dumps({"policy_a": "safety_latency", "policy_b": "safety_latency_intervention"}, ensure_ascii=False),
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
        "--streaming-intervention-sweep-dir",
        str(compare_dir / "stream_intervention_sweep"),
        "--streaming-codec-sweep-dir",
        str(compare_dir / "stream_codec_sweep"),
        "--reranker-sweep-dir",
        str(compare_dir / "reranker_sweep"),
        "--repo-policy-sweep-dir",
        str(compare_dir / "repo_policy_sweep"),
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

    header = panel_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "task" in header
    assert "budget_seconds" in header
    assert "primary_a" in header
    assert "primary_b" in header
    assert "delta_primary" in header
    assert "safety_critical_fn_rate_a" in header
