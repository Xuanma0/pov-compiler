from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.budget_policy import parse_budget_keys
from pov_compiler.streaming.runner import StreamingConfig, run_streaming


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _parse_json_dict(raw: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if raw is None:
        return dict(default)
    text = str(raw).strip()
    if not text:
        return dict(default)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    path = Path(text)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return dict(default)


def _write_report(path: Path, payload: dict[str, Any], policy_gates: dict[str, Any], policy_targets: dict[str, Any]) -> None:
    summary = dict(payload.get("summary", {}))
    steps = list(payload.get("step_rows", []))
    lines: list[str] = []
    lines.append("# Streaming Budget Smoke Report")
    lines.append("")
    lines.append(f"- video_id: {summary.get('video_id', '')}")
    lines.append(f"- policy: {summary.get('policy_name', '')}")
    lines.append(f"- budgets: {summary.get('budgets', [])}")
    lines.append(f"- context_use_repo: {bool(summary.get('context_use_repo', False))}")
    lines.append(f"- repo_read_policy: {summary.get('repo_read_policy', '')}")
    lines.append(f"- steps: {int(summary.get('steps', 0))}")
    lines.append(f"- queries_total: {int(summary.get('queries_total', 0))}")
    lines.append(f"- avg_trials_per_query: {float(summary.get('avg_trials_per_query', 0.0)):.4f}")
    lines.append(f"- retrieval_latency_p50_ms: {float(summary.get('retrieval_latency_p50_ms', 0.0)):.3f}")
    lines.append(f"- retrieval_latency_p95_ms: {float(summary.get('retrieval_latency_p95_ms', 0.0)):.3f}")
    lines.append(f"- e2e_latency_p50_ms: {float(summary.get('e2e_latency_p50_ms', 0.0)):.3f}")
    lines.append(f"- e2e_latency_p95_ms: {float(summary.get('e2e_latency_p95_ms', 0.0)):.3f}")
    lines.append(f"- hit_at_k_strict: {float(summary.get('hit_at_k_strict', 0.0)):.4f}")
    lines.append(f"- top1_in_distractor_rate: {float(summary.get('top1_in_distractor_rate', 0.0)):.4f}")
    lines.append(f"- safety_critical_fn_rate: {float(summary.get('safety_critical_fn_rate', 0.0)):.4f}")
    lines.append(f"- safety_budget_insufficient_rate: {float(summary.get('safety_budget_insufficient_rate', 0.0)):.4f}")
    lines.append(f"- latency_cap_ms: {float(summary.get('latency_cap_ms', 0.0)):.3f}")
    lines.append(f"- max_trials_per_query: {int(summary.get('max_trials_per_query', 0))}")
    lines.append(f"- num_escalate: {int(summary.get('num_escalate', 0))}")
    lines.append(f"- num_deescalate: {int(summary.get('num_deescalate', 0))}")
    lines.append(f"- num_accept: {int(summary.get('num_accept', 0))}")
    lines.append(f"- num_give_up: {int(summary.get('num_give_up', 0))}")
    lines.append(f"- intervention_trials_total: {int(summary.get('intervention_trials_total', 0))}")
    lines.append(f"- intervention_success_rate: {float(summary.get('intervention_success_rate', 0.0)):.4f}")
    lines.append(f"- policy_action_order: `{json.dumps(summary.get('policy_action_order', {}), ensure_ascii=False, sort_keys=True)}`")
    lines.append(f"- intervention_cfg_name: {summary.get('intervention_cfg_name', '')}")
    lines.append(f"- intervention_cfg_hash: {summary.get('intervention_cfg_hash', '')}")
    lines.append(
        "- e2e_includes: step slicing + events_v1 incremental update + index update + policy trials + retrieval + rerank + metrics + write"
    )
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(f"- gates: `{json.dumps(policy_gates, ensure_ascii=False, sort_keys=True)}`")
    lines.append(f"- targets: `{json.dumps(policy_targets, ensure_ascii=False, sort_keys=True)}`")
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    lines.append(
        "| step_idx | t0_s | t1_s | index_size | events_v1_added | avg_trials_per_query | retrieval_latency_p50_ms | retrieval_latency_p95_ms | e2e_ms | hit@k_strict | fp_rate |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in steps:
        lines.append(
            f"| {int(row.get('step_idx', 0))} | {float(row.get('t0_s', 0.0)):.3f} | "
            f"{float(row.get('t1_s', row.get('end_t', 0.0))):.3f} | {int(row.get('index_size', 0))} | "
            f"{int(row.get('events_v1_added', 0))} | {float(row.get('avg_trials_per_query', 0.0)):.3f} | "
            f"{float(row.get('retrieval_latency_p50_ms', 0.0)):.3f} | {float(row.get('retrieval_latency_p95_ms', 0.0)):.3f} | "
            f"{float(row.get('e2e_ms', 0.0)):.3f} | {float(row.get('hit_at_k_strict', 0.0)):.4f} | {float(row.get('fp_rate', 0.0)):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(out_dir: Path, query_rows: list[dict[str, Any]], formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_fig = out_dir / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        list(query_rows),
        key=lambda r: (
            int(r.get("step_idx", 0)),
            str(r.get("query_id", "")),
            int(r.get("trial_index", 0)),
        ),
    )
    final_rows = [r for r in rows if int(r.get("final_trial", 0)) == 1]
    plot_rows = final_rows if final_rows else rows
    x = list(range(1, len(plot_rows) + 1))
    y_budget = [float(r.get("budget_seconds", r.get("chosen_budget_seconds", 0.0)) or 0.0) for r in plot_rows]
    all_paths: list[str] = []

    p1 = out_fig / "fig_policy_budget_over_queries"
    plt.figure(figsize=(8.2, 4.4))
    plt.plot(x, y_budget, marker="o", linewidth=1.2)
    esc_x = [i for i, r in enumerate(plot_rows, start=1) if str(r.get("action", "")).startswith("escalate_")]
    esc_y = [y_budget[i - 1] for i in esc_x]
    dec_x = [i for i, r in enumerate(plot_rows, start=1) if str(r.get("action", "")).startswith("deescalate_")]
    dec_y = [y_budget[i - 1] for i in dec_x]
    if esc_x:
        plt.scatter(esc_x, esc_y, marker="^", s=70, label="escalate")
    if dec_x:
        plt.scatter(dec_x, dec_y, marker="v", s=70, label="deescalate")
    plt.xlabel("Query Index (final trials)")
    plt.ylabel("Budget Seconds")
    plt.title("Online Policy Budget Switch Trace")
    plt.grid(True, alpha=0.35)
    if esc_x or dec_x:
        plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        all_paths.append(str(p))
    plt.close()

    p2 = out_fig / "fig_policy_safety_vs_latency"
    lat = [float(r.get("latency_e2e_ms", r.get("e2e_ms", 0.0)) or 0.0) for r in plot_rows]
    safety = [float(r.get("safety_is_critical_fn", 0.0) or 0.0) for r in plot_rows]
    plt.figure(figsize=(8.2, 4.4))
    ax1 = plt.gca()
    ax1.plot(x, lat, marker="o", linestyle="-", label="latency_e2e_ms")
    ax1.set_xlabel("Query Index (final trials)")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(True, alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(x, safety, marker="s", linestyle="--", label="safety_is_critical_fn")
    ax2.set_ylabel("Safety Critical FN (0/1)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Policy Safety vs Latency")
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        all_paths.append(str(p))
    plt.close()

    # 3) intervention trial count / actions over queries
    p3 = out_fig / "fig_policy_interventions_over_queries"
    trials_by_query: dict[str, int] = {}
    action_by_query: dict[str, str] = {}
    for r in rows:
        qid = str(r.get("query_id", ""))
        if not qid:
            continue
        trials_by_query[qid] = max(int(trials_by_query.get(qid, 0)), int(r.get("trial_index", 0) or 0))
        if int(r.get("final_trial", 0)) == 1:
            action_by_query[qid] = str(r.get("action", ""))
    qids = sorted(trials_by_query.keys())
    xq = list(range(1, len(qids) + 1))
    ytr = [int(trials_by_query[q]) for q in qids]
    plt.figure(figsize=(8.2, 4.4))
    plt.bar(xq, ytr)
    for i, qid in enumerate(qids, start=1):
        act = action_by_query.get(qid, "")
        if act:
            plt.text(i, ytr[i - 1] + 0.05, act, rotation=45, ha="left", va="bottom", fontsize=7)
    plt.xlabel("Query Index")
    plt.ylabel("Trials Count")
    plt.title("Policy Interventions Over Queries")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        p = p3.with_suffix(f".{ext}")
        plt.savefig(p)
        all_paths.append(str(p))
    plt.close()

    # 4) intervention attribution/action breakdown
    p4 = out_fig / "fig_policy_intervention_breakdown"
    finals = [r for r in rows if int(r.get("final_trial", 0)) == 1]
    keys: list[str] = []
    counts: dict[str, int] = {}
    succ: dict[str, float] = {}
    for r in finals:
        key = f"{str(r.get('attribution', '') or 'none')}->{str(r.get('action', '') or 'none')}"
        if key not in counts:
            keys.append(key)
            counts[key] = 0
            succ[key] = 0.0
        counts[key] += 1
        succ[key] += float(r.get("success", 0.0) or 0.0)
    keys = sorted(keys)
    xk = list(range(len(keys)))
    ycnt = [counts[k] for k in keys]
    ysucc = [(succ[k] / max(1, counts[k])) for k in keys]
    plt.figure(figsize=(9.2, 4.8))
    if keys:
        plt.bar(xk, ycnt, label="count")
        ax2 = plt.gca().twinx()
        ax2.plot(xk, ysucc, marker="o", linestyle="--", label="success_rate")
        plt.gca().set_xticks(xk)
        plt.gca().set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
        plt.gca().set_ylabel("Count")
        ax2.set_ylabel("Success Rate")
        l1, lab1 = plt.gca().get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        plt.gca().legend(l1 + l2, lab1 + lab2, loc="upper right")
    else:
        plt.text(0.5, 0.5, "no interventions", ha="center", va="center")
    plt.title("Policy Intervention Breakdown")
    plt.tight_layout()
    for ext in formats:
        p = p4.with_suffix(f".{ext}")
        plt.savefig(p)
        all_paths.append(str(p))
    plt.close()
    return all_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming online budget-policy smoke (fixed/recommend/adaptive)")
    parser.add_argument("--json", required=True, help="Input pipeline json (v03 decisions)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--step-s", type=float, default=8.0, help="Streaming step size (seconds)")
    parser.add_argument("--mode", default="hard_pseudo_nlq", help="Query mode (default hard_pseudo_nlq)")
    parser.add_argument("--query", action="append", default=[], help="Optional query text (repeatable)")
    parser.add_argument("--budgets", required=True, help='Budget list like "20/50/4,40/100/8,60/200/12"')
    parser.add_argument("--codec-name", choices=["all_events", "fixed_k"], default="all_events")
    parser.add_argument("--codec-k", type=int, default=16, help="K for fixed_k codec")
    parser.add_argument("--codec-cfg", default=None, help="Optional codec cfg path (yaml/json)")
    group_repo = parser.add_mutually_exclusive_group()
    group_repo.add_argument("--context-use-repo", dest="context_use_repo", action="store_true")
    group_repo.add_argument("--no-context-use-repo", dest="context_use_repo", action="store_false")
    parser.set_defaults(context_use_repo=False)
    parser.add_argument("--repo-read-policy", default="budgeted_topk", help="Repo read policy name (e.g. budgeted_topk/query_aware)")
    parser.add_argument("--repo-budget", default=None, help='Optional repo budget key like "40/100/8"')
    parser.add_argument(
        "--budget-policy",
        "--policy",
        dest="budget_policy",
        choices=["fixed", "recommend", "adaptive", "safety_latency", "safety_latency_intervention"],
        default="fixed",
    )
    parser.add_argument("--fixed-budget", default="40/100/8", help="Budget key for fixed policy")
    parser.add_argument("--recommend-dir", default=None, help="Path to v1.5 budget_recommend output dir")
    parser.add_argument(
        "--intervention-cfg",
        default=None,
        help="YAML path for safety_latency_intervention action-selection config",
    )
    parser.add_argument("--policy-gates-json", default=None, help="JSON string/path for adaptive gate constraints")
    parser.add_argument("--policy-targets-json", default=None, help="JSON string/path for adaptive targets")
    parser.add_argument("--latency-cap-ms", type=float, default=25.0, help="Latency cap for safety_latency policy")
    parser.add_argument("--max-trials-per-query", "--max-trials", dest="max_trials_per_query", type=int, default=3, help="Max trials per query for safety policies")
    parser.add_argument("--strict-threshold", type=float, default=1.0, help="Strict hit@k threshold for safety intervention success")
    parser.add_argument("--max-top1-in-distractor-rate", type=float, default=0.2, help="Risk threshold for intervention success")
    group_pref = parser.add_mutually_exclusive_group()
    group_pref.add_argument("--prefer-lower-budget", dest="prefer_lower_budget", action="store_true")
    group_pref.add_argument("--prefer-higher-budget", dest="prefer_lower_budget", action="store_false")
    parser.set_defaults(prefer_lower_budget=True)
    parser.add_argument(
        "--escalate-on-reason",
        action="append",
        default=[],
        help="Reason names that trigger escalation in safety_latency policy (repeatable)",
    )
    parser.add_argument("--formats", default="png,pdf", help="Figure formats, e.g. png,pdf")
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    budgets = parse_budget_keys(args.budgets)
    if not budgets:
        print("error=no budgets parsed")
        return 2

    default_gates = {
        "top1_in_distractor_rate": {"op": "<=", "value": 0.20},
        "fp_rate": {"op": "<=", "value": 0.20},
    }
    default_targets = {"hit_at_k_strict": {"op": ">=", "value": 0.60}}
    gates = _parse_json_dict(args.policy_gates_json, default_gates)
    targets = _parse_json_dict(args.policy_targets_json, default_targets)

    payload = run_streaming(
        Path(args.json),
        config=StreamingConfig(
            step_s=float(args.step_s),
            top_k=int(args.top_k),
            queries=[str(x) for x in list(args.query or []) if str(x).strip()],
            budgets=budgets,
            budget_policy=str(args.budget_policy),
            fixed_budget=str(args.fixed_budget),
            recommend_dir=args.recommend_dir,
            policy_gates=gates,
            policy_targets=targets,
            latency_cap_ms=float(args.latency_cap_ms),
            max_trials_per_query=int(args.max_trials_per_query),
            strict_threshold=float(args.strict_threshold),
            max_top1_in_distractor_rate=float(args.max_top1_in_distractor_rate),
            prefer_lower_budget=bool(args.prefer_lower_budget),
            escalate_on_reasons=[str(x).strip() for x in list(args.escalate_on_reason or []) if str(x).strip()]
            or ["budget_insufficient"],
            intervention_cfg=args.intervention_cfg,
            codec_name=str(args.codec_name),
            codec_k=int(args.codec_k),
            codec_cfg=args.codec_cfg,
            context_use_repo=bool(args.context_use_repo),
            repo_read_policy=str(args.repo_read_policy),
            repo_budget=str(args.repo_budget) if args.repo_budget else None,
            allow_gt_fallback=False,
            nlq_mode=str(args.mode),
            nlq_seed=int(args.seed),
            nlq_n_highlight=10,
            nlq_n_token=10,
            nlq_n_decision=10,
            mode="budgeted",
        ),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_csv = out_dir / "steps.csv"
    queries_csv = out_dir / "queries.csv"
    interventions_csv = out_dir / "interventions.csv"
    report_md = out_dir / "report.md"
    snapshot_json = out_dir / "snapshot.json"

    _write_csv(steps_csv, list(payload.get("step_rows", [])))
    qrows = list(payload.get("query_rows", []))
    _write_csv(queries_csv, qrows)
    _write_csv(interventions_csv, [r for r in qrows if str(r.get("action", "")) not in {"", "accept"}])
    _write_report(report_md, payload, policy_gates=gates, policy_targets=targets)
    figure_paths = _make_figures(
        out_dir=out_dir,
        query_rows=list(payload.get("query_rows", [])),
        formats=[x.strip() for x in str(args.formats).split(",") if x.strip()],
    )

    summary = dict(payload.get("summary", {}))
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json": str(args.json),
            "step_s": float(args.step_s),
            "mode": str(args.mode),
            "budgets": [b.key for b in budgets],
            "codec_name": str(args.codec_name),
            "codec_k": int(args.codec_k),
            "codec_cfg": str(args.codec_cfg) if args.codec_cfg else None,
            "context_use_repo": bool(args.context_use_repo),
            "repo_read_policy": str(args.repo_read_policy),
            "repo_budget": str(args.repo_budget) if args.repo_budget else None,
            "budget_policy": str(args.budget_policy),
            "fixed_budget": str(args.fixed_budget),
            "recommend_dir": str(args.recommend_dir) if args.recommend_dir else None,
            "intervention_cfg": str(args.intervention_cfg) if args.intervention_cfg else None,
            "queries": [str(x) for x in list(args.query or []) if str(x).strip()],
            "policy_gates": gates,
            "policy_targets": targets,
            "latency_cap_ms": float(args.latency_cap_ms),
            "max_trials_per_query": int(args.max_trials_per_query),
            "strict_threshold": float(args.strict_threshold),
            "max_top1_in_distractor_rate": float(args.max_top1_in_distractor_rate),
            "prefer_lower_budget": bool(args.prefer_lower_budget),
            "escalate_on_reasons": [str(x).strip() for x in list(args.escalate_on_reason or []) if str(x).strip()]
            or ["budget_insufficient"],
            "action_order": summary.get("policy_action_order", {}),
            "action_set": sorted(
                {
                    str(k)
                    for k in (summary.get("policy_action_counts", {}) or {}).keys()
                    if str(k)
                }
            ),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "intervention_cfg_name": str(summary.get("intervention_cfg_name", "")),
            "intervention_cfg_hash": str(summary.get("intervention_cfg_hash", "")),
        },
        "summary": summary,
        "outputs": {
            "steps_csv": str(steps_csv),
            "queries_csv": str(queries_csv),
            "interventions_csv": str(interventions_csv),
            "figures": figure_paths,
            "report_md": str(report_md),
        },
    }
    snapshot_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"policy={args.budget_policy}")
    print(f"codec_name={args.codec_name}")
    print(f"codec_k={int(args.codec_k)}")
    print(f"context_use_repo={bool(args.context_use_repo)}")
    print(f"repo_read_policy={str(args.repo_read_policy)}")
    print(f"budgets={len(budgets)}")
    print(f"steps={int(summary.get('steps', 0))}")
    print(f"queries_total={int(summary.get('queries_total', 0))}")
    print(f"avg_trials_per_query={float(summary.get('avg_trials_per_query', 0.0)):.4f}")
    print(f"intervention_cfg_name={str(summary.get('intervention_cfg_name', ''))}")
    print(f"intervention_cfg_hash={str(summary.get('intervention_cfg_hash', ''))}")
    print(f"saved_steps={steps_csv}")
    print(f"saved_queries={queries_csv}")
    print(f"saved_interventions={interventions_csv}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_report={report_md}")
    print(f"saved_snapshot={snapshot_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
