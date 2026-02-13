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
    lines.append(f"- steps: {int(summary.get('steps', 0))}")
    lines.append(f"- queries_total: {int(summary.get('queries_total', 0))}")
    lines.append(f"- avg_trials_per_query: {float(summary.get('avg_trials_per_query', 0.0)):.4f}")
    lines.append(f"- retrieval_latency_p50_ms: {float(summary.get('retrieval_latency_p50_ms', 0.0)):.3f}")
    lines.append(f"- retrieval_latency_p95_ms: {float(summary.get('retrieval_latency_p95_ms', 0.0)):.3f}")
    lines.append(f"- e2e_latency_p50_ms: {float(summary.get('e2e_latency_p50_ms', 0.0)):.3f}")
    lines.append(f"- e2e_latency_p95_ms: {float(summary.get('e2e_latency_p95_ms', 0.0)):.3f}")
    lines.append(f"- hit_at_k_strict: {float(summary.get('hit_at_k_strict', 0.0)):.4f}")
    lines.append(f"- top1_in_distractor_rate: {float(summary.get('top1_in_distractor_rate', 0.0)):.4f}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming online budget-policy smoke (fixed/recommend/adaptive)")
    parser.add_argument("--json", required=True, help="Input pipeline json (v03 decisions)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--step-s", type=float, default=8.0, help="Streaming step size (seconds)")
    parser.add_argument("--mode", default="hard_pseudo_nlq", help="Query mode (default hard_pseudo_nlq)")
    parser.add_argument("--budgets", required=True, help='Budget list like "20/50/4,40/100/8,60/200/12"')
    parser.add_argument("--budget-policy", choices=["fixed", "recommend", "adaptive"], default="fixed")
    parser.add_argument("--fixed-budget", default="40/100/8", help="Budget key for fixed policy")
    parser.add_argument("--recommend-dir", default=None, help="Path to v1.5 budget_recommend output dir")
    parser.add_argument("--policy-gates-json", default=None, help="JSON string/path for adaptive gate constraints")
    parser.add_argument("--policy-targets-json", default=None, help="JSON string/path for adaptive targets")
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
            budgets=budgets,
            budget_policy=str(args.budget_policy),
            fixed_budget=str(args.fixed_budget),
            recommend_dir=args.recommend_dir,
            policy_gates=gates,
            policy_targets=targets,
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
    report_md = out_dir / "report.md"
    snapshot_json = out_dir / "snapshot.json"

    _write_csv(steps_csv, list(payload.get("step_rows", [])))
    _write_csv(queries_csv, list(payload.get("query_rows", [])))
    _write_report(report_md, payload, policy_gates=gates, policy_targets=targets)

    summary = dict(payload.get("summary", {}))
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json": str(args.json),
            "step_s": float(args.step_s),
            "mode": str(args.mode),
            "budgets": [b.key for b in budgets],
            "budget_policy": str(args.budget_policy),
            "fixed_budget": str(args.fixed_budget),
            "recommend_dir": str(args.recommend_dir) if args.recommend_dir else None,
            "policy_gates": gates,
            "policy_targets": targets,
            "top_k": int(args.top_k),
            "seed": int(args.seed),
        },
        "summary": summary,
        "outputs": {
            "steps_csv": str(steps_csv),
            "queries_csv": str(queries_csv),
            "report_md": str(report_md),
        },
    }
    snapshot_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"policy={args.budget_policy}")
    print(f"budgets={len(budgets)}")
    print(f"steps={int(summary.get('steps', 0))}")
    print(f"queries_total={int(summary.get('queries_total', 0))}")
    print(f"avg_trials_per_query={float(summary.get('avg_trials_per_query', 0.0)):.4f}")
    print(f"saved_steps={steps_csv}")
    print(f"saved_queries={queries_csv}")
    print(f"saved_report={report_md}")
    print(f"saved_snapshot={snapshot_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
