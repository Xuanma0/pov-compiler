from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _parse_bool_text(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming compare: non-repo vs repo-aware context")
    parser.add_argument("--json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--step-s", type=float, default=8.0)
    parser.add_argument("--budgets", required=True, help='e.g. "20/50/4,60/200/12"')
    parser.add_argument("--query", action="append", default=[], help="Repeatable query")
    parser.add_argument("--policy-a", default="safety_latency")
    parser.add_argument("--policy-b", default="safety_latency")
    parser.add_argument("--a-use-repo", type=_parse_bool_text, default=False)
    parser.add_argument("--b-use-repo", type=_parse_bool_text, default=True)
    parser.add_argument("--a-repo-policy", default="budgeted_topk")
    parser.add_argument("--b-repo-policy", default="query_aware")
    parser.add_argument("--intervention-cfg", default=None, help="Optional cfg for run_b")
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--max-trials-per-query", type=int, default=3)
    parser.add_argument("--latency-cap-ms", type=float, default=25.0)
    parser.add_argument("--strict-threshold", type=float, default=1.0)
    parser.add_argument("--max-top1-in-distractor-rate", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def _render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _write_commands(path: Path, cmd: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(f"# {ts}\n{_render_cmd(cmd)}\n\n")


def _run(cmd: list[str], *, cwd: Path, log_prefix: Path, commands_file: Path) -> int:
    _write_commands(commands_file, cmd)
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    (log_prefix.with_suffix(".stdout.log")).write_text(result.stdout or "", encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    return int(result.returncode)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    idx = (len(xs) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def _final_query_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    finals = [r for r in rows if int(float(_to_float(r.get("final_trial")) or 0.0)) == 1]
    if finals:
        return finals
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        qid = str(row.get("query_id", ""))
        idx = int(float(_to_float(row.get("trial_index")) or 0.0))
        prev = latest.get(qid)
        if prev is None or idx >= int(float(_to_float(prev.get("trial_index")) or 0.0)):
            latest[qid] = row
    return [latest[k] for k in sorted(latest.keys())]


def _collect_metrics(run_dir: Path) -> dict[str, float | int]:
    qrows = _read_csv(run_dir / "queries.csv")
    finals = _final_query_rows(qrows)
    strict_vals: list[float] = []
    mrr_vals: list[float] = []
    critical_vals: list[float] = []
    distractor_vals: list[float] = []
    e2e_vals: list[float] = []
    trials_vals: list[float] = []
    repo_selected_vals: list[float] = []
    context_chars_vals: list[float] = []
    for row in finals:
        strict_vals.append(float(_to_float(row.get("hit_at_k_strict")) or 0.0))
        mrr_vals.append(float(_to_float(row.get("mrr")) or 0.0))
        critical_vals.append(float(_to_float(row.get("safety_is_critical_fn")) or 0.0))
        distractor_vals.append(float(_to_float(row.get("top1_in_distractor_rate")) or 0.0))
        e2e_vals.append(float(_to_float(row.get("latency_e2e_ms")) or _to_float(row.get("e2e_ms")) or 0.0))
        trials_vals.append(float(_to_float(row.get("trial_count_for_query")) or _to_float(row.get("trials_count")) or 1.0))
        repo_selected_vals.append(float(_to_float(row.get("repo_selected_chunks")) or 0.0))
        context_chars_vals.append(float(_to_float(row.get("context_len_chars")) or 0.0))

    def _mean(vals: list[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "strict_hit_at_k_rate": _mean(strict_vals),
        "mrr": _mean(mrr_vals),
        "critical_fn_rate": _mean(critical_vals),
        "top1_in_distractor_rate": _mean(distractor_vals),
        "latency_p50_ms": _percentile(e2e_vals, 50.0),
        "latency_p95_ms": _percentile(e2e_vals, 95.0),
        "avg_trials_per_query": _mean(trials_vals),
        "repo_selected_chunks_mean": _mean(repo_selected_vals),
        "context_len_chars_mean": _mean(context_chars_vals),
        "queries_total": int(len(finals)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, row: dict[str, Any], columns: list[str], summary_lines: list[str]) -> None:
    lines = [
        "# Streaming Repo Compare",
        "",
        *summary_lines,
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
        "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(
    out_dir: Path,
    row: dict[str, Any],
    *,
    policy_a: str,
    policy_b: str,
    formats: list[str],
) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    p1 = out_dir / "fig_streaming_repo_compare_safety_latency"
    xa = float(row.get("latency_p95_ms_a", 0.0) or 0.0)
    xb = float(row.get("latency_p95_ms_b", 0.0) or 0.0)
    ya = float(row.get("strict_hit_at_k_rate_a", 0.0) or 0.0)
    yb = float(row.get("strict_hit_at_k_rate_b", 0.0) or 0.0)
    plt.figure(figsize=(6.8, 4.2))
    plt.scatter([xa], [ya], marker="o", s=100, label=policy_a)
    plt.scatter([xb], [yb], marker="s", s=100, label=policy_b)
    plt.plot([xa, xb], [ya, yb], linestyle="--", linewidth=1.0)
    plt.xlabel("latency_p95_ms")
    plt.ylabel("strict_hit_at_k_rate")
    plt.title("Streaming Repo-Aware Compare")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    p2 = out_dir / "fig_streaming_repo_compare_delta"
    labels = [
        "strict_hit_at_k",
        "critical_fn_rate",
        "top1_in_distractor",
        "latency_p95_ms",
        "repo_selected_chunks",
    ]
    deltas = [
        float(row.get("delta_strict_hit_at_k_rate", 0.0) or 0.0),
        float(row.get("delta_critical_fn_rate", 0.0) or 0.0),
        float(row.get("delta_top1_in_distractor_rate", 0.0) or 0.0),
        float(row.get("delta_latency_p95_ms", 0.0) or 0.0),
        float(row.get("delta_repo_selected_chunks_mean", 0.0) or 0.0),
    ]
    plt.figure(figsize=(7.6, 4.2))
    x = list(range(len(labels)))
    plt.bar(x, deltas)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xticks(x, labels, rotation=22, ha="right")
    plt.ylabel("delta (B-A)")
    plt.title("Streaming Repo Delta Metrics")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()
    return paths


def _build_smoke_cmd(
    *,
    json_path: str,
    out_dir: Path,
    step_s: float,
    budgets: str,
    policy: str,
    mode: str,
    top_k: int,
    seed: int,
    max_trials: int,
    max_trials_per_query: int,
    latency_cap_ms: float,
    strict_threshold: float,
    max_top1_in_distractor_rate: float,
    use_repo: bool,
    repo_policy: str,
    intervention_cfg: str | None,
    queries: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "streaming_budget_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        str(float(step_s)),
        "--budgets",
        str(budgets),
        "--policy",
        str(policy),
        "--mode",
        str(mode),
        "--top-k",
        str(int(top_k)),
        "--seed",
        str(int(seed)),
        "--latency-cap-ms",
        str(float(latency_cap_ms)),
        "--max-trials-per-query",
        str(int(max_trials_per_query)),
        "--max-trials",
        str(int(max_trials)),
        "--strict-threshold",
        str(float(strict_threshold)),
        "--max-top1-in-distractor-rate",
        str(float(max_top1_in_distractor_rate)),
        "--repo-read-policy",
        str(repo_policy),
    ]
    cmd.append("--context-use-repo" if use_repo else "--no-context-use-repo")
    if intervention_cfg:
        cmd.extend(["--intervention-cfg", str(intervention_cfg)])
    for q in queries:
        cmd.extend(["--query", str(q)])
    return cmd


def _write_readme(path: Path, *, cmd_a: list[str], cmd_b: list[str], row: dict[str, Any], figures: list[str]) -> None:
    lines = [
        "# Streaming Repo Compare",
        "",
        "## Commands",
        "",
        "```text",
        _render_cmd(cmd_a),
        _render_cmd(cmd_b),
        "```",
        "",
        "## Summary",
        "",
        f"- strict_hit_at_k_rate: A={row.get('strict_hit_at_k_rate_a')} B={row.get('strict_hit_at_k_rate_b')} delta={row.get('delta_strict_hit_at_k_rate')}",
        f"- critical_fn_rate: A={row.get('critical_fn_rate_a')} B={row.get('critical_fn_rate_b')} delta={row.get('delta_critical_fn_rate')}",
        f"- top1_in_distractor_rate: A={row.get('top1_in_distractor_rate_a')} B={row.get('top1_in_distractor_rate_b')} delta={row.get('delta_top1_in_distractor_rate')}",
        f"- latency_p95_ms: A={row.get('latency_p95_ms_a')} B={row.get('latency_p95_ms_b')} delta={row.get('delta_latency_p95_ms')}",
        f"- repo_selected_chunks_mean: A={row.get('repo_selected_chunks_mean_a')} B={row.get('repo_selected_chunks_mean_b')} delta={row.get('delta_repo_selected_chunks_mean')}",
        "",
        "## Figures",
        "",
        f"- {figures}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    run_a = out_dir / "run_a"
    run_b = out_dir / "run_b"
    compare_dir = out_dir / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    logs_dir = compare_dir / "logs"
    commands_file = compare_dir / "commands.sh"

    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not commands_file.exists():
        commands_file.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    queries = [str(x).strip() for x in list(args.query or []) if str(x).strip()]
    cmd_a = _build_smoke_cmd(
        json_path=str(args.json),
        out_dir=run_a,
        step_s=float(args.step_s),
        budgets=str(args.budgets),
        policy=str(args.policy_a),
        mode=str(args.mode),
        top_k=int(args.top_k),
        seed=int(args.seed),
        max_trials=int(args.max_trials),
        max_trials_per_query=int(args.max_trials_per_query),
        latency_cap_ms=float(args.latency_cap_ms),
        strict_threshold=float(args.strict_threshold),
        max_top1_in_distractor_rate=float(args.max_top1_in_distractor_rate),
        use_repo=bool(args.a_use_repo),
        repo_policy=str(args.a_repo_policy),
        intervention_cfg=None,
        queries=queries,
    )
    cmd_b = _build_smoke_cmd(
        json_path=str(args.json),
        out_dir=run_b,
        step_s=float(args.step_s),
        budgets=str(args.budgets),
        policy=str(args.policy_b),
        mode=str(args.mode),
        top_k=int(args.top_k),
        seed=int(args.seed),
        max_trials=int(args.max_trials),
        max_trials_per_query=int(args.max_trials_per_query),
        latency_cap_ms=float(args.latency_cap_ms),
        strict_threshold=float(args.strict_threshold),
        max_top1_in_distractor_rate=float(args.max_top1_in_distractor_rate),
        use_repo=bool(args.b_use_repo),
        repo_policy=str(args.b_repo_policy),
        intervention_cfg=args.intervention_cfg,
        queries=queries,
    )

    rc_a = _run(cmd_a, cwd=ROOT, log_prefix=logs_dir / "run_a", commands_file=commands_file)
    if rc_a != 0:
        return rc_a
    rc_b = _run(cmd_b, cwd=ROOT, log_prefix=logs_dir / "run_b", commands_file=commands_file)
    if rc_b != 0:
        return rc_b

    metrics_a = _collect_metrics(run_a)
    metrics_b = _collect_metrics(run_b)
    row = {
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
        "a_use_repo": bool(args.a_use_repo),
        "b_use_repo": bool(args.b_use_repo),
        "a_repo_policy": str(args.a_repo_policy),
        "b_repo_policy": str(args.b_repo_policy),
        "strict_hit_at_k_rate_a": float(metrics_a.get("strict_hit_at_k_rate", 0.0)),
        "strict_hit_at_k_rate_b": float(metrics_b.get("strict_hit_at_k_rate", 0.0)),
        "delta_strict_hit_at_k_rate": float(metrics_b.get("strict_hit_at_k_rate", 0.0)) - float(metrics_a.get("strict_hit_at_k_rate", 0.0)),
        "mrr_a": float(metrics_a.get("mrr", 0.0)),
        "mrr_b": float(metrics_b.get("mrr", 0.0)),
        "delta_mrr": float(metrics_b.get("mrr", 0.0)) - float(metrics_a.get("mrr", 0.0)),
        "critical_fn_rate_a": float(metrics_a.get("critical_fn_rate", 0.0)),
        "critical_fn_rate_b": float(metrics_b.get("critical_fn_rate", 0.0)),
        "delta_critical_fn_rate": float(metrics_b.get("critical_fn_rate", 0.0)) - float(metrics_a.get("critical_fn_rate", 0.0)),
        "top1_in_distractor_rate_a": float(metrics_a.get("top1_in_distractor_rate", 0.0)),
        "top1_in_distractor_rate_b": float(metrics_b.get("top1_in_distractor_rate", 0.0)),
        "delta_top1_in_distractor_rate": float(metrics_b.get("top1_in_distractor_rate", 0.0))
        - float(metrics_a.get("top1_in_distractor_rate", 0.0)),
        "latency_p50_ms_a": float(metrics_a.get("latency_p50_ms", 0.0)),
        "latency_p50_ms_b": float(metrics_b.get("latency_p50_ms", 0.0)),
        "delta_latency_p50_ms": float(metrics_b.get("latency_p50_ms", 0.0)) - float(metrics_a.get("latency_p50_ms", 0.0)),
        "latency_p95_ms_a": float(metrics_a.get("latency_p95_ms", 0.0)),
        "latency_p95_ms_b": float(metrics_b.get("latency_p95_ms", 0.0)),
        "delta_latency_p95_ms": float(metrics_b.get("latency_p95_ms", 0.0)) - float(metrics_a.get("latency_p95_ms", 0.0)),
        "avg_trials_per_query_a": float(metrics_a.get("avg_trials_per_query", 0.0)),
        "avg_trials_per_query_b": float(metrics_b.get("avg_trials_per_query", 0.0)),
        "delta_avg_trials_per_query": float(metrics_b.get("avg_trials_per_query", 0.0))
        - float(metrics_a.get("avg_trials_per_query", 0.0)),
        "repo_selected_chunks_mean_a": float(metrics_a.get("repo_selected_chunks_mean", 0.0)),
        "repo_selected_chunks_mean_b": float(metrics_b.get("repo_selected_chunks_mean", 0.0)),
        "delta_repo_selected_chunks_mean": float(metrics_b.get("repo_selected_chunks_mean", 0.0))
        - float(metrics_a.get("repo_selected_chunks_mean", 0.0)),
        "context_len_chars_mean_a": float(metrics_a.get("context_len_chars_mean", 0.0)),
        "context_len_chars_mean_b": float(metrics_b.get("context_len_chars_mean", 0.0)),
        "delta_context_len_chars_mean": float(metrics_b.get("context_len_chars_mean", 0.0))
        - float(metrics_a.get("context_len_chars_mean", 0.0)),
    }

    columns = [
        "policy_a",
        "policy_b",
        "a_use_repo",
        "b_use_repo",
        "a_repo_policy",
        "b_repo_policy",
        "strict_hit_at_k_rate_a",
        "strict_hit_at_k_rate_b",
        "delta_strict_hit_at_k_rate",
        "mrr_a",
        "mrr_b",
        "delta_mrr",
        "critical_fn_rate_a",
        "critical_fn_rate_b",
        "delta_critical_fn_rate",
        "top1_in_distractor_rate_a",
        "top1_in_distractor_rate_b",
        "delta_top1_in_distractor_rate",
        "latency_p50_ms_a",
        "latency_p50_ms_b",
        "delta_latency_p50_ms",
        "latency_p95_ms_a",
        "latency_p95_ms_b",
        "delta_latency_p95_ms",
        "avg_trials_per_query_a",
        "avg_trials_per_query_b",
        "delta_avg_trials_per_query",
        "repo_selected_chunks_mean_a",
        "repo_selected_chunks_mean_b",
        "delta_repo_selected_chunks_mean",
        "context_len_chars_mean_a",
        "context_len_chars_mean_b",
        "delta_context_len_chars_mean",
    ]
    table_csv = tables_dir / "table_streaming_repo_compare.csv"
    table_md = tables_dir / "table_streaming_repo_compare.md"
    _write_csv(table_csv, [row], columns)
    _write_md(
        table_md,
        row,
        columns,
        summary_lines=[
            f"- json: `{args.json}`",
            f"- step_s: {float(args.step_s)}",
            f"- budgets: `{args.budgets}`",
            f"- policies: {args.policy_a} vs {args.policy_b}",
            f"- use_repo: A={bool(args.a_use_repo)} B={bool(args.b_use_repo)}",
            f"- repo_policy: A={args.a_repo_policy} B={args.b_repo_policy}",
            f"- queries_count: {len(queries)}",
        ],
    )

    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    figure_paths = _make_figures(
        figures_dir,
        row,
        policy_a=str(args.policy_a),
        policy_b=str(args.policy_b),
        formats=formats,
    )

    compare_summary = {
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
        "a_use_repo": bool(args.a_use_repo),
        "b_use_repo": bool(args.b_use_repo),
        "a_repo_policy": str(args.a_repo_policy),
        "b_repo_policy": str(args.b_repo_policy),
        "inputs": {
            "json": str(args.json),
            "step_s": float(args.step_s),
            "budgets": str(args.budgets),
            "queries": queries,
            "max_trials": int(args.max_trials),
            "max_trials_per_query": int(args.max_trials_per_query),
            "latency_cap_ms": float(args.latency_cap_ms),
            "strict_threshold": float(args.strict_threshold),
            "max_top1_in_distractor_rate": float(args.max_top1_in_distractor_rate),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "mode": str(args.mode),
            "intervention_cfg": str(args.intervention_cfg) if args.intervention_cfg else None,
        },
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "delta": {
            "strict_hit_at_k_rate": row["delta_strict_hit_at_k_rate"],
            "mrr": row["delta_mrr"],
            "critical_fn_rate": row["delta_critical_fn_rate"],
            "top1_in_distractor_rate": row["delta_top1_in_distractor_rate"],
            "latency_p95_ms": row["delta_latency_p95_ms"],
            "repo_selected_chunks_mean": row["delta_repo_selected_chunks_mean"],
            "context_len_chars_mean": row["delta_context_len_chars_mean"],
        },
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
        },
    }
    summary_path = compare_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "json": str(args.json),
            "step_s": float(args.step_s),
            "budgets": str(args.budgets),
            "queries": queries,
        },
        "inputs": {
            "policy_a": str(args.policy_a),
            "policy_b": str(args.policy_b),
            "a_use_repo": bool(args.a_use_repo),
            "b_use_repo": bool(args.b_use_repo),
            "a_repo_policy": str(args.a_repo_policy),
            "b_repo_policy": str(args.b_repo_policy),
            "max_trials": int(args.max_trials),
            "max_trials_per_query": int(args.max_trials_per_query),
            "latency_cap_ms": float(args.latency_cap_ms),
            "strict_threshold": float(args.strict_threshold),
            "max_top1_in_distractor_rate": float(args.max_top1_in_distractor_rate),
            "intervention_cfg": str(args.intervention_cfg) if args.intervention_cfg else None,
        },
        "runs": {
            "run_a_dir": str(run_a),
            "run_b_dir": str(run_b),
            "run_a_snapshot": str(run_a / "snapshot.json"),
            "run_b_snapshot": str(run_b / "snapshot.json"),
        },
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
            "compare_summary_json": str(summary_path),
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_readme(compare_dir / "README.md", cmd_a=cmd_a, cmd_b=cmd_b, row=row, figures=figure_paths)

    print(f"saved_a={run_a}")
    print(f"saved_b={run_b}")
    print(f"saved_compare={compare_dir}")
    print(f"budgets={len([x for x in str(args.budgets).split(',') if x.strip()])}")
    print(f"policies={args.policy_a},{args.policy_b}")
    print(f"a_use_repo={bool(args.a_use_repo)}")
    print(f"b_use_repo={bool(args.b_use_repo)}")
    print(f"b_repo_policy={args.b_repo_policy}")
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

