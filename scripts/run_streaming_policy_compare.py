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


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming baseline vs intervention policy compare harness")
    parser.add_argument("--json", required=True, help="Input *_v03_decisions.json")
    parser.add_argument("--out_dir", required=True, help="Output root")
    parser.add_argument("--step-s", type=float, default=8.0)
    parser.add_argument("--budgets", required=True, help='Budget list like "20/50/4,60/200/12,120/400/24"')
    parser.add_argument("--query", action="append", default=[], help="Repeatable query")

    parser.add_argument("--policy-a", default="safety_latency")
    parser.add_argument("--policy-b", default="safety_latency_intervention")
    parser.add_argument("--intervention-cfg", default=None, help="YAML config for intervention policy (run_b)")
    parser.add_argument("--max-trials", type=int, default=5)
    parser.add_argument("--latency-cap-ms", type=float, default=25.0)
    parser.add_argument("--max-trials-per-query", type=int, default=3)
    parser.add_argument("--strict-threshold", type=float, default=1.0)
    parser.add_argument("--max-top1-in-distractor-rate", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--formats", default="png,pdf")
    _parse_bool_with_neg(parser, "prefer-lower-budget", default=True)
    parser.add_argument("--recommend-dir", default=None)
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


def _load_summary(run_dir: Path) -> dict[str, Any]:
    snap = run_dir / "snapshot.json"
    if snap.exists():
        try:
            payload = json.loads(snap.read_text(encoding="utf-8"))
            summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
            if isinstance(summary, dict):
                return dict(summary)
        except Exception:
            pass
    return {}


def _collect_metrics(run_dir: Path) -> dict[str, float | str | int]:
    summary = _load_summary(run_dir)
    qrows = _read_csv(run_dir / "queries.csv")
    finals = _final_query_rows(qrows)

    strict_vals: list[float] = []
    critical_vals: list[float] = []
    latency_vals: list[float] = []
    trials_vals: list[float] = []
    chain_rows: list[dict[str, str]] = []
    for r in finals:
        final_success = _to_float(r.get("final_success"))
        strict_hit = _to_float(r.get("strict_hit_at_k"))
        if final_success is not None:
            strict_vals.append(1.0 if final_success > 0.0 else 0.0)
        elif strict_hit is not None:
            strict_vals.append(1.0 if strict_hit > 0.0 else 0.0)
        critical_vals.append(float(_to_float(r.get("safety_is_critical_fn")) or 0.0))
        latency_vals.append(float(_to_float(r.get("latency_e2e_ms")) or _to_float(r.get("e2e_ms")) or 0.0))
        trials_vals.append(float(_to_float(r.get("trial_count_for_query")) or _to_float(r.get("trials_count")) or 1.0))
        if int(float(_to_float(r.get("is_chain")) or 0.0)) > 0:
            chain_rows.append(r)

    strict_success_rate = float(sum(strict_vals) / len(strict_vals)) if strict_vals else float(_to_float(summary.get("hit_at_k_strict")) or 0.0)
    critical_fn_rate = float(sum(critical_vals) / len(critical_vals)) if critical_vals else float(_to_float(summary.get("safety_critical_fn_rate")) or 0.0)
    latency_p95 = float(_percentile(latency_vals, 95.0)) if latency_vals else float(_to_float(summary.get("e2e_latency_p95_ms")) or 0.0)
    avg_trials = float(sum(trials_vals) / len(trials_vals)) if trials_vals else float(_to_float(summary.get("avg_trials_per_query")) or 0.0)
    chain_success_rate = (
        float(sum(float(_to_float(r.get("chain_success")) or 0.0) for r in chain_rows) / len(chain_rows))
        if chain_rows
        else 0.0
    )
    chain_waiting_rate = (
        float(sum(float(_to_float(r.get("chain_waiting")) or 0.0) for r in chain_rows) / len(chain_rows))
        if chain_rows
        else 0.0
    )
    chain_distractor_rate = (
        float(sum(float(_to_float(r.get("top1_in_distractor_rate")) or 0.0) for r in chain_rows) / len(chain_rows))
        if chain_rows
        else 0.0
    )
    chain_constraints_over_filtered_rate = (
        float(
            sum(
                1.0 if str(r.get("chain_fail_reason", "")).strip() == "constraints_over_filtered" else 0.0
                for r in chain_rows
            )
            / len(chain_rows)
        )
        if chain_rows
        else 0.0
    )

    return {
        "strict_success_rate": float(strict_success_rate),
        "critical_fn_rate": float(critical_fn_rate),
        "latency_p95_e2e_ms": float(latency_p95),
        "avg_trials_per_query": float(avg_trials),
        "queries_total": int(len(finals)) if finals else int(_to_float(summary.get("queries_total")) or 0),
        "chain_queries_total": int(len(chain_rows)),
        "chain_success_rate": float(chain_success_rate),
        "chain_waiting_rate": float(chain_waiting_rate),
        "chain_distractor_rate": float(chain_distractor_rate),
        "chain_constraints_over_filtered_rate": float(chain_constraints_over_filtered_rate),
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
        "# Streaming Policy Compare",
        "",
        *summary_lines,
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
        "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(out_dir: Path, row: dict[str, Any], policy_a: str, policy_b: str, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # safety-latency scatter
    p1 = out_dir / "fig_streaming_policy_compare_safety_latency"
    xa = float(row.get("latency_p95_e2e_ms_a", 0.0) or 0.0)
    xb = float(row.get("latency_p95_e2e_ms_b", 0.0) or 0.0)
    ya = float(row.get("strict_success_rate_a", 0.0) or 0.0)
    yb = float(row.get("strict_success_rate_b", 0.0) or 0.0)
    plt.figure(figsize=(6.8, 4.2))
    plt.scatter([xa], [ya], marker="o", s=100, label=policy_a)
    plt.scatter([xb], [yb], marker="s", s=100, label=policy_b)
    plt.plot([xa, xb], [ya, yb], linestyle="--", linewidth=1.0)
    plt.xlabel("latency_p95_e2e_ms")
    plt.ylabel("strict_success_rate")
    plt.title("Streaming Policy Safety-Latency Compare")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # delta bar
    p2 = out_dir / "fig_streaming_policy_compare_delta"
    labels = [
        "strict_success_rate",
        "critical_fn_rate",
        "latency_p95_e2e_ms",
        "avg_trials_per_query",
    ]
    deltas = [
        float(row.get("delta_strict_success_rate", 0.0) or 0.0),
        float(row.get("delta_critical_fn_rate", 0.0) or 0.0),
        float(row.get("delta_latency_p95_e2e_ms", 0.0) or 0.0),
        float(row.get("delta_avg_trials_per_query", 0.0) or 0.0),
    ]
    plt.figure(figsize=(7.4, 4.2))
    x = list(range(len(labels)))
    plt.bar(x, deltas)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel(f"delta ({policy_b}-{policy_a})")
    plt.title("Streaming Policy Delta Metrics")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # chain success compare
    p3 = out_dir / "fig_streaming_policy_compare_chain_success"
    chain_labels = ["chain_success_rate", "chain_waiting_rate"]
    avals = [
        float(row.get("chain_success_rate_a", 0.0) or 0.0),
        float(row.get("chain_waiting_rate_a", 0.0) or 0.0),
    ]
    bvals = [
        float(row.get("chain_success_rate_b", 0.0) or 0.0),
        float(row.get("chain_waiting_rate_b", 0.0) or 0.0),
    ]
    x = list(range(len(chain_labels)))
    width = 0.35
    plt.figure(figsize=(7.4, 4.2))
    plt.bar([i - width / 2 for i in x], avals, width=width, label=policy_a)
    plt.bar([i + width / 2 for i in x], bvals, width=width, label=policy_b)
    plt.xticks(x, chain_labels, rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Streaming Policy Chain Success Compare")
    plt.grid(True, axis="y", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p3.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # chain delta bar
    p4 = out_dir / "fig_streaming_policy_compare_chain_delta"
    chain_delta_labels = [
        "chain_success_rate",
        "chain_waiting_rate",
        "chain_distractor_rate",
        "chain_constraints_over_filtered_rate",
    ]
    chain_deltas = [
        float(row.get("delta_chain_success_rate", 0.0) or 0.0),
        float(row.get("delta_chain_waiting_rate", 0.0) or 0.0),
        float(row.get("delta_chain_distractor_rate", 0.0) or 0.0),
        float(row.get("delta_chain_constraints_over_filtered_rate", 0.0) or 0.0),
    ]
    plt.figure(figsize=(7.4, 4.2))
    xc = list(range(len(chain_delta_labels)))
    plt.bar(xc, chain_deltas)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xticks(xc, chain_delta_labels, rotation=20, ha="right")
    plt.ylabel(f"delta ({policy_b}-{policy_a})")
    plt.title("Streaming Policy Chain Delta Metrics")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        p = p4.with_suffix(f".{ext}")
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
    recommend_dir: str | None,
    intervention_cfg: str | None,
    max_trials: int,
    max_trials_per_query: int,
    latency_cap_ms: float,
    strict_threshold: float,
    max_top1_in_distractor_rate: float,
    prefer_lower_budget: bool,
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
    ]
    if recommend_dir:
        cmd.extend(["--recommend-dir", str(recommend_dir)])
    if intervention_cfg:
        cmd.extend(["--intervention-cfg", str(intervention_cfg)])
    cmd.append("--prefer-lower-budget" if prefer_lower_budget else "--prefer-higher-budget")
    for q in queries:
        cmd.extend(["--query", str(q)])
    return cmd


def _write_readme(path: Path, *, cmd_a: list[str], cmd_b: list[str], row: dict[str, Any], figure_paths: list[str]) -> None:
    lines = [
        "# Streaming Policy Compare",
        "",
        "## Commands",
        "",
        "```text",
        _render_cmd(cmd_a),
        _render_cmd(cmd_b),
        "```",
        "",
        "## Metrics",
        "",
        f"- strict_success_rate: A={row.get('strict_success_rate_a')} B={row.get('strict_success_rate_b')} delta={row.get('delta_strict_success_rate')}",
        f"- critical_fn_rate: A={row.get('critical_fn_rate_a')} B={row.get('critical_fn_rate_b')} delta={row.get('delta_critical_fn_rate')}",
        f"- latency_p95_e2e_ms: A={row.get('latency_p95_e2e_ms_a')} B={row.get('latency_p95_e2e_ms_b')} delta={row.get('delta_latency_p95_e2e_ms')}",
        f"- avg_trials_per_query: A={row.get('avg_trials_per_query_a')} B={row.get('avg_trials_per_query_b')} delta={row.get('delta_avg_trials_per_query')}",
        f"- chain_success_rate: A={row.get('chain_success_rate_a')} B={row.get('chain_success_rate_b')} delta={row.get('delta_chain_success_rate')}",
        f"- chain_waiting_rate: A={row.get('chain_waiting_rate_a')} B={row.get('chain_waiting_rate_b')} delta={row.get('delta_chain_waiting_rate')}",
        f"- chain_distractor_rate: A={row.get('chain_distractor_rate_a')} B={row.get('chain_distractor_rate_b')} delta={row.get('delta_chain_distractor_rate')}",
        "",
        "## Figures",
        "",
        f"- {figure_paths}",
        "",
        "Y-axis for safety-latency figure is strict_success_rate; X-axis is latency_p95_e2e_ms.",
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
        recommend_dir=args.recommend_dir,
        intervention_cfg=None,
        max_trials=int(args.max_trials),
        max_trials_per_query=int(args.max_trials_per_query),
        latency_cap_ms=float(args.latency_cap_ms),
        strict_threshold=float(args.strict_threshold),
        max_top1_in_distractor_rate=float(args.max_top1_in_distractor_rate),
        prefer_lower_budget=bool(args.prefer_lower_budget),
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
        recommend_dir=args.recommend_dir,
        intervention_cfg=args.intervention_cfg,
        max_trials=int(args.max_trials),
        max_trials_per_query=int(args.max_trials_per_query),
        latency_cap_ms=float(args.latency_cap_ms),
        strict_threshold=float(args.strict_threshold),
        max_top1_in_distractor_rate=float(args.max_top1_in_distractor_rate),
        prefer_lower_budget=bool(args.prefer_lower_budget),
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
    run_b_snapshot_payload: dict[str, Any] = {}
    run_b_snapshot = run_b / "snapshot.json"
    if run_b_snapshot.exists():
        try:
            run_b_snapshot_payload = json.loads(run_b_snapshot.read_text(encoding="utf-8"))
        except Exception:
            run_b_snapshot_payload = {}
    run_b_summary = dict(run_b_snapshot_payload.get("summary", {})) if isinstance(run_b_snapshot_payload, dict) else {}
    intervention_cfg_name = str(run_b_summary.get("intervention_cfg_name", ""))
    intervention_cfg_hash = str(run_b_summary.get("intervention_cfg_hash", ""))

    row = {
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
        "strict_success_rate_a": float(metrics_a.get("strict_success_rate", 0.0)),
        "strict_success_rate_b": float(metrics_b.get("strict_success_rate", 0.0)),
        "delta_strict_success_rate": float(metrics_b.get("strict_success_rate", 0.0)) - float(metrics_a.get("strict_success_rate", 0.0)),
        "critical_fn_rate_a": float(metrics_a.get("critical_fn_rate", 0.0)),
        "critical_fn_rate_b": float(metrics_b.get("critical_fn_rate", 0.0)),
        "delta_critical_fn_rate": float(metrics_b.get("critical_fn_rate", 0.0)) - float(metrics_a.get("critical_fn_rate", 0.0)),
        "latency_p95_e2e_ms_a": float(metrics_a.get("latency_p95_e2e_ms", 0.0)),
        "latency_p95_e2e_ms_b": float(metrics_b.get("latency_p95_e2e_ms", 0.0)),
        "delta_latency_p95_e2e_ms": float(metrics_b.get("latency_p95_e2e_ms", 0.0)) - float(metrics_a.get("latency_p95_e2e_ms", 0.0)),
        "avg_trials_per_query_a": float(metrics_a.get("avg_trials_per_query", 0.0)),
        "avg_trials_per_query_b": float(metrics_b.get("avg_trials_per_query", 0.0)),
        "delta_avg_trials_per_query": float(metrics_b.get("avg_trials_per_query", 0.0)) - float(metrics_a.get("avg_trials_per_query", 0.0)),
        "chain_success_rate_a": float(metrics_a.get("chain_success_rate", 0.0)),
        "chain_success_rate_b": float(metrics_b.get("chain_success_rate", 0.0)),
        "delta_chain_success_rate": float(metrics_b.get("chain_success_rate", 0.0)) - float(metrics_a.get("chain_success_rate", 0.0)),
        "chain_waiting_rate_a": float(metrics_a.get("chain_waiting_rate", 0.0)),
        "chain_waiting_rate_b": float(metrics_b.get("chain_waiting_rate", 0.0)),
        "delta_chain_waiting_rate": float(metrics_b.get("chain_waiting_rate", 0.0)) - float(metrics_a.get("chain_waiting_rate", 0.0)),
        "chain_distractor_rate_a": float(metrics_a.get("chain_distractor_rate", 0.0)),
        "chain_distractor_rate_b": float(metrics_b.get("chain_distractor_rate", 0.0)),
        "delta_chain_distractor_rate": float(metrics_b.get("chain_distractor_rate", 0.0)) - float(metrics_a.get("chain_distractor_rate", 0.0)),
        "chain_constraints_over_filtered_rate_a": float(metrics_a.get("chain_constraints_over_filtered_rate", 0.0)),
        "chain_constraints_over_filtered_rate_b": float(metrics_b.get("chain_constraints_over_filtered_rate", 0.0)),
        "delta_chain_constraints_over_filtered_rate": float(metrics_b.get("chain_constraints_over_filtered_rate", 0.0))
        - float(metrics_a.get("chain_constraints_over_filtered_rate", 0.0)),
    }
    columns = [
        "policy_a",
        "policy_b",
        "strict_success_rate_a",
        "strict_success_rate_b",
        "delta_strict_success_rate",
        "critical_fn_rate_a",
        "critical_fn_rate_b",
        "delta_critical_fn_rate",
        "latency_p95_e2e_ms_a",
        "latency_p95_e2e_ms_b",
        "delta_latency_p95_e2e_ms",
        "avg_trials_per_query_a",
        "avg_trials_per_query_b",
        "delta_avg_trials_per_query",
        "chain_success_rate_a",
        "chain_success_rate_b",
        "delta_chain_success_rate",
        "chain_waiting_rate_a",
        "chain_waiting_rate_b",
        "delta_chain_waiting_rate",
        "chain_distractor_rate_a",
        "chain_distractor_rate_b",
        "delta_chain_distractor_rate",
        "chain_constraints_over_filtered_rate_a",
        "chain_constraints_over_filtered_rate_b",
        "delta_chain_constraints_over_filtered_rate",
    ]

    table_csv = tables_dir / "table_streaming_policy_compare.csv"
    table_md = tables_dir / "table_streaming_policy_compare.md"
    _write_csv(table_csv, [row], columns)
    summary_lines = [
        f"- policies: {args.policy_a} vs {args.policy_b}",
        f"- json: `{args.json}`",
        f"- step_s: {float(args.step_s)}",
        f"- budgets: `{args.budgets}`",
        f"- queries_count: {len(queries)}",
        f"- max_trials: {int(args.max_trials)}",
    ]
    _write_md(table_md, row, columns, summary_lines)

    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    figure_paths = _make_figures(figures_dir, row, str(args.policy_a), str(args.policy_b), formats)

    compare_summary = {
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
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
            "intervention_cfg_name": intervention_cfg_name,
            "intervention_cfg_hash": intervention_cfg_hash,
        },
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "delta": {
            "strict_success_rate": row["delta_strict_success_rate"],
            "critical_fn_rate": row["delta_critical_fn_rate"],
            "latency_p95_e2e_ms": row["delta_latency_p95_e2e_ms"],
            "avg_trials_per_query": row["delta_avg_trials_per_query"],
            "chain_success_rate": row["delta_chain_success_rate"],
            "chain_waiting_rate": row["delta_chain_waiting_rate"],
            "chain_distractor_rate": row["delta_chain_distractor_rate"],
            "chain_constraints_over_filtered_rate": row["delta_chain_constraints_over_filtered_rate"],
        },
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
        },
    }
    compare_summary_path = compare_dir / "compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")

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
            "max_trials": int(args.max_trials),
            "max_trials_per_query": int(args.max_trials_per_query),
            "latency_cap_ms": float(args.latency_cap_ms),
            "strict_threshold": float(args.strict_threshold),
            "max_top1_in_distractor_rate": float(args.max_top1_in_distractor_rate),
            "prefer_lower_budget": bool(args.prefer_lower_budget),
            "recommend_dir": str(args.recommend_dir) if args.recommend_dir else None,
            "intervention_cfg": str(args.intervention_cfg) if args.intervention_cfg else None,
            "intervention_cfg_name": intervention_cfg_name,
            "intervention_cfg_hash": intervention_cfg_hash,
        },
        "runs": {
            "run_a_snapshot": str(run_a / "snapshot.json"),
            "run_b_snapshot": str(run_b / "snapshot.json"),
            "run_a_dir": str(run_a),
            "run_b_dir": str(run_b),
        },
        "outputs": {
            "compare_summary_json": str(compare_summary_path),
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_readme(compare_dir / "README.md", cmd_a=cmd_a, cmd_b=cmd_b, row=row, figure_paths=figure_paths)

    print(f"saved_a={run_a}")
    print(f"saved_b={run_b}")
    print(f"saved_compare={compare_dir}")
    print(f"policies={args.policy_a},{args.policy_b}")
    print(f"budgets={len([x for x in str(args.budgets).split(',') if x.strip()])}")
    print(f"intervention_cfg_name={intervention_cfg_name}")
    print(f"intervention_cfg_hash={intervention_cfg_hash}")
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
