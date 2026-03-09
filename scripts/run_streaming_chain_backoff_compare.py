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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare streaming chain backoff strategies (strict/ladder/adaptive)")
    parser.add_argument("--json", required=True, help="Input *_v03_decisions.json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--budgets", required=True, help='Budget list like "20/50/4,60/200/12"')
    parser.add_argument("--step-s", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--budget-policy", default="safety_latency_chain")
    parser.add_argument("--policies", default="strict,ladder,adaptive", help="Comma list of chain backoff strategies")
    parser.add_argument("--query", action="append", default=[], help="Repeatable query text")
    parser.add_argument("--latency-cap-ms", type=float, default=25.0)
    parser.add_argument("--max-trials-per-query", type=int, default=3)
    parser.add_argument("--strict-threshold", type=float, default=1.0)
    parser.add_argument("--max-top1-in-distractor-rate", type=float, default=0.2)
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def _render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _append_command(path: Path, cmd: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"# {datetime.now(timezone.utc).isoformat()}\n")
        f.write(_render_cmd(cmd) + "\n\n")


def _run_cmd(cmd: list[str], *, cwd: Path, log_prefix: Path, commands_path: Path) -> int:
    _append_command(commands_path, cmd)
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    (log_prefix.with_suffix(".stdout.log")).write_text(proc.stdout or "", encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
    return int(proc.returncode)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _final_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    finals = [r for r in rows if int(float(_to_float(r.get("final_trial")) or 0.0)) == 1]
    return finals if finals else rows


def _collect_metrics_by_budget(run_dir: Path) -> dict[float, dict[str, float]]:
    rows = _final_rows(_read_csv(run_dir / "queries.csv"))
    chain_rows = [r for r in rows if int(float(_to_float(r.get("is_chain")) or 0.0)) > 0]
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in chain_rows:
        sec = _to_float(row.get("chain_step2_budget_seconds"))
        if sec is None:
            sec = _to_float(row.get("budget_seconds"))
        bsec = float(sec or 0.0)
        grouped.setdefault(bsec, []).append(row)

    out: dict[float, dict[str, float]] = {}
    for sec, items in grouped.items():
        lat = [float(_to_float(x.get("latency_e2e_ms")) or 0.0) for x in items]
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
        p95_idx = int(max(0, round((len(lat_sorted) - 1) * 0.95))) if lat_sorted else 0
        p95 = lat_sorted[p95_idx] if lat_sorted else 0.0
        n = max(1, len(items))
        out[sec] = {
            "queries": float(len(items)),
            "chain_success_rate": float(sum(float(_to_float(x.get("chain_success")) or 0.0) for x in items) / n),
            "chain_waiting_rate": float(sum(float(_to_float(x.get("chain_waiting")) or 0.0) for x in items) / n),
            "chain_backoff_used_rate": float(sum(float(_to_float(x.get("chain_backoff_used")) or 0.0) for x in items) / n),
            "chain_backoff_mean_level": float(sum(float(_to_float(x.get("chain_backoff_level")) or 0.0) for x in items) / n),
            "chain_backoff_exhausted_rate": float(
                sum(float(_to_float(x.get("chain_backoff_exhausted")) or 0.0) for x in items) / n
            ),
            "latency_p50_ms": float(p50),
            "latency_p95_ms": float(p95),
            "safety_critical_fn_rate": float(
                sum(float(_to_float(x.get("safety_is_critical_fn")) or 0.0) for x in items) / n
            ),
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], summary: list[str]) -> None:
    lines = [
        "# Streaming Chain Backoff Compare",
        "",
        *summary,
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(
    *,
    by_strategy: dict[str, dict[float, dict[str, float]]],
    out_dir: Path,
    formats: list[str],
) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    strategies = list(by_strategy.keys())
    budgets = sorted({b for m in by_strategy.values() for b in m.keys()})
    if not budgets:
        return paths

    # success vs budget
    p1 = out_dir / "fig_streaming_chain_backoff_success_vs_budget_seconds"
    plt.figure(figsize=(8.2, 4.4))
    for s in strategies:
        ys = [float(by_strategy.get(s, {}).get(b, {}).get("chain_success_rate", 0.0)) for b in budgets]
        plt.plot(budgets, ys, marker="o", label=s)
    plt.xlabel("budget_seconds")
    plt.ylabel("chain_success_rate")
    plt.title("Streaming Chain Backoff: Success vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # latency vs budget
    p2 = out_dir / "fig_streaming_chain_backoff_latency_vs_budget_seconds"
    plt.figure(figsize=(8.2, 4.4))
    for s in strategies:
        ys = [float(by_strategy.get(s, {}).get(b, {}).get("latency_p95_ms", 0.0)) for b in budgets]
        plt.plot(budgets, ys, marker="o", label=f"{s}_p95")
    plt.xlabel("budget_seconds")
    plt.ylabel("latency_p95_ms")
    plt.title("Streaming Chain Backoff: Latency vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # backoff level/used vs budget
    p3 = out_dir / "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds"
    plt.figure(figsize=(8.2, 4.4))
    ax0 = plt.gca()
    ax1 = ax0.twinx()
    for s in strategies:
        lvl = [float(by_strategy.get(s, {}).get(b, {}).get("chain_backoff_mean_level", 0.0)) for b in budgets]
        used = [float(by_strategy.get(s, {}).get(b, {}).get("chain_backoff_used_rate", 0.0)) for b in budgets]
        ax0.plot(budgets, lvl, marker="o", label=f"{s}_mean_level")
        ax1.plot(budgets, used, marker="s", linestyle="--", label=f"{s}_used_rate")
    ax0.set_xlabel("budget_seconds")
    ax0.set_ylabel("chain_backoff_mean_level")
    ax1.set_ylabel("chain_backoff_used_rate")
    ax0.grid(True, alpha=0.35)
    l0, t0 = ax0.get_legend_handles_labels()
    l1, t1 = ax1.get_legend_handles_labels()
    ax0.legend(l0 + l1, t0 + t1, loc="upper right", fontsize=8)
    plt.title("Streaming Chain Backoff: Mean Level and Used Rate")
    plt.tight_layout()
    for ext in formats:
        p = p3.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # delta (vs first strategy)
    base = strategies[0]
    p4 = out_dir / "fig_streaming_chain_backoff_delta"
    labels: list[str] = []
    deltas: list[float] = []
    for s in strategies[1:]:
        base_vals = by_strategy.get(base, {})
        cur_vals = by_strategy.get(s, {})
        common = sorted(set(base_vals.keys()) & set(cur_vals.keys()))
        if not common:
            continue
        d_success = sum(cur_vals[b].get("chain_success_rate", 0.0) - base_vals[b].get("chain_success_rate", 0.0) for b in common) / len(common)
        d_latency = sum(cur_vals[b].get("latency_p95_ms", 0.0) - base_vals[b].get("latency_p95_ms", 0.0) for b in common) / len(common)
        d_backoff = sum(
            cur_vals[b].get("chain_backoff_mean_level", 0.0) - base_vals[b].get("chain_backoff_mean_level", 0.0) for b in common
        ) / len(common)
        labels.extend([f"{s}:success", f"{s}:latency", f"{s}:backoff"])
        deltas.extend([float(d_success), float(d_latency), float(d_backoff)])
    plt.figure(figsize=(9.0, 4.6))
    if labels:
        x = list(range(len(labels)))
        plt.bar(x, deltas)
        plt.axhline(y=0.0, linewidth=1.0)
        plt.xticks(x, labels, rotation=25, ha="right")
    else:
        plt.text(0.5, 0.5, "no matched budgets for delta", ha="center", va="center")
    plt.ylabel(f"delta vs {base}")
    plt.title("Streaming Chain Backoff Delta")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        p = p4.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    return paths


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    compare_dir = out_dir / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    logs_dir = compare_dir / "logs"
    commands_path = compare_dir / "commands.sh"
    compare_dir.mkdir(parents=True, exist_ok=True)
    if not commands_path.exists():
        commands_path.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    requested_policies = [str(x).strip().lower() for x in str(args.policies).split(",") if str(x).strip()]
    policies = [p for p in requested_policies if p in {"strict", "ladder", "adaptive"}]
    if not policies:
        print("error=no valid policies")
        return 2
    labels = ["A", "B", "C", "D", "E"]
    policy_runs: list[tuple[str, str, Path]] = []
    for idx, policy in enumerate(policies):
        code = labels[idx] if idx < len(labels) else f"P{idx+1}"
        run_dir = out_dir / f"run_{code}"
        run_dir.mkdir(parents=True, exist_ok=True)
        policy_runs.append((code, policy, run_dir))

    queries = [str(x).strip() for x in list(args.query or []) if str(x).strip()]
    if not queries:
        queries = ["lost_object=door which=last top_k=6 then token=SCENE_CHANGE which=last top_k=6 chain_derive=time+object chain_object_mode=hard"]

    for code, strategy, run_dir in policy_runs:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "streaming_budget_smoke.py"),
            "--json",
            str(args.json),
            "--out_dir",
            str(run_dir),
            "--step-s",
            str(float(args.step_s)),
            "--budgets",
            str(args.budgets),
            "--policy",
            str(args.budget_policy),
            "--chain-backoff-strategy",
            str(strategy),
            "--chain-backoff-seed",
            str(int(args.seed)),
            "--top-k",
            str(int(args.top_k)),
            "--seed",
            str(int(args.seed)),
            "--mode",
            str(args.mode),
            "--latency-cap-ms",
            str(float(args.latency_cap_ms)),
            "--max-trials-per-query",
            str(int(args.max_trials_per_query)),
            "--strict-threshold",
            str(float(args.strict_threshold)),
            "--max-top1-in-distractor-rate",
            str(float(args.max_top1_in_distractor_rate)),
        ]
        for q in queries:
            cmd.extend(["--query", str(q)])
        rc = _run_cmd(cmd, cwd=ROOT, log_prefix=logs_dir / f"run_{code}", commands_path=commands_path)
        if rc != 0:
            return rc

    by_strategy: dict[str, dict[float, dict[str, float]]] = {}
    for _, strategy, run_dir in policy_runs:
        by_strategy[strategy] = _collect_metrics_by_budget(run_dir)

    budgets = sorted({b for rows in by_strategy.values() for b in rows.keys()})
    rows: list[dict[str, Any]] = []
    for strategy, budget_map in by_strategy.items():
        base_map = by_strategy.get(policies[0], {})
        for sec in budgets:
            current = budget_map.get(sec, {})
            baseline = base_map.get(sec, {})
            row = {
                "strategy": str(strategy),
                "budget_seconds": float(sec),
                "chain_success_rate": float(current.get("chain_success_rate", 0.0)),
                "chain_waiting_rate": float(current.get("chain_waiting_rate", 0.0)),
                "chain_backoff_used_rate": float(current.get("chain_backoff_used_rate", 0.0)),
                "chain_backoff_mean_level": float(current.get("chain_backoff_mean_level", 0.0)),
                "chain_backoff_exhausted_rate": float(current.get("chain_backoff_exhausted_rate", 0.0)),
                "latency_p50_ms": float(current.get("latency_p50_ms", 0.0)),
                "latency_p95_ms": float(current.get("latency_p95_ms", 0.0)),
                "safety_critical_fn_rate": float(current.get("safety_critical_fn_rate", 0.0)),
                "queries": int(current.get("queries", 0.0)),
                "delta_chain_success_rate_vs_base": float(current.get("chain_success_rate", 0.0) - baseline.get("chain_success_rate", 0.0)),
                "delta_chain_backoff_mean_level_vs_base": float(
                    current.get("chain_backoff_mean_level", 0.0) - baseline.get("chain_backoff_mean_level", 0.0)
                ),
                "delta_latency_p95_ms_vs_base": float(current.get("latency_p95_ms", 0.0) - baseline.get("latency_p95_ms", 0.0)),
            }
            rows.append(row)

    columns = [
        "strategy",
        "budget_seconds",
        "chain_success_rate",
        "chain_waiting_rate",
        "chain_backoff_used_rate",
        "chain_backoff_mean_level",
        "chain_backoff_exhausted_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "safety_critical_fn_rate",
        "queries",
        "delta_chain_success_rate_vs_base",
        "delta_chain_backoff_mean_level_vs_base",
        "delta_latency_p95_ms_vs_base",
    ]
    table_csv = tables_dir / "table_streaming_chain_backoff_compare.csv"
    table_md = tables_dir / "table_streaming_chain_backoff_compare.md"
    _write_csv(table_csv, rows, columns)
    _write_md(
        table_md,
        rows,
        columns,
        summary=[
            f"- base_strategy: {policies[0]}",
            f"- policies: {policies}",
            f"- budgets: {args.budgets}",
            f"- queries_count: {len(queries)}",
        ],
    )

    figure_paths = _make_figures(by_strategy=by_strategy, out_dir=figures_dir, formats=[x.strip() for x in str(args.formats).split(",") if x.strip()])

    budgets_matched = sorted(
        set.intersection(*(set(v.keys()) for v in by_strategy.values())) if by_strategy and all(by_strategy.values()) else set()
    )
    compare_summary = {
        "policies": policies,
        "base_strategy": policies[0],
        "budgets_total": len(budgets),
        "budgets_matched": len(budgets_matched),
        "budgets_matched_values": [float(x) for x in budgets_matched],
        "by_strategy": by_strategy,
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
        "inputs": {
            "json": str(args.json),
            "budgets": str(args.budgets),
            "step_s": float(args.step_s),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "mode": str(args.mode),
            "budget_policy": str(args.budget_policy),
            "policies": policies,
            "queries": queries,
        },
        "runs": {
            code: {"strategy": strategy, "run_dir": str(run_dir), "snapshot": str(run_dir / "snapshot.json")}
            for code, strategy, run_dir in policy_runs
        },
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
            "compare_summary": str(compare_summary_path),
            "commands_sh": str(commands_path),
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = compare_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Streaming Chain Backoff Compare",
                "",
                f"- policies: `{policies}`",
                f"- base_strategy: `{policies[0]}`",
                f"- table: `{table_csv}`",
                f"- figures: `{figure_paths}`",
                f"- summary: `{compare_summary_path}`",
                f"- snapshot: `{snapshot_path}`",
                f"- commands: `{commands_path}`",
            ]
        ),
        encoding="utf-8",
    )

    for code, _, run_dir in policy_runs:
        print(f"saved_run_{code}={run_dir}")
    print(f"saved_compare={compare_dir}")
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

