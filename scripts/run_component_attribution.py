from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Component Attribution Panel (Repo / Perception / Streaming)")
    parser.add_argument("--root", required=True, help="Ego root path")
    parser.add_argument("--uids-file", required=True, help="UID list file (strictly used across all settings)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--budgets", required=True, help='e.g. "20/50/4,60/200/12"')
    parser.add_argument("--queries", action="append", default=[], help="Repeatable query")
    parser.add_argument("--query", action="append", default=[], help="Repeatable query (alias)")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-uids", type=int, default=None)

    _parse_bool_with_neg(parser, "with-nlq", default=True)
    parser.add_argument("--nlq-mode", default="hard_pseudo_nlq")
    parser.add_argument("--nlq-eval-script", default=None, help="Optional fake eval_nlq path for smoke tests")

    _parse_bool_with_neg(parser, "with-perception", default=True)
    parser.add_argument("--perception-fps", type=float, default=5.0)
    parser.add_argument("--perception-max-frames", type=int, default=300)
    parser.add_argument("--stub-perception-backend", choices=["stub", "real"], default="stub")
    parser.add_argument("--real-perception-backend", choices=["stub", "real"], default="real")
    _parse_bool_with_neg(parser, "real-perception-strict", default=True)

    _parse_bool_with_neg(parser, "with-streaming-budget", default=True)
    parser.add_argument("--streaming-step-s", type=float, default=8.0)
    parser.add_argument("--repo-read-policy", default="query_aware")
    parser.add_argument("--repo-budget", default=None)

    parser.add_argument("--min-size-bytes", type=int, default=0)
    parser.add_argument("--probe-candidates", type=int, default=0)
    parser.add_argument("--no-proxy", action="store_true", default=True)
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


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _read_uids(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "").strip()
        if not text:
            continue
        if text.startswith("#"):
            continue
        if "#" in text:
            text = text.split("#", 1)[0].strip()
        if text:
            if text.lower().endswith(".mp4"):
                text = text[:-4]
            rows.append(text)
    return rows


def _budget_keys(budgets_text: str) -> list[str]:
    return [x.strip() for x in str(budgets_text).split(",") if x.strip()]


@dataclass
class Setting:
    code: str
    name: str
    use_repo: bool
    perception_backend: str
    perception_strict: bool


def _build_ego_cmd(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    uids_file: Path,
    setting: Setting,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ego4d_smoke.py"),
        "--root",
        str(args.root),
        "--out_dir",
        str(out_dir),
        "--uids-file",
        str(uids_file),
        "--jobs",
        str(int(args.jobs)),
        "--min-size-bytes",
        str(int(args.min_size_bytes)),
        "--probe-candidates",
        str(int(args.probe_candidates)),
        "--no-proxy",
        "--resume",
        "--run-eval" if bool(args.with_nlq) else "--no-run-eval",
        "--no-run-nlq",
    ]
    if bool(args.with_perception):
        cmd.extend(
            [
                "--run-perception",
                "--perception-backend",
                str(setting.perception_backend),
                "--perception-fps",
                str(float(args.perception_fps)),
                "--perception-max-frames",
                str(int(args.perception_max_frames)),
            ]
        )
        if setting.perception_strict:
            cmd.append("--perception-strict")
    else:
        cmd.append("--no-run-perception")
    return cmd


def _build_nlq_sweep_cmd(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    uids_file: Path,
    out_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_nlq_budgets.py"),
        "--json_dir",
        str(run_dir / "json"),
        "--index_dir",
        str(run_dir / "cache"),
        "--uids-file",
        str(uids_file),
        "--strict-uids",
        "--out_dir",
        str(out_dir),
        "--budgets",
        str(args.budgets),
        "--mode",
        str(args.nlq_mode),
        "--seed",
        "0",
        "--top-k",
        str(int(args.top_k)),
        "--no-allow-gt-fallback",
        "--hard-constraints",
        "--safety-report",
    ]
    if args.nlq_eval_script:
        cmd.extend(["--eval-script", str(args.nlq_eval_script)])
    return cmd


def _build_streaming_sweep_cmd(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    uids_file: Path,
    out_dir: Path,
    setting: Setting,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_streaming_budgets.py"),
        "--json_dir",
        str(run_dir / "json"),
        "--uids-file",
        str(uids_file),
        "--strict-uids",
        "--out_dir",
        str(out_dir),
        "--budgets",
        str(args.budgets),
        "--step-s",
        str(float(args.streaming_step_s)),
        "--mode",
        str(args.nlq_mode),
        "--policy",
        "fixed",
        "--formats",
        "png,pdf",
        "--context-use-repo" if setting.use_repo else "--no-context-use-repo",
        "--repo-read-policy",
        str(args.repo_read_policy),
    ]
    if args.repo_budget:
        cmd.extend(["--repo-budget", str(args.repo_budget)])
    return cmd


def _collect_setting_metrics(run_dir: Path) -> dict[str, Any]:
    nlq_csv = run_dir / "nlq_budget" / "aggregate" / "metrics_by_budget.csv"
    stream_csv = run_dir / "streaming_budget" / "aggregate" / "metrics_by_budget.csv"
    nlq_rows = _read_csv(nlq_csv)
    stream_rows = _read_csv(stream_csv)
    strict_vals: list[float] = []
    mrr_vals: list[float] = []
    critical_vals: list[float] = []
    distractor_vals: list[float] = []
    latency_vals: list[float] = []

    if nlq_rows:
        for row in nlq_rows:
            s = _to_float(row.get("nlq_full_hit_at_k_strict"))
            if s is not None:
                strict_vals.append(float(s))
            m = _to_float(row.get("nlq_full_mrr"))
            if m is not None:
                mrr_vals.append(float(m))
            c = _to_float(row.get("safety_critical_fn_rate"))
            if c is not None:
                critical_vals.append(float(c))
            d = _to_float(row.get("nlq_full_top1_in_distractor_rate"))
            if d is not None:
                distractor_vals.append(float(d))

    if stream_rows:
        for row in stream_rows:
            if not strict_vals:
                s = _to_float(row.get("hit@k_strict"))
                if s is not None:
                    strict_vals.append(float(s))
            if not distractor_vals:
                d = _to_float(row.get("top1_in_distractor_rate"))
                if d is not None:
                    distractor_vals.append(float(d))
            l = _to_float(row.get("e2e_ms_p95"))
            if l is not None:
                latency_vals.append(float(l))

    return {
        "strict_metric": "nlq_full_hit_at_k_strict" if nlq_rows else "hit@k_strict",
        "strict_value": _mean(strict_vals),
        "mrr_strict": _mean(mrr_vals),
        "critical_fn_rate": _mean(critical_vals),
        "top1_in_distractor_rate": _mean(distractor_vals),
        "latency_p95_ms": _mean(latency_vals),
        "nlq_rows": len(nlq_rows),
        "stream_rows": len(stream_rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], header: list[str]) -> None:
    lines = ["# Component Attribution Panel", "", *header, "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(compare_dir: Path, rows_map: dict[str, dict[str, Any]], formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    fig_dir = compare_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    base = rows_map["A"]
    b = rows_map["B"]
    c = rows_map["C"]
    d = rows_map["D"]

    def _delta(metric: str, row: dict[str, Any]) -> float:
        return float(row.get(metric, 0.0) or 0.0) - float(base.get(metric, 0.0) or 0.0)

    d_repo = {"strict": _delta("strict_value", b), "critical": _delta("critical_fn_rate", b), "distractor": _delta("top1_in_distractor_rate", b), "latency": _delta("latency_p95_ms", b)}
    d_perc = {"strict": _delta("strict_value", c), "critical": _delta("critical_fn_rate", c), "distractor": _delta("top1_in_distractor_rate", c), "latency": _delta("latency_p95_ms", c)}
    d_both = {"strict": _delta("strict_value", d), "critical": _delta("critical_fn_rate", d), "distractor": _delta("top1_in_distractor_rate", d), "latency": _delta("latency_p95_ms", d)}
    synergy = {
        "strict": d_both["strict"] - d_repo["strict"] - d_perc["strict"],
        "critical": d_both["critical"] - d_repo["critical"] - d_perc["critical"],
        "distractor": d_both["distractor"] - d_repo["distractor"] - d_perc["distractor"],
        "latency": d_both["latency"] - d_repo["latency"] - d_perc["latency"],
    }

    comp_labels = ["+Repo", "+Perception", "+Repo+Perception", "Synergy"]
    metric_keys = [("strict", "strict_value"), ("critical", "critical_fn_rate"), ("distractor", "top1_in_distractor_rate"), ("latency", "latency_p95_ms")]
    p1 = fig_dir / "fig_component_attribution_delta"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8))
    for ax, (k, title) in zip(axes.flatten(), metric_keys):
        vals = [d_repo[k], d_perc[k], d_both[k], synergy[k]]
        ax.bar(range(len(comp_labels)), vals)
        ax.axhline(y=0.0, linewidth=1.0)
        ax.set_title(f"delta {title} vs A")
        ax.set_xticks(range(len(comp_labels)))
        ax.set_xticklabels(comp_labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in formats:
        path = p1.with_suffix(f".{ext}")
        fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)

    p2 = fig_dir / "fig_component_attribution_tradeoff"
    fig2, axes2 = plt.subplots(1, 2, figsize=(10.0, 4.2))
    labels = ["A", "B", "C", "D"]
    rows = [base, b, c, d]
    x1 = [float(r.get("latency_p95_ms", 0.0) or 0.0) for r in rows]
    y1 = [1.0 - float(r.get("critical_fn_rate", 0.0) or 0.0) for r in rows]
    axes2[0].scatter(x1, y1)
    for i, label in enumerate(labels):
        axes2[0].annotate(label, (x1[i], y1[i]))
    axes2[0].set_xlabel("latency_p95_ms")
    axes2[0].set_ylabel("1 - critical_fn_rate")
    axes2[0].set_title("Safety vs Latency")
    axes2[0].grid(True, alpha=0.3)

    x2 = [float(r.get("top1_in_distractor_rate", 0.0) or 0.0) for r in rows]
    y2 = [float(r.get("strict_value", 0.0) or 0.0) for r in rows]
    axes2[1].scatter(x2, y2)
    for i, label in enumerate(labels):
        axes2[1].annotate(label, (x2[i], y2[i]))
    axes2[1].set_xlabel("top1_in_distractor_rate")
    axes2[1].set_ylabel("strict_value")
    axes2[1].set_title("Strict vs Distractor")
    axes2[1].grid(True, alpha=0.3)
    fig2.tight_layout()
    for ext in formats:
        path = p2.with_suffix(f".{ext}")
        fig2.savefig(path)
        paths.append(str(path))
    plt.close(fig2)
    return paths


def _write_readme(path: Path, *, commands_file: Path, rows: list[dict[str, Any]], figure_paths: list[str]) -> None:
    lines = [
        "# Component Attribution Panel",
        "",
        "## Settings",
        "",
        "- A: Baseline (stub perception + non-repo)",
        "- B: +Repo (stub perception + repo query_aware)",
        "- C: +Perception (real perception strict + non-repo)",
        "- D: +Repo +Perception",
        "",
        "## Commands",
        "",
        f"- commands log: `{commands_file}`",
        "",
        "## Metrics",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row.get('setting')}: strict={row.get('strict_value')} critical={row.get('critical_fn_rate')} "
            f"distractor={row.get('top1_in_distractor_rate')} latency_p95={row.get('latency_p95_ms')}"
        )
    lines.extend(["", "## Figures", "", f"- {figure_paths}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    compare_dir = out_dir / "compare"
    tables_dir = compare_dir / "tables"
    commands_file = compare_dir / "commands.sh"
    logs_dir = compare_dir / "logs"
    for d in (out_dir, compare_dir, tables_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not commands_file.exists():
        commands_file.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    requested_uids = _read_uids(Path(args.uids_file))
    if not requested_uids:
        print("error=empty uids-file")
        return 2
    if args.max_uids is not None and int(args.max_uids) > 0:
        requested_uids = requested_uids[: int(args.max_uids)]
    uids_used_file = out_dir / "uids_used.txt"
    uids_used_file.write_text("\n".join(requested_uids) + "\n", encoding="utf-8")

    settings = [
        Setting("A", "Baseline", use_repo=False, perception_backend=str(args.stub_perception_backend), perception_strict=False),
        Setting("B", "+Repo", use_repo=True, perception_backend=str(args.stub_perception_backend), perception_strict=False),
        Setting(
            "C",
            "+Perception",
            use_repo=False,
            perception_backend=str(args.real_perception_backend),
            perception_strict=bool(args.real_perception_strict),
        ),
        Setting(
            "D",
            "+Repo+Perception",
            use_repo=True,
            perception_backend=str(args.real_perception_backend),
            perception_strict=bool(args.real_perception_strict),
        ),
    ]
    queries = [str(x).strip() for x in list(args.queries or []) + list(args.query or []) if str(x).strip()]

    setting_metrics: dict[str, dict[str, Any]] = {}
    for setting in settings:
        run_dir = out_dir / f"run_{setting.code}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd_smoke = _build_ego_cmd(args=args, out_dir=run_dir, uids_file=uids_used_file, setting=setting)
        rc = _run(cmd_smoke, cwd=ROOT, log_prefix=logs_dir / f"run_{setting.code}_smoke", commands_file=commands_file)
        if rc != 0:
            return rc

        if bool(args.with_nlq):
            nlq_out = run_dir / "nlq_budget"
            cmd_nlq = _build_nlq_sweep_cmd(args=args, run_dir=run_dir, uids_file=uids_used_file, out_dir=nlq_out)
            rc = _run(cmd_nlq, cwd=ROOT, log_prefix=logs_dir / f"run_{setting.code}_nlq_sweep", commands_file=commands_file)
            if rc != 0:
                return rc

        if bool(args.with_streaming_budget):
            stream_out = run_dir / "streaming_budget"
            cmd_stream = _build_streaming_sweep_cmd(
                args=args,
                run_dir=run_dir,
                uids_file=uids_used_file,
                out_dir=stream_out,
                setting=setting,
            )
            rc = _run(
                cmd_stream,
                cwd=ROOT,
                log_prefix=logs_dir / f"run_{setting.code}_streaming_sweep",
                commands_file=commands_file,
            )
            if rc != 0:
                return rc

        metrics = _collect_setting_metrics(run_dir)
        metrics["setting"] = setting.code
        metrics["name"] = setting.name
        metrics["use_repo"] = bool(setting.use_repo)
        metrics["perception_backend"] = str(setting.perception_backend)
        metrics["perception_strict"] = bool(setting.perception_strict)
        metrics["run_dir"] = str(run_dir)
        setting_metrics[setting.code] = metrics

    if "A" not in setting_metrics:
        print("error=missing baseline A metrics")
        return 2
    base = setting_metrics["A"]
    for code, row in setting_metrics.items():
        row["delta_strict_value"] = float(row.get("strict_value", 0.0) or 0.0) - float(base.get("strict_value", 0.0) or 0.0)
        row["delta_critical_fn_rate"] = float(row.get("critical_fn_rate", 0.0) or 0.0) - float(
            base.get("critical_fn_rate", 0.0) or 0.0
        )
        row["delta_top1_in_distractor_rate"] = float(row.get("top1_in_distractor_rate", 0.0) or 0.0) - float(
            base.get("top1_in_distractor_rate", 0.0) or 0.0
        )
        row["delta_latency_p95_ms"] = float(row.get("latency_p95_ms", 0.0) or 0.0) - float(base.get("latency_p95_ms", 0.0) or 0.0)

    b = setting_metrics.get("B", {})
    c = setting_metrics.get("C", {})
    d = setting_metrics.get("D", {})
    synergy = {
        "setting": "SYNERGY",
        "name": "InteractionTerm",
        "use_repo": "",
        "perception_backend": "",
        "perception_strict": "",
        "strict_metric": "interaction",
        "strict_value": "",
        "mrr_strict": "",
        "critical_fn_rate": "",
        "top1_in_distractor_rate": "",
        "latency_p95_ms": "",
        "delta_strict_value": float(d.get("delta_strict_value", 0.0) or 0.0)
        - float(b.get("delta_strict_value", 0.0) or 0.0)
        - float(c.get("delta_strict_value", 0.0) or 0.0),
        "delta_critical_fn_rate": float(d.get("delta_critical_fn_rate", 0.0) or 0.0)
        - float(b.get("delta_critical_fn_rate", 0.0) or 0.0)
        - float(c.get("delta_critical_fn_rate", 0.0) or 0.0),
        "delta_top1_in_distractor_rate": float(d.get("delta_top1_in_distractor_rate", 0.0) or 0.0)
        - float(b.get("delta_top1_in_distractor_rate", 0.0) or 0.0)
        - float(c.get("delta_top1_in_distractor_rate", 0.0) or 0.0),
        "delta_latency_p95_ms": float(d.get("delta_latency_p95_ms", 0.0) or 0.0)
        - float(b.get("delta_latency_p95_ms", 0.0) or 0.0)
        - float(c.get("delta_latency_p95_ms", 0.0) or 0.0),
        "run_dir": "",
    }

    rows = [setting_metrics["A"], setting_metrics["B"], setting_metrics["C"], setting_metrics["D"], synergy]
    columns = [
        "setting",
        "name",
        "use_repo",
        "perception_backend",
        "perception_strict",
        "strict_metric",
        "strict_value",
        "mrr_strict",
        "critical_fn_rate",
        "top1_in_distractor_rate",
        "latency_p95_ms",
        "delta_strict_value",
        "delta_critical_fn_rate",
        "delta_top1_in_distractor_rate",
        "delta_latency_p95_ms",
        "run_dir",
    ]
    table_csv = tables_dir / "table_component_attribution.csv"
    table_md = tables_dir / "table_component_attribution.md"
    _write_csv(table_csv, rows, columns)
    _write_md(
        table_md,
        rows,
        columns,
        header=[
            f"- selected_uids: {len(requested_uids)}",
            f"- budgets: {_budget_keys(args.budgets)}",
            f"- queries_total: {len(queries)}",
            "- deltas are computed against A baseline",
            "- synergy = (D-A) - (B-A) - (C-A)",
        ],
    )
    figure_paths = _make_figures(compare_dir, setting_metrics, formats=["png", "pdf"])

    summary = {
        "selected_uids": int(len(requested_uids)),
        "budgets": _budget_keys(args.budgets),
        "queries_total": int(len(queries)),
        "baseline": setting_metrics["A"],
        "delta_B_vs_A": setting_metrics["B"]["delta_strict_value"],
        "delta_C_vs_A": setting_metrics["C"]["delta_strict_value"],
        "delta_D_vs_A": setting_metrics["D"]["delta_strict_value"],
        "synergy_strict": synergy["delta_strict_value"],
        "synergy_critical_fn": synergy["delta_critical_fn_rate"],
        "synergy_distractor": synergy["delta_top1_in_distractor_rate"],
        "synergy_latency": synergy["delta_latency_p95_ms"],
    }
    compare_summary = {
        "summary": summary,
        "rows": rows,
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
            "root": str(args.root),
            "uids_file": str(args.uids_file),
            "uids_used_file": str(uids_used_file),
            "selected_uids": requested_uids,
            "jobs": int(args.jobs),
            "budgets": _budget_keys(args.budgets),
            "queries": queries,
            "with_nlq": bool(args.with_nlq),
            "nlq_mode": str(args.nlq_mode),
            "with_streaming_budget": bool(args.with_streaming_budget),
            "streaming_step_s": float(args.streaming_step_s),
            "with_perception": bool(args.with_perception),
            "repo_read_policy": str(args.repo_read_policy),
            "repo_budget": str(args.repo_budget) if args.repo_budget else None,
        },
        "settings": [
            {
                "code": s.code,
                "name": s.name,
                "use_repo": s.use_repo,
                "perception_backend": s.perception_backend,
                "perception_strict": s.perception_strict,
                "run_dir": str(out_dir / f"run_{s.code}"),
            }
            for s in settings
        ],
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
            "compare_summary_json": str(compare_summary_path),
            "commands_sh": str(commands_file),
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_readme(compare_dir / "README.md", commands_file=commands_file, rows=rows, figure_paths=figure_paths)

    print(f"selection_mode=uids_file")
    print(f"selected_uids={len(requested_uids)}")
    print(f"budgets={len(_budget_keys(args.budgets))}")
    print(f"queries_total={len(queries)}")
    for s in settings:
        print(f"saved_run_{s.code}={out_dir / f'run_{s.code}'}")
    print(f"saved_compare={compare_dir}")
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
