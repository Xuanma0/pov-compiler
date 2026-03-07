from __future__ import annotations

import argparse
import csv
import json
import subprocess
import shutil
import sys
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
    parser = argparse.ArgumentParser(description="Export unified paper-ready budget panel across BYE/NLQ/Streaming")
    parser.add_argument("--compare_dir", default=None, help="AB compare directory (optional when using direct panel dirs)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label_a", default="stub")
    parser.add_argument("--label_b", default="real")
    parser.add_argument("--primary_metrics_json", default=None)
    parser.add_argument(
        "--streaming-policy-compare-dir",
        default=None,
        help="Optional directory from run_streaming_policy_compare.py compare/ output",
    )
    parser.add_argument(
        "--streaming-chain-backoff-compare-dir",
        default=None,
        help="Optional directory from run_streaming_chain_backoff_compare.py compare/ output",
    )
    parser.add_argument(
        "--streaming-repo-compare-dir",
        default=None,
        help="Optional directory from run_streaming_repo_compare.py compare/ output",
    )
    parser.add_argument(
        "--streaming-intervention-sweep-dir",
        default=None,
        help="Optional directory from sweep_streaming_interventions.py output",
    )
    parser.add_argument(
        "--streaming-codec-sweep-dir",
        default=None,
        help="Optional directory from sweep_streaming_codec_k.py output",
    )
    parser.add_argument(
        "--repo-policy-sweep-dir",
        default=None,
        help="Optional directory from sweep_repo_policies.py output",
    )
    parser.add_argument(
        "--repo-summary-sweep-dir",
        default=None,
        help="Optional directory from sweep_repo_summary_budgets.py output",
    )
    parser.add_argument(
        "--repo-summary-retrieval-compare-dir",
        default=None,
        help="Optional directory from run_repo_summary_retrieval_compare.py compare/ output",
    )
    parser.add_argument(
        "--repo-query-selection-sweep-dir",
        default=None,
        help="Optional directory from sweep_repo_query_selection.py output",
    )
    parser.add_argument(
        "--component-attribution-dir",
        default=None,
        help="Optional directory from run_component_attribution.py compare/ output",
    )
    parser.add_argument(
        "--bye-report-compare-dir",
        default=None,
        help="Optional directory from compare_bye_report_metrics.py output",
    )
    parser.add_argument(
        "--decisions-backend-compare-dir",
        default=None,
        help="Optional directory from run_decisions_backend_compare.py compare output",
    )
    parser.add_argument(
        "--planner-backend-compare-dir",
        default=None,
        help="Optional directory from run_planner_backend_compare.py compare output",
    )
    parser.add_argument(
        "--model-stack-compare-dir",
        default=None,
        help="Optional directory from run_model_stack_compare.py compare output",
    )
    parser.add_argument(
        "--model-cost-compare-dir",
        default=None,
        help="Optional directory containing model cost tables/figures (defaults to --model-stack-compare-dir when omitted)",
    )
    parser.add_argument(
        "--lost-object-panel-dir",
        default=None,
        help="Optional directory containing table_lost_object_budget.(csv/md) and optional figures",
    )
    parser.add_argument(
        "--reranker-sweep-dir",
        default=None,
        help="Optional directory from sweep_reranker.py output",
    )
    parser.add_argument(
        "--chain-nlq-dir",
        default=None,
        help="Optional directory containing chain NLQ outputs (table_chain_summary.* and optional figures)",
    )
    parser.add_argument(
        "--chain-repo-compare-dir",
        default=None,
        help="Optional directory from run_chain_repo_compare.py compare/ output",
    )
    parser.add_argument(
        "--chain-attribution-dir",
        default=None,
        help="Optional directory from run_chain_attribution.py compare/ output",
    )
    parser.add_argument(
        "--signal-selection-dir",
        default=None,
        help="Optional directory containing signal selection artifacts (coverage.md, selection_report.md, selected_uids.txt)",
    )
    parser.add_argument("--suite-dir", default=None, help="Optional benchmark suite root containing manifest/ and ledger/")
    parser.add_argument("--significance-dir", default=None, help="Optional statistical significance output directory")
    parser.add_argument("--result-health-dir", default=None, help="Optional result health output directory")
    parser.add_argument("--benchmark-freeze-dir", default=None, help="Optional benchmark freeze output directory")
    parser.add_argument("--prompt-registry", default=None, help="Optional prompt registry YAML path")
    parser.add_argument("--prompt-lock", default=None, help="Optional prompt lock JSON path")
    parser.add_argument("--submission-pack-dir", default=None, help="Optional explicit submission_pack output directory")
    _parse_bool_with_neg(parser, "export-submission-pack", default=False)
    parser.add_argument("--format", choices=["md", "csv", "md+csv"], default="md+csv")
    _parse_bool_with_neg(parser, "with-figs", default=True)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def _parse_json_arg(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    p = Path(text)
    if p.exists():
        try:
            val = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(val, dict):
                return val
        except Exception:
            return {}
    return {}


def _budget_key(row: dict[str, Any]) -> str:
    def _pick(*keys: str) -> float | None:
        for k in keys:
            if k in row:
                v = _to_float(row.get(k))
                if v is not None:
                    return v
        return None

    s = _pick("budget_seconds", "budget_max_total_s", "max_total_s")
    t = _pick("budget_max_tokens", "max_tokens")
    d = _pick("budget_max_decisions", "max_decisions")
    if s is not None and t is not None and d is not None:
        return f"{int(round(s))}/{int(t)}/{int(d)}"
    tag = str(row.get("budget_key", row.get("budget_tag", ""))).strip()
    return tag or "unknown_budget"


def _budget_seconds(row: dict[str, Any]) -> float:
    for k in ("budget_seconds", "budget_max_total_s", "max_total_s"):
        v = _to_float(row.get(k))
        if v is not None:
            return float(v)
    key = _budget_key(row)
    parts = key.split("/")
    if len(parts) == 3:
        v = _to_float(parts[0])
        if v is not None:
            return float(v)
    return 0.0


def _to_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _budget_key(row)
        if key not in out:
            out[key] = row
    return out


def _numeric_cols(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    reserved = {
        "budget_key",
        "budget_tag",
        "budget_seconds",
        "budget_max_total_s",
        "budget_max_tokens",
        "budget_max_decisions",
        "num_uids",
        "runs_total",
        "runs_ok",
        "selection_mode",
        "uids_requested",
        "uids_found",
        "uids_missing_count",
        "policy",
    }
    out: list[str] = []
    for key in rows[0].keys():
        if key in reserved:
            continue
        if any(_to_float(r.get(key)) is not None for r in rows):
            out.append(str(key))
    return out


def _select_primary(task: str, rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], override: dict[str, Any]) -> str | None:
    rows = list(rows_a) + list(rows_b)
    cols = _numeric_cols(rows)
    if not cols:
        return None
    requested = str(override.get(task, "")).strip()
    if requested and requested in cols:
        return requested
    candidates: list[str]
    task_l = task.lower()
    if task_l == "bye":
        candidates = ["qualityScore", "bye_qualityScore", "bye_primary", "primary_metric"]
        for cand in candidates:
            if cand in cols:
                return cand
        pref = [c for c in cols if c.startswith("bye_numeric_primary_")]
        if pref:
            pref.sort()
            return pref[0]
    elif task_l == "nlq":
        candidates = ["objective", "nlq_full_hit_at_k_strict", "full_hit_at_k_strict", "hit_at_k_strict"]
        for cand in candidates:
            if cand in cols:
                return cand
    elif task_l == "repo":
        candidates = ["repo_quality_proxy", "repo_coverage_ratio", "repo_importance_mean"]
        for cand in candidates:
            if cand in cols:
                return cand
    else:
        candidates = ["hit@k_strict", "hit_at_k_strict", "nlq_full_hit_at_k_strict", "mrr"]
        for cand in candidates:
            if cand in cols:
                return cand
    cols.sort()
    return cols[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str, summary_lines: list[str]) -> None:
    lines = [f"# {title}", "", *summary_lines, "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(
    *,
    panel_rows: list[dict[str, Any]],
    out_dir: Path,
    label_a: str,
    label_b: str,
    with_figs: bool,
    formats: list[str],
    recommend_points: dict[str, dict[str, str]],
) -> list[str]:
    if not with_figs:
        return []
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    task_names = sorted({str(r.get("task", "")) for r in panel_rows if str(r.get("task", ""))})

    # 1) Primary panel.
    fig1 = out_dir / "fig_budget_primary_vs_seconds_panel"
    plt.figure(figsize=(8.2, 4.6))
    for task in task_names:
        rows = sorted([r for r in panel_rows if str(r.get("task")) == task], key=lambda x: float(x.get("budget_seconds", 0.0)))
        xs = [float(r.get("budget_seconds", 0.0)) for r in rows]
        ya = [float(_to_float(r.get("primary_a")) or 0.0) for r in rows]
        yb = [float(_to_float(r.get("primary_b")) or 0.0) for r in rows]
        plt.plot(xs, ya, marker="o", linestyle="-", label=f"{task}-{label_a}")
        plt.plot(xs, yb, marker="o", linestyle="--", label=f"{task}-{label_b}")
    plt.xlabel("Budget Seconds")
    plt.ylabel("Primary Metric")
    plt.title("Budget Primary Curves Panel")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()
    for ext in formats:
        p = fig1.with_suffix(f".{ext}")
        plt.savefig(p)
        figure_paths.append(str(p))
    plt.close()

    # 2) Delta panel.
    fig2 = out_dir / "fig_budget_primary_delta_vs_seconds_panel"
    plt.figure(figsize=(8.2, 4.6))
    for task in task_names:
        rows = sorted([r for r in panel_rows if str(r.get("task")) == task], key=lambda x: float(x.get("budget_seconds", 0.0)))
        xs = [float(r.get("budget_seconds", 0.0)) for r in rows]
        yd = [float(_to_float(r.get("delta_primary")) or 0.0) for r in rows]
        plt.plot(xs, yd, marker="o", label=task)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("Budget Seconds")
    plt.ylabel(f"Delta Primary ({label_b}-{label_a})")
    plt.title("Budget Primary Delta Panel")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig2.with_suffix(f".{ext}")
        plt.savefig(p)
        figure_paths.append(str(p))
    plt.close()

    # 3) Streaming latency.
    fig3 = out_dir / "fig_budget_latency_vs_seconds_streaming"
    stream_rows = sorted([r for r in panel_rows if str(r.get("task")) == "streaming"], key=lambda x: float(x.get("budget_seconds", 0.0)))
    plt.figure(figsize=(8.2, 4.6))
    if stream_rows:
        xs = [float(r.get("budget_seconds", 0.0)) for r in stream_rows]
        e2e_a = [float(_to_float(r.get("e2e_ms_p50_a")) or 0.0) for r in stream_rows]
        e2e_b = [float(_to_float(r.get("e2e_ms_p50_b")) or 0.0) for r in stream_rows]
        p95_a = [float(_to_float(r.get("e2e_ms_p95_a")) or 0.0) for r in stream_rows]
        p95_b = [float(_to_float(r.get("e2e_ms_p95_b")) or 0.0) for r in stream_rows]
        plt.plot(xs, e2e_a, marker="o", linestyle="-", label=f"e2e_p50_{label_a}")
        plt.plot(xs, e2e_b, marker="o", linestyle="--", label=f"e2e_p50_{label_b}")
        plt.plot(xs, p95_a, marker="s", linestyle="-", label=f"e2e_p95_{label_a}")
        plt.plot(xs, p95_b, marker="s", linestyle="--", label=f"e2e_p95_{label_b}")
    else:
        plt.text(0.5, 0.5, "streaming metrics missing", ha="center", va="center")
    plt.xlabel("Budget Seconds")
    plt.ylabel("Latency (ms)")
    plt.title("Streaming Latency vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig3.with_suffix(f".{ext}")
        plt.savefig(p)
        figure_paths.append(str(p))
    plt.close()

    # 4) recommended points.
    fig4 = out_dir / "fig_budget_recommended_points"
    plt.figure(figsize=(8.2, 4.6))
    drew = False
    for side, cfg in ((label_a, recommend_points.get(label_a, {})), (label_b, recommend_points.get(label_b, {}))):
        budget_key = str(cfg.get("top1_budget_key", "")).strip()
        if not budget_key:
            continue
        for task in task_names:
            rows = [r for r in panel_rows if str(r.get("task")) == task and str(r.get("budget_key")) == budget_key]
            if not rows:
                continue
            row = rows[0]
            y = _to_float(row.get("primary_a" if side == label_a else "primary_b"))
            if y is None:
                continue
            x = float(row.get("budget_seconds", 0.0))
            plt.scatter([x], [float(y)], marker="*", s=140, label=f"{task}-{side}-top1")
            drew = True
    if not drew:
        plt.text(0.5, 0.5, "recommendation points missing", ha="center", va="center")
    plt.xlabel("Budget Seconds")
    plt.ylabel("Primary Metric")
    plt.title("Recommended Budget Points")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig4.with_suffix(f".{ext}")
        plt.savefig(p)
        figure_paths.append(str(p))
    plt.close()

    return figure_paths


def _make_nlq_safety_figures(
    *,
    panel_rows: list[dict[str, Any]],
    out_dir: Path,
    label_a: str,
    label_b: str,
    with_figs: bool,
    formats: list[str],
) -> list[str]:
    if not with_figs:
        return []
    import matplotlib.pyplot as plt

    rows = sorted(
        [r for r in panel_rows if str(r.get("task")) == "nlq"],
        key=lambda x: float(x.get("budget_seconds", 0.0)),
    )
    if not rows:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    xs = [float(_to_float(r.get("budget_seconds")) or 0.0) for r in rows]
    y_crit_a = [float(_to_float(r.get("safety_critical_fn_rate_a")) or 0.0) for r in rows]
    y_crit_b = [float(_to_float(r.get("safety_critical_fn_rate_b")) or 0.0) for r in rows]
    y_bi_a = [float(_to_float(r.get("safety_reason_budget_insufficient_rate_a")) or 0.0) for r in rows]
    y_bi_b = [float(_to_float(r.get("safety_reason_budget_insufficient_rate_b")) or 0.0) for r in rows]
    y_other_a = [
        float(_to_float(r.get("safety_reason_evidence_missing_rate_a")) or 0.0)
        + float(_to_float(r.get("safety_reason_constraints_over_filtered_rate_a")) or 0.0)
        + float(_to_float(r.get("safety_reason_retrieval_distractor_rate_a")) or 0.0)
        + float(_to_float(r.get("safety_reason_other_rate_a")) or 0.0)
        for r in rows
    ]
    y_other_b = [
        float(_to_float(r.get("safety_reason_evidence_missing_rate_b")) or 0.0)
        + float(_to_float(r.get("safety_reason_constraints_over_filtered_rate_b")) or 0.0)
        + float(_to_float(r.get("safety_reason_retrieval_distractor_rate_b")) or 0.0)
        + float(_to_float(r.get("safety_reason_other_rate_b")) or 0.0)
        for r in rows
    ]

    out_paths: list[str] = []
    p1 = out_dir / "fig_nlq_critical_fn_rate_vs_seconds"
    plt.figure(figsize=(7.4, 4.2))
    plt.plot(xs, y_crit_a, marker="o", linestyle="-", label=f"{label_a}")
    plt.plot(xs, y_crit_b, marker="o", linestyle="--", label=f"{label_b}")
    plt.xlabel("Budget Seconds")
    plt.ylabel("safety_critical_fn_rate")
    plt.title("NLQ Critical FN Rate vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        out_paths.append(str(p))
    plt.close()

    p2 = out_dir / "fig_nlq_failure_attribution_vs_seconds"
    plt.figure(figsize=(8.0, 4.4))
    width = 1.8
    x_left = [x - width * 0.3 for x in xs]
    x_right = [x + width * 0.3 for x in xs]
    plt.bar(x_left, y_bi_a, width=width * 0.6, label=f"{label_a}: budget_insufficient")
    plt.bar(x_left, y_other_a, width=width * 0.6, bottom=y_bi_a, label=f"{label_a}: other_reasons")
    plt.bar(x_right, y_bi_b, width=width * 0.6, label=f"{label_b}: budget_insufficient")
    plt.bar(x_right, y_other_b, width=width * 0.6, bottom=y_bi_b, label=f"{label_b}: other_reasons")
    plt.xlabel("Budget Seconds")
    plt.ylabel("Failure Attribution Rate")
    plt.title("NLQ Failure Attribution vs Budget Seconds")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        out_paths.append(str(p))
    plt.close()
    return out_paths


def main() -> int:
    args = parse_args()
    compare_dir = Path(args.compare_dir) if args.compare_dir else Path(".")
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.png or args.pdf:
        formats = []
        if args.png:
            formats.append("png")
        if args.pdf:
            formats.append("pdf")
    else:
        formats = ["png", "pdf"]

    primary_override = _parse_json_arg(args.primary_metrics_json)
    task_sources = {
        "bye": {
            args.label_a: compare_dir / "bye_budget" / args.label_a / "aggregate" / "metrics_by_budget.csv",
            args.label_b: compare_dir / "bye_budget" / args.label_b / "aggregate" / "metrics_by_budget.csv",
        },
        "nlq": {
            args.label_a: compare_dir / "nlq_budget" / args.label_a / "aggregate" / "metrics_by_budget.csv",
            args.label_b: compare_dir / "nlq_budget" / args.label_b / "aggregate" / "metrics_by_budget.csv",
        },
        "streaming": {
            args.label_a: compare_dir / "streaming_budget" / args.label_a / "aggregate" / "metrics_by_budget.csv",
            args.label_b: compare_dir / "streaming_budget" / args.label_b / "aggregate" / "metrics_by_budget.csv",
        },
        "repo": {
            args.label_a: compare_dir / "repo_budget" / args.label_a / "aggregate" / "metrics_by_budget.csv",
            args.label_b: compare_dir / "repo_budget" / args.label_b / "aggregate" / "metrics_by_budget.csv",
        },
    }
    recommend_sources = {
        args.label_a: compare_dir / "budget_recommend" / args.label_a / "recommend_summary.json",
        args.label_b: compare_dir / "budget_recommend" / args.label_b / "recommend_summary.json",
    }

    panel_rows: list[dict[str, Any]] = []
    missing_tasks: list[str] = []
    chosen_primary: dict[str, str] = {}
    safety_fields = [
        "safety_critical_fn_denominator",
        "safety_critical_fn_count",
        "safety_critical_fn_rate",
        "safety_reason_budget_insufficient_rate",
        "safety_reason_evidence_missing_rate",
        "safety_reason_constraints_over_filtered_rate",
        "safety_reason_retrieval_distractor_rate",
        "safety_reason_other_rate",
        "safety_budget_insufficient_share",
    ]
    safety_present = False
    for task, paths in task_sources.items():
        rows_a = _read_csv(Path(paths[args.label_a]))
        rows_b = _read_csv(Path(paths[args.label_b]))
        if not rows_a and not rows_b:
            missing_tasks.append(task)
            continue
        primary = _select_primary(task, rows_a, rows_b, primary_override)
        if not primary:
            missing_tasks.append(task)
            continue
        chosen_primary[task] = primary
        map_a = _to_map(rows_a)
        map_b = _to_map(rows_b)
        all_keys = sorted(set(map_a.keys()) | set(map_b.keys()), key=lambda k: _budget_seconds(map_a.get(k, map_b.get(k, {}))))
        for key in all_keys:
            ra = map_a.get(key, {})
            rb = map_b.get(key, {})
            pa = _to_float(ra.get(primary))
            pb = _to_float(rb.get(primary))
            row = {
                "task": task,
                "budget_key": key,
                "budget_seconds": _budget_seconds(ra or rb),
                "primary_metric": primary,
                "primary_a": pa,
                "primary_b": pb,
                "delta_primary": None if pa is None or pb is None else float(pb - pa),
                "status_a": "ok" if ra else "missing",
                "status_b": "ok" if rb else "missing",
            }
            # attach common latency fields for streaming chart.
            for fld in ("e2e_ms_p50", "e2e_ms_p95", "retrieval_ms_p50", "retrieval_ms_p95"):
                row[f"{fld}_a"] = _to_float(ra.get(fld))
                row[f"{fld}_b"] = _to_float(rb.get(fld))
            row["safety_count_granularity_a"] = str(ra.get("safety_count_granularity", "")) if ra else ""
            row["safety_count_granularity_b"] = str(rb.get("safety_count_granularity", "")) if rb else ""
            for sf in safety_fields:
                row[f"{sf}_a"] = _to_float(ra.get(sf))
                row[f"{sf}_b"] = _to_float(rb.get(sf))
                if task == "nlq" and (row[f"{sf}_a"] is not None or row[f"{sf}_b"] is not None):
                    safety_present = True
            panel_rows.append(row)

    panel_rows.sort(key=lambda r: (str(r.get("task", "")), float(_to_float(r.get("budget_seconds")) or 0.0), str(r.get("budget_key", ""))))

    delta_rows = [
        {
            "task": str(r.get("task", "")),
            "budget_key": str(r.get("budget_key", "")),
            "budget_seconds": float(_to_float(r.get("budget_seconds")) or 0.0),
            "delta_primary": _to_float(r.get("delta_primary")),
            "primary_metric": str(r.get("primary_metric", "")),
        }
        for r in panel_rows
    ]

    panel_cols = [
        "task",
        "budget_key",
        "budget_seconds",
        "primary_metric",
        "primary_a",
        "primary_b",
        "delta_primary",
        "status_a",
        "status_b",
        "e2e_ms_p50_a",
        "e2e_ms_p50_b",
        "e2e_ms_p95_a",
        "e2e_ms_p95_b",
        "retrieval_ms_p50_a",
        "retrieval_ms_p50_b",
        "retrieval_ms_p95_a",
        "retrieval_ms_p95_b",
        "safety_count_granularity_a",
        "safety_count_granularity_b",
        "safety_critical_fn_denominator_a",
        "safety_critical_fn_denominator_b",
        "safety_critical_fn_count_a",
        "safety_critical_fn_count_b",
        "safety_critical_fn_rate_a",
        "safety_critical_fn_rate_b",
        "safety_reason_budget_insufficient_rate_a",
        "safety_reason_budget_insufficient_rate_b",
        "safety_reason_evidence_missing_rate_a",
        "safety_reason_evidence_missing_rate_b",
        "safety_reason_constraints_over_filtered_rate_a",
        "safety_reason_constraints_over_filtered_rate_b",
        "safety_reason_retrieval_distractor_rate_a",
        "safety_reason_retrieval_distractor_rate_b",
        "safety_reason_other_rate_a",
        "safety_reason_other_rate_b",
        "safety_budget_insufficient_share_a",
        "safety_budget_insufficient_share_b",
    ]
    delta_cols = ["task", "budget_key", "budget_seconds", "primary_metric", "delta_primary"]

    panel_csv = tables_dir / "table_budget_panel.csv"
    panel_md = tables_dir / "table_budget_panel.md"
    delta_csv = tables_dir / "table_budget_panel_delta.csv"
    delta_md = tables_dir / "table_budget_panel_delta.md"

    if args.format in {"csv", "md+csv"}:
        _write_csv(panel_csv, panel_rows, panel_cols)
        _write_csv(delta_csv, delta_rows, delta_cols)
    if args.format in {"md", "md+csv"}:
        summary_lines = [
            f"- compare_dir: `{compare_dir}`",
            f"- labels: `{args.label_a}` vs `{args.label_b}`",
            f"- missing_tasks: `{missing_tasks}`",
            f"- primary_metrics: `{json.dumps(chosen_primary, ensure_ascii=False, sort_keys=True)}`",
        ]
        _write_md(panel_md, panel_rows, panel_cols, "Unified Budget Panel", summary_lines)
        _write_md(delta_md, delta_rows, delta_cols, "Unified Budget Panel Deltas", summary_lines)

    recommend_points: dict[str, dict[str, str]] = {}
    for side, p in recommend_sources.items():
        if not p.exists():
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            recommend_points[side] = {"top1_budget_key": str(payload.get("top1_budget_key", "")).strip()}

    figure_paths = _make_figures(
        panel_rows=panel_rows,
        out_dir=figures_dir,
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        with_figs=bool(args.with_figs),
        formats=formats,
        recommend_points=recommend_points,
    )
    safety_figure_paths = _make_nlq_safety_figures(
        panel_rows=panel_rows,
        out_dir=figures_dir,
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        with_figs=bool(args.with_figs) and bool(safety_present),
        formats=formats,
    )
    figure_paths = list(figure_paths) + list(safety_figure_paths)

    streaming_policy_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_tables": [],
        "copied_figures": [],
        "copied_summary": None,
    }
    if args.streaming_policy_compare_dir:
        spc_dir = Path(args.streaming_policy_compare_dir)
        streaming_policy_compare["enabled"] = True
        streaming_policy_compare["source_dir"] = str(spc_dir)
        table_src_csv = spc_dir / "tables" / "table_streaming_policy_compare.csv"
        table_src_md = spc_dir / "tables" / "table_streaming_policy_compare.md"
        fig_src_png = spc_dir / "figures" / "fig_streaming_policy_compare_safety_latency.png"
        fig_src_pdf = spc_dir / "figures" / "fig_streaming_policy_compare_safety_latency.pdf"
        fig_delta_png = spc_dir / "figures" / "fig_streaming_policy_compare_delta.png"
        fig_delta_pdf = spc_dir / "figures" / "fig_streaming_policy_compare_delta.pdf"
        fig_chain_png = spc_dir / "figures" / "fig_streaming_policy_compare_chain_success.png"
        fig_chain_pdf = spc_dir / "figures" / "fig_streaming_policy_compare_chain_success.pdf"
        fig_chain_delta_png = spc_dir / "figures" / "fig_streaming_policy_compare_chain_delta.png"
        fig_chain_delta_pdf = spc_dir / "figures" / "fig_streaming_policy_compare_chain_delta.pdf"
        summary_src = spc_dir / "compare_summary.json"

        copied_table_csv = _copy_if_exists(table_src_csv, tables_dir / table_src_csv.name)
        copied_table_md = _copy_if_exists(table_src_md, tables_dir / table_src_md.name)
        copied_summary = _copy_if_exists(summary_src, out_dir / summary_src.name)
        copied_figures = [
            _copy_if_exists(fig_src_png, figures_dir / fig_src_png.name),
            _copy_if_exists(fig_src_pdf, figures_dir / fig_src_pdf.name),
            _copy_if_exists(fig_delta_png, figures_dir / fig_delta_png.name),
            _copy_if_exists(fig_delta_pdf, figures_dir / fig_delta_pdf.name),
            _copy_if_exists(fig_chain_png, figures_dir / fig_chain_png.name),
            _copy_if_exists(fig_chain_pdf, figures_dir / fig_chain_pdf.name),
            _copy_if_exists(fig_chain_delta_png, figures_dir / fig_chain_delta_png.name),
            _copy_if_exists(fig_chain_delta_pdf, figures_dir / fig_chain_delta_pdf.name),
        ]
        streaming_policy_compare["copied_tables"] = [x for x in [copied_table_csv, copied_table_md] if x]
        streaming_policy_compare["copied_figures"] = [x for x in copied_figures if x]
        streaming_policy_compare["copied_summary"] = copied_summary
        figure_paths.extend(streaming_policy_compare["copied_figures"])

    streaming_chain_backoff_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_tables": [],
        "copied_figures": [],
        "copied_summary": None,
    }
    if args.streaming_chain_backoff_compare_dir:
        src_dir = Path(args.streaming_chain_backoff_compare_dir)
        streaming_chain_backoff_compare["enabled"] = True
        streaming_chain_backoff_compare["source_dir"] = str(src_dir)
        table_src_csv = src_dir / "tables" / "table_streaming_chain_backoff_compare.csv"
        table_src_md = src_dir / "tables" / "table_streaming_chain_backoff_compare.md"
        fig1_png = src_dir / "figures" / "fig_streaming_chain_backoff_success_vs_budget_seconds.png"
        fig1_pdf = src_dir / "figures" / "fig_streaming_chain_backoff_success_vs_budget_seconds.pdf"
        fig2_png = src_dir / "figures" / "fig_streaming_chain_backoff_latency_vs_budget_seconds.png"
        fig2_pdf = src_dir / "figures" / "fig_streaming_chain_backoff_latency_vs_budget_seconds.pdf"
        fig3_png = src_dir / "figures" / "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds.png"
        fig3_pdf = src_dir / "figures" / "fig_streaming_chain_backoff_backoff_level_vs_budget_seconds.pdf"
        fig4_png = src_dir / "figures" / "fig_streaming_chain_backoff_delta.png"
        fig4_pdf = src_dir / "figures" / "fig_streaming_chain_backoff_delta.pdf"
        summary_src = src_dir / "compare_summary.json"

        copied_table_csv = _copy_if_exists(table_src_csv, tables_dir / table_src_csv.name)
        copied_table_md = _copy_if_exists(table_src_md, tables_dir / table_src_md.name)
        copied_summary = _copy_if_exists(summary_src, out_dir / "streaming_chain_backoff_compare_summary.json")
        copied_figures = [
            _copy_if_exists(fig1_png, figures_dir / fig1_png.name),
            _copy_if_exists(fig1_pdf, figures_dir / fig1_pdf.name),
            _copy_if_exists(fig2_png, figures_dir / fig2_png.name),
            _copy_if_exists(fig2_pdf, figures_dir / fig2_pdf.name),
            _copy_if_exists(fig3_png, figures_dir / fig3_png.name),
            _copy_if_exists(fig3_pdf, figures_dir / fig3_pdf.name),
            _copy_if_exists(fig4_png, figures_dir / fig4_png.name),
            _copy_if_exists(fig4_pdf, figures_dir / fig4_pdf.name),
        ]
        streaming_chain_backoff_compare["copied_tables"] = [x for x in [copied_table_csv, copied_table_md] if x]
        streaming_chain_backoff_compare["copied_figures"] = [x for x in copied_figures if x]
        streaming_chain_backoff_compare["copied_summary"] = copied_summary
        figure_paths.extend(streaming_chain_backoff_compare["copied_figures"])

    streaming_repo_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_tables": [],
        "copied_figures": [],
        "copied_summary": None,
    }
    if args.streaming_repo_compare_dir:
        src_dir = Path(args.streaming_repo_compare_dir)
        streaming_repo_compare["enabled"] = True
        streaming_repo_compare["source_dir"] = str(src_dir)
        table_src_csv = src_dir / "tables" / "table_streaming_repo_compare.csv"
        table_src_md = src_dir / "tables" / "table_streaming_repo_compare.md"
        fig_src_png = src_dir / "figures" / "fig_streaming_repo_compare_safety_latency.png"
        fig_src_pdf = src_dir / "figures" / "fig_streaming_repo_compare_safety_latency.pdf"
        fig_delta_png = src_dir / "figures" / "fig_streaming_repo_compare_delta.png"
        fig_delta_pdf = src_dir / "figures" / "fig_streaming_repo_compare_delta.pdf"
        summary_src = src_dir / "compare_summary.json"

        copied_table_csv = _copy_if_exists(table_src_csv, tables_dir / table_src_csv.name)
        copied_table_md = _copy_if_exists(table_src_md, tables_dir / table_src_md.name)
        copied_summary = _copy_if_exists(summary_src, out_dir / "streaming_repo_compare_summary.json")
        copied_figures = [
            _copy_if_exists(fig_src_png, figures_dir / fig_src_png.name),
            _copy_if_exists(fig_src_pdf, figures_dir / fig_src_pdf.name),
            _copy_if_exists(fig_delta_png, figures_dir / fig_delta_png.name),
            _copy_if_exists(fig_delta_pdf, figures_dir / fig_delta_pdf.name),
        ]
        streaming_repo_compare["copied_tables"] = [x for x in [copied_table_csv, copied_table_md] if x]
        streaming_repo_compare["copied_figures"] = [x for x in copied_figures if x]
        streaming_repo_compare["copied_summary"] = copied_summary
        figure_paths.extend(streaming_repo_compare["copied_figures"])

    streaming_intervention_sweep: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "best_summary": {},
    }
    if args.streaming_intervention_sweep_dir:
        sis_dir = Path(args.streaming_intervention_sweep_dir)
        streaming_intervention_sweep["enabled"] = True
        streaming_intervention_sweep["source_dir"] = str(sis_dir)
        dst_root = out_dir / "streaming_intervention_sweep"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            sis_dir / "best_config.yaml",
            sis_dir / "best_report.md",
            sis_dir / "snapshot.json",
            sis_dir / "figures" / "fig_objective_vs_latency.png",
            sis_dir / "figures" / "fig_objective_vs_latency.pdf",
            sis_dir / "figures" / "fig_pareto_frontier.png",
            sis_dir / "figures" / "fig_pareto_frontier.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        streaming_intervention_sweep["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        snap_src = sis_dir / "snapshot.json"
        if snap_src.exists():
            try:
                snap_payload = json.loads(snap_src.read_text(encoding="utf-8"))
                if isinstance(snap_payload, dict):
                    best = snap_payload.get("best", {})
                    if isinstance(best, dict):
                        streaming_intervention_sweep["best_summary"] = {
                            "cfg_name": best.get("cfg_name", ""),
                            "cfg_hash": best.get("cfg_hash", ""),
                            "objective": best.get("objective", 0.0),
                        }
                    default = snap_payload.get("default", {})
                    if isinstance(default, dict):
                        streaming_intervention_sweep["best_summary"]["default_objective"] = default.get("objective", 0.0)
            except Exception:
                pass

    streaming_codec_sweep: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.streaming_codec_sweep_dir:
        scs_dir = Path(args.streaming_codec_sweep_dir)
        streaming_codec_sweep["enabled"] = True
        streaming_codec_sweep["source_dir"] = str(scs_dir)
        dst_root = out_dir / "streaming_codec_sweep"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            scs_dir / "aggregate" / "metrics_by_k.csv",
            scs_dir / "aggregate" / "metrics_by_k.md",
            scs_dir / "snapshot.json",
            scs_dir / "figures" / "fig_streaming_quality_vs_k.png",
            scs_dir / "figures" / "fig_streaming_quality_vs_k.pdf",
            scs_dir / "figures" / "fig_streaming_safety_vs_k.png",
            scs_dir / "figures" / "fig_streaming_safety_vs_k.pdf",
            scs_dir / "figures" / "fig_streaming_latency_vs_k.png",
            scs_dir / "figures" / "fig_streaming_latency_vs_k.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        streaming_codec_sweep["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    reranker_sweep: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "best_summary": {},
    }
    if args.reranker_sweep_dir:
        rs_dir = Path(args.reranker_sweep_dir)
        reranker_sweep["enabled"] = True
        reranker_sweep["source_dir"] = str(rs_dir)
        dst_root = out_dir / "reranker_sweep"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            rs_dir / "aggregate" / "metrics_by_weights.csv",
            rs_dir / "aggregate" / "metrics_by_weights.md",
            rs_dir / "best_weights.yaml",
            rs_dir / "best_report.md",
            rs_dir / "snapshot.json",
            rs_dir / "figures" / "fig_objective_vs_weights_id.png",
            rs_dir / "figures" / "fig_objective_vs_weights_id.pdf",
            rs_dir / "figures" / "fig_tradeoff_strict_vs_distractor.png",
            rs_dir / "figures" / "fig_tradeoff_strict_vs_distractor.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        reranker_sweep["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        snap_src = rs_dir / "snapshot.json"
        if snap_src.exists():
            try:
                snap_payload = json.loads(snap_src.read_text(encoding="utf-8"))
                if isinstance(snap_payload, dict):
                    best = snap_payload.get("best", {})
                    if isinstance(best, dict):
                        reranker_sweep["best_summary"] = {
                            "cfg_name": best.get("cfg_name", ""),
                            "cfg_hash": best.get("cfg_hash", ""),
                            "objective": best.get("objective", 0.0),
                        }
            except Exception:
                pass

    repo_policy_sweep: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.repo_policy_sweep_dir:
        rps_dir = Path(args.repo_policy_sweep_dir)
        repo_policy_sweep["enabled"] = True
        repo_policy_sweep["source_dir"] = str(rps_dir)
        dst_root = out_dir / "repo_policy"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            rps_dir / "aggregate" / "metrics_by_setting.csv",
            rps_dir / "aggregate" / "metrics_by_setting.md",
            rps_dir / "best_report.md",
            rps_dir / "snapshot.json",
            rps_dir / "figures" / "fig_repo_quality_vs_budget_seconds.png",
            rps_dir / "figures" / "fig_repo_quality_vs_budget_seconds.pdf",
            rps_dir / "figures" / "fig_repo_size_vs_budget_seconds.png",
            rps_dir / "figures" / "fig_repo_size_vs_budget_seconds.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        repo_policy_sweep["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    repo_summary_sweep: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.repo_summary_sweep_dir:
        rss_dir = Path(args.repo_summary_sweep_dir)
        repo_summary_sweep["enabled"] = True
        repo_summary_sweep["source_dir"] = str(rss_dir)
        dst_root = out_dir / "repo_summary"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            rss_dir / "aggregate" / "metrics_by_policy_budget.csv",
            rss_dir / "aggregate" / "metrics_by_policy_budget.md",
            rss_dir / "snapshot.json",
            rss_dir / "figures" / "fig_repo_summary_quality_vs_budget_seconds.png",
            rss_dir / "figures" / "fig_repo_summary_quality_vs_budget_seconds.pdf",
            rss_dir / "figures" / "fig_repo_summary_size_vs_budget_seconds.png",
            rss_dir / "figures" / "fig_repo_summary_size_vs_budget_seconds.pdf",
            rss_dir / "figures" / "fig_repo_summary_delta_vs_budget_seconds.png",
            rss_dir / "figures" / "fig_repo_summary_delta_vs_budget_seconds.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        repo_summary_sweep["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    repo_summary_retrieval_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "summary": {},
    }
    if args.repo_summary_retrieval_compare_dir:
        rsr_dir = Path(args.repo_summary_retrieval_compare_dir)
        repo_summary_retrieval_compare["enabled"] = True
        repo_summary_retrieval_compare["source_dir"] = str(rsr_dir)
        dst_root = out_dir / "repo_summary_retrieval"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            rsr_dir / "tables" / "table_repo_summary_retrieval_compare.csv",
            rsr_dir / "tables" / "table_repo_summary_retrieval_compare.md",
            rsr_dir / "compare_summary.json",
            rsr_dir / "snapshot.json",
            rsr_dir / "figures" / "fig_repo_summary_retrieval_quality_vs_budget_seconds.png",
            rsr_dir / "figures" / "fig_repo_summary_retrieval_quality_vs_budget_seconds.pdf",
            rsr_dir / "figures" / "fig_repo_summary_retrieval_delta.png",
            rsr_dir / "figures" / "fig_repo_summary_retrieval_delta.pdf",
            rsr_dir / "figures" / "fig_repo_summary_retrieval_candidate_scale.png",
            rsr_dir / "figures" / "fig_repo_summary_retrieval_candidate_scale.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        repo_summary_retrieval_compare["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        summary_src = rsr_dir / "compare_summary.json"
        if summary_src.exists():
            try:
                payload = json.loads(summary_src.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    repo_summary_retrieval_compare["summary"] = {
                        "budgets_matched": payload.get("budgets_matched", 0),
                        "uids_total": payload.get("uids_total", 0),
                    }
            except Exception:
                pass

    repo_query_selection_sweep: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "best_summary": {},
    }
    if args.repo_query_selection_sweep_dir:
        rq_dir = Path(args.repo_query_selection_sweep_dir)
        repo_query_selection_sweep["enabled"] = True
        repo_query_selection_sweep["source_dir"] = str(rq_dir)
        dst_root = out_dir / "repo_query_selection"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            rq_dir / "aggregate" / "metrics_by_policy_budget.csv",
            rq_dir / "aggregate" / "metrics_by_policy_budget.md",
            rq_dir / "best_report.md",
            rq_dir / "snapshot.json",
            rq_dir / "figures" / "fig_repo_query_selection_quality_vs_budget.png",
            rq_dir / "figures" / "fig_repo_query_selection_quality_vs_budget.pdf",
            rq_dir / "figures" / "fig_repo_query_selection_distractor_vs_budget.png",
            rq_dir / "figures" / "fig_repo_query_selection_distractor_vs_budget.pdf",
            rq_dir / "figures" / "fig_repo_query_selection_chunks_by_level.png",
            rq_dir / "figures" / "fig_repo_query_selection_chunks_by_level.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        repo_query_selection_sweep["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        snap_src = rq_dir / "snapshot.json"
        if snap_src.exists():
            try:
                snap_payload = json.loads(snap_src.read_text(encoding="utf-8"))
                if isinstance(snap_payload, dict):
                    best = snap_payload.get("outputs", {}).get("best", {})
                    baseline = snap_payload.get("outputs", {}).get("baseline_best", {})
                    if isinstance(best, dict):
                        repo_query_selection_sweep["best_summary"]["best"] = best
                    if isinstance(baseline, dict):
                        repo_query_selection_sweep["best_summary"]["baseline"] = baseline
            except Exception:
                pass

    component_attribution: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "summary": {},
    }
    if args.component_attribution_dir:
        ca_dir = Path(args.component_attribution_dir)
        component_attribution["enabled"] = True
        component_attribution["source_dir"] = str(ca_dir)
        dst_root = out_dir / "component_attribution"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            ca_dir / "tables" / "table_component_attribution.csv",
            ca_dir / "tables" / "table_component_attribution.md",
            ca_dir / "compare_summary.json",
            ca_dir / "snapshot.json",
            ca_dir / "figures" / "fig_component_attribution_delta.png",
            ca_dir / "figures" / "fig_component_attribution_delta.pdf",
            ca_dir / "figures" / "fig_component_attribution_tradeoff.png",
            ca_dir / "figures" / "fig_component_attribution_tradeoff.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        component_attribution["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        summary_src = ca_dir / "compare_summary.json"
        if summary_src.exists():
            try:
                payload = json.loads(summary_src.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    component_attribution["summary"] = dict(payload.get("summary", payload))
            except Exception:
                pass

    bye_report_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "summary": {},
    }
    if args.bye_report_compare_dir:
        br_dir = Path(args.bye_report_compare_dir)
        bye_report_compare["enabled"] = True
        bye_report_compare["source_dir"] = str(br_dir)
        dst_root = out_dir / "bye_report"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            br_dir / "tables" / "table_bye_report_compare.csv",
            br_dir / "tables" / "table_bye_report_compare.md",
            br_dir / "compare_summary.json",
            br_dir / "figures" / "fig_bye_critical_fn_delta.png",
            br_dir / "figures" / "fig_bye_critical_fn_delta.pdf",
            br_dir / "figures" / "fig_bye_latency_delta.png",
            br_dir / "figures" / "fig_bye_latency_delta.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        bye_report_compare["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        summary_src = br_dir / "compare_summary.json"
        if summary_src.exists():
            try:
                payload = json.loads(summary_src.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    bye_report_compare["summary"] = dict(payload)
            except Exception:
                pass

    decisions_backend_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "summary": {},
    }
    if args.decisions_backend_compare_dir:
        db_dir = Path(args.decisions_backend_compare_dir)
        decisions_backend_compare["enabled"] = True
        decisions_backend_compare["source_dir"] = str(db_dir)
        dst_root = out_dir / "decisions_backend"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            db_dir / "tables" / "table_decisions_backend_compare.csv",
            db_dir / "tables" / "table_decisions_backend_compare.md",
            db_dir / "compare_summary.json",
            db_dir / "figures" / "fig_decisions_backend_delta.png",
            db_dir / "figures" / "fig_decisions_backend_delta.pdf",
            db_dir / "figures" / "fig_decisions_backend_tradeoff.png",
            db_dir / "figures" / "fig_decisions_backend_tradeoff.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        decisions_backend_compare["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        summary_src = db_dir / "compare_summary.json"
        if summary_src.exists():
            try:
                payload = json.loads(summary_src.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    decisions_backend_compare["summary"] = dict(payload)
            except Exception:
                pass

    planner_backend_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "summary": {},
    }
    if args.planner_backend_compare_dir:
        pb_dir = Path(args.planner_backend_compare_dir)
        planner_backend_compare["enabled"] = True
        planner_backend_compare["source_dir"] = str(pb_dir)
        dst_root = out_dir / "planner_backend"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            pb_dir / "tables" / "table_planner_backend_compare.csv",
            pb_dir / "tables" / "table_planner_backend_compare.md",
            pb_dir / "compare_summary.json",
            pb_dir / "figures" / "fig_planner_backend_delta.png",
            pb_dir / "figures" / "fig_planner_backend_delta.pdf",
            pb_dir / "figures" / "fig_planner_backend_tradeoff.png",
            pb_dir / "figures" / "fig_planner_backend_tradeoff.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        planner_backend_compare["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        summary_src = pb_dir / "compare_summary.json"
        if summary_src.exists():
            try:
                payload = json.loads(summary_src.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    planner_backend_compare["summary"] = dict(payload)
            except Exception:
                pass

    model_stack_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
        "summary": {},
    }
    if args.model_stack_compare_dir:
        ms_dir = Path(args.model_stack_compare_dir)
        model_stack_compare["enabled"] = True
        model_stack_compare["source_dir"] = str(ms_dir)
        dst_root = out_dir / "model_stack"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            ms_dir / "tables" / "table_model_stack_compare.csv",
            ms_dir / "tables" / "table_model_stack_compare.md",
            ms_dir / "compare_summary.json",
            ms_dir / "figures" / "fig_model_stack_delta.png",
            ms_dir / "figures" / "fig_model_stack_delta.pdf",
            ms_dir / "figures" / "fig_model_stack_tradeoff.png",
            ms_dir / "figures" / "fig_model_stack_tradeoff.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        model_stack_compare["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))
        summary_src = ms_dir / "compare_summary.json"
        if summary_src.exists():
            try:
                payload = json.loads(summary_src.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    model_stack_compare["summary"] = dict(payload)
            except Exception:
                pass

    model_cost_compare: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    resolved_model_cost_dir: Path | None = None
    if args.model_cost_compare_dir:
        resolved_model_cost_dir = Path(args.model_cost_compare_dir)
    elif args.model_stack_compare_dir:
        resolved_model_cost_dir = Path(args.model_stack_compare_dir)
    if resolved_model_cost_dir:
        model_cost_compare["enabled"] = True
        model_cost_compare["source_dir"] = str(resolved_model_cost_dir)
        dst_root = out_dir / "model_stack"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            resolved_model_cost_dir / "tables" / "table_model_cost_compare.csv",
            resolved_model_cost_dir / "tables" / "table_model_cost_compare.md",
            resolved_model_cost_dir / "figures" / "fig_model_cost_vs_quality.png",
            resolved_model_cost_dir / "figures" / "fig_model_cost_vs_quality.pdf",
            resolved_model_cost_dir / "figures" / "fig_model_parse_fail_rate.png",
            resolved_model_cost_dir / "figures" / "fig_model_parse_fail_rate.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        model_cost_compare["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    lost_object_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.lost_object_panel_dir:
        lo_dir = Path(args.lost_object_panel_dir)
        lost_object_panel["enabled"] = True
        lost_object_panel["source_dir"] = str(lo_dir)
        dst_root = out_dir / "lost_object_panel"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            lo_dir / "table_lost_object_budget.csv",
            lo_dir / "table_lost_object_budget.md",
            lo_dir / "figures" / "fig_lost_object_budget.png",
            lo_dir / "figures" / "fig_lost_object_budget.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.parent.name == "figures":
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        lost_object_panel["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    chain_nlq_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.chain_nlq_dir:
        ch_dir = Path(args.chain_nlq_dir)
        chain_nlq_panel["enabled"] = True
        chain_nlq_panel["source_dir"] = str(ch_dir)
        dst_root = out_dir / "chain_nlq_panel"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            ch_dir / "table_chain_summary.csv",
            ch_dir / "table_chain_summary.md",
            ch_dir / "table_chain_failure_attribution.csv",
            ch_dir / "table_chain_failure_attribution.md",
            ch_dir / "fig_chain_success_vs_budget_seconds.png",
            ch_dir / "fig_chain_success_vs_budget_seconds.pdf",
            ch_dir / "fig_chain_failure_attribution_vs_budget_seconds.png",
            ch_dir / "fig_chain_failure_attribution_vs_budget_seconds.pdf",
            ch_dir / "fig_chain_success_vs_derive.png",
            ch_dir / "fig_chain_success_vs_derive.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.suffix.lower() in {".png", ".pdf"}:
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        chain_nlq_panel["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    chain_repo_compare_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.chain_repo_compare_dir:
        cr_dir = Path(args.chain_repo_compare_dir)
        chain_repo_compare_panel["enabled"] = True
        chain_repo_compare_panel["source_dir"] = str(cr_dir)
        dst_root = out_dir / "chain_repo_compare"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            cr_dir / "tables" / "table_chain_repo_compare.csv",
            cr_dir / "tables" / "table_chain_repo_compare.md",
            cr_dir / "compare_summary.json",
            cr_dir / "snapshot.json",
            cr_dir / "figures" / "fig_chain_repo_compare_success_vs_budget_seconds.png",
            cr_dir / "figures" / "fig_chain_repo_compare_success_vs_budget_seconds.pdf",
            cr_dir / "figures" / "fig_chain_repo_compare_delta.png",
            cr_dir / "figures" / "fig_chain_repo_compare_delta.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.suffix.lower() in {".png", ".pdf"}:
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        chain_repo_compare_panel["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    chain_attribution_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.chain_attribution_dir:
        ca_dir = Path(args.chain_attribution_dir)
        chain_attribution_panel["enabled"] = True
        chain_attribution_panel["source_dir"] = str(ca_dir)
        dst_root = out_dir / "chain_attribution"
        dst_root.mkdir(parents=True, exist_ok=True)
        to_copy = [
            ca_dir / "tables" / "table_chain_attribution.csv",
            ca_dir / "tables" / "table_chain_attribution.md",
            ca_dir / "tables" / "table_chain_failure_breakdown.csv",
            ca_dir / "tables" / "table_chain_failure_breakdown.md",
            ca_dir / "compare_summary.json",
            ca_dir / "snapshot.json",
            ca_dir / "figures" / "fig_chain_attribution_success_vs_budget_seconds.png",
            ca_dir / "figures" / "fig_chain_attribution_success_vs_budget_seconds.pdf",
            ca_dir / "figures" / "fig_chain_attribution_delta_success_vs_budget_seconds.png",
            ca_dir / "figures" / "fig_chain_attribution_delta_success_vs_budget_seconds.pdf",
            ca_dir / "figures" / "fig_chain_attribution_failure_attribution_vs_budget_seconds.png",
            ca_dir / "figures" / "fig_chain_attribution_failure_attribution_vs_budget_seconds.pdf",
            ca_dir / "figures" / "fig_chain_attribution_tradeoff.png",
            ca_dir / "figures" / "fig_chain_attribution_tradeoff.pdf",
            ca_dir / "figures" / "fig_chain_attribution_backoff_vs_budget_seconds.png",
            ca_dir / "figures" / "fig_chain_attribution_backoff_vs_budget_seconds.pdf",
        ]
        copied: list[str] = []
        for src in to_copy:
            if not src.exists():
                continue
            if src.suffix.lower() in {".png", ".pdf"}:
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        chain_attribution_panel["copied_files"] = copied
        for p in copied:
            if str(p).endswith(".png") or str(p).endswith(".pdf"):
                figure_paths.append(str(p))

    resolved_signal_selection_dir: Path | None = None
    if args.signal_selection_dir:
        resolved_signal_selection_dir = Path(args.signal_selection_dir)
    elif compare_dir:
        inferred = compare_dir / "selection"
        if inferred.exists():
            resolved_signal_selection_dir = inferred

    signal_selection_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if resolved_signal_selection_dir:
        ss_dir = resolved_signal_selection_dir
        signal_selection_panel["enabled"] = True
        signal_selection_panel["source_dir"] = str(ss_dir)
        dst_root = out_dir / "selection"
        dst_root.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for name in ("coverage.md", "selection_report.md", "selected_uids.txt", "coverage.csv", "snapshot.json"):
            cp = _copy_if_exists(ss_dir / name, dst_root / name)
            if cp:
                copied.append(cp)
        signal_selection_panel["copied_files"] = copied

    significance_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    if args.significance_dir:
        sig_dir = Path(args.significance_dir)
        significance_panel["enabled"] = True
        significance_panel["source_dir"] = str(sig_dir)
        dst_root = out_dir / "significance"
        dst_root.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for src in (
            sig_dir / "tables" / "table_significance_main.csv",
            sig_dir / "tables" / "table_significance_main.md",
            sig_dir / "tables" / "table_confidence_intervals.csv",
            sig_dir / "tables" / "table_confidence_intervals.md",
            sig_dir / "figures" / "fig_significance_delta_vs_budget_seconds.png",
            sig_dir / "figures" / "fig_significance_delta_vs_budget_seconds.pdf",
            sig_dir / "figures" / "fig_effect_size_forest.png",
            sig_dir / "figures" / "fig_effect_size_forest.pdf",
            sig_dir / "report.md",
            sig_dir / "snapshot.json",
        ):
            if src.suffix.lower() in {".png", ".pdf"}:
                dst = figures_dir / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
                if str(cp).endswith(".png") or str(cp).endswith(".pdf"):
                    figure_paths.append(str(cp))
        significance_panel["copied_files"] = copied

    resolved_result_health_dir: Path | None = None
    if args.result_health_dir:
        resolved_result_health_dir = Path(args.result_health_dir)
    elif args.suite_dir:
        candidate = Path(args.suite_dir) / "result_health"
        if candidate.exists():
            resolved_result_health_dir = candidate
    result_health_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    result_health_snapshot: dict[str, Any] = {}
    if resolved_result_health_dir:
        result_health_panel["enabled"] = True
        result_health_panel["source_dir"] = str(resolved_result_health_dir)
        dst_root = out_dir / "result_health"
        copied: list[str] = []
        for src in (
            resolved_result_health_dir / "tables" / "table_result_health.csv",
            resolved_result_health_dir / "tables" / "table_result_health.md",
            resolved_result_health_dir / "figures" / "fig_result_health_breakdown.png",
            resolved_result_health_dir / "figures" / "fig_result_health_breakdown.pdf",
            resolved_result_health_dir / "snapshot.json",
        ):
            if src.suffix.lower() in {".png", ".pdf"}:
                dst = dst_root / "figures" / src.name
            elif src.suffix.lower() in {".csv", ".md"}:
                dst = dst_root / "tables" / src.name
            else:
                dst = dst_root / src.name
            cp = _copy_if_exists(src, dst)
            if cp:
                copied.append(cp)
        result_health_panel["copied_files"] = copied
        snapshot_src = resolved_result_health_dir / "snapshot.json"
        if snapshot_src.exists():
            try:
                result_health_snapshot = json.loads(snapshot_src.read_text(encoding="utf-8"))
            except Exception:
                result_health_snapshot = {}

    resolved_freeze_dir: Path | None = None
    if args.benchmark_freeze_dir:
        resolved_freeze_dir = Path(args.benchmark_freeze_dir)
    elif args.suite_dir:
        candidate = Path(args.suite_dir) / "freeze"
        if candidate.exists():
            resolved_freeze_dir = candidate
    benchmark_freeze_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "copied_files": [],
    }
    freeze_manifest_payload: dict[str, Any] = {}
    if resolved_freeze_dir:
        benchmark_freeze_panel["enabled"] = True
        benchmark_freeze_panel["source_dir"] = str(resolved_freeze_dir)
        dst_root = out_dir / "freeze"
        copied: list[str] = []
        for src in (
            resolved_freeze_dir / "freeze_manifest.json",
            resolved_freeze_dir / "artifacts_sha256.csv",
        ):
            cp = _copy_if_exists(src, dst_root / src.name)
            if cp:
                copied.append(cp)
        benchmark_freeze_panel["copied_files"] = copied
        freeze_src = resolved_freeze_dir / "freeze_manifest.json"
        if freeze_src.exists():
            try:
                freeze_manifest_payload = json.loads(freeze_src.read_text(encoding="utf-8"))
            except Exception:
                freeze_manifest_payload = {}

    suite_provenance_panel: dict[str, Any] = {
        "enabled": False,
        "source_dir": None,
        "manifest_files": [],
        "provenance_files": [],
    }
    if args.suite_dir:
        suite_dir = Path(args.suite_dir)
        suite_provenance_panel["enabled"] = True
        suite_provenance_panel["source_dir"] = str(suite_dir)
        manifest_dst = out_dir / "manifest"
        provenance_dst = out_dir / "provenance"
        manifest_dst.mkdir(parents=True, exist_ok=True)
        provenance_dst.mkdir(parents=True, exist_ok=True)
        manifest_copied: list[str] = []
        provenance_copied: list[str] = []
        for src in (
            suite_dir / "manifest" / "experiment_manifest.yaml",
            suite_dir / "manifest" / "manifest_resolved.json",
            suite_dir / "manifest" / "prompt_lock.json",
            suite_dir / "manifest" / "query_bank_lock.json",
        ):
            cp = _copy_if_exists(src, manifest_dst / src.name)
            if cp:
                manifest_copied.append(cp)
        query_banks_src = suite_dir / "manifest" / "query_banks"
        if query_banks_src.exists():
            target = manifest_dst / "query_banks"
            target.mkdir(parents=True, exist_ok=True)
            for src in query_banks_src.glob("*"):
                if not src.is_file():
                    continue
                cp = _copy_if_exists(src, target / src.name)
                if cp:
                    manifest_copied.append(cp)
        for src in (
            suite_dir / "ledger" / "results_long.csv",
            suite_dir / "ledger" / "runs.jsonl",
            suite_dir / "compare" / "commands.sh",
            suite_dir / "compare" / "compare_summary.json",
            suite_dir / "compare" / "snapshot.json",
            suite_dir / "compare" / "README.md",
        ):
            target_name = "compare_snapshot.json" if src.name == "snapshot.json" else src.name
            cp = _copy_if_exists(src, provenance_dst / target_name)
            if cp:
                provenance_copied.append(cp)
        suite_provenance_panel["manifest_files"] = manifest_copied
        suite_provenance_panel["provenance_files"] = provenance_copied

    prompt_panel: dict[str, Any] = {
        "registry": None,
        "prompt_lock": None,
        "copied_files": [],
    }
    if args.prompt_registry or args.prompt_lock:
        prompt_dst = out_dir / "prompts"
        prompt_dst.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        if args.prompt_registry:
            registry_src = Path(args.prompt_registry)
            cp = _copy_if_exists(registry_src, prompt_dst / registry_src.name)
            if cp:
                prompt_panel["registry"] = cp
                copied.append(cp)
        if args.prompt_lock:
            lock_src = Path(args.prompt_lock)
            cp = _copy_if_exists(lock_src, (out_dir / "manifest") / "prompt_lock.json")
            if cp:
                prompt_panel["prompt_lock"] = cp
                copied.append(cp)
        prompt_panel["copied_files"] = copied

    submission_pack_dir = Path(args.submission_pack_dir) if args.submission_pack_dir else (out_dir / "submission_pack")
    if bool(args.export_submission_pack):
        report_lines_submission = f"- submission_pack_dir: `{submission_pack_dir}`"
    else:
        report_lines_submission = "- submission_pack_dir: `disabled`"

    report_path = out_dir / "report.md"
    if safety_present:
        safety_line = (
            "- NLQ safety metrics detected: denominator uses `safety_count_granularity`; "
            "failure attribution uses budget_insufficient/evidence_missing/constraints_over_filtered/"
            "retrieval_distractor, remaining reasons are merged into `other`."
        )
    else:
        safety_line = "- NLQ safety metrics missing."
    report_lines = [
        "# Paper-ready Unified Budget Panel",
        "",
        f"- compare_dir: `{compare_dir}`",
        f"- labels: `{args.label_a}` vs `{args.label_b}`",
        f"- missing_tasks: `{missing_tasks}`",
        f"- primary_metrics: `{json.dumps(chosen_primary, ensure_ascii=False, sort_keys=True)}`",
        f"- recommend_points: `{json.dumps(recommend_points, ensure_ascii=False, sort_keys=True)}`",
        safety_line,
    ]
    if args.streaming_policy_compare_dir:
        if streaming_policy_compare.get("copied_tables") or streaming_policy_compare.get("copied_figures"):
            report_lines.extend(
                [
                    f"- streaming_policy_compare_dir: `{streaming_policy_compare.get('source_dir')}`",
                    f"- streaming_policy_compare_tables: `{streaming_policy_compare.get('copied_tables')}`",
                    f"- streaming_policy_compare_figures: `{streaming_policy_compare.get('copied_figures')}`",
                ]
            )
        else:
            report_lines.append("- streaming_policy_compare: source provided but artifacts missing.")
    if args.streaming_chain_backoff_compare_dir:
        if streaming_chain_backoff_compare.get("copied_tables") or streaming_chain_backoff_compare.get("copied_figures"):
            report_lines.extend(
                [
                    f"- streaming_chain_backoff_compare_dir: `{streaming_chain_backoff_compare.get('source_dir')}`",
                    f"- streaming_chain_backoff_compare_tables: `{streaming_chain_backoff_compare.get('copied_tables')}`",
                    f"- streaming_chain_backoff_compare_figures: `{streaming_chain_backoff_compare.get('copied_figures')}`",
                ]
            )
        else:
            report_lines.append("- streaming_chain_backoff_compare: source provided but artifacts missing.")
    if args.streaming_repo_compare_dir:
        if streaming_repo_compare.get("copied_tables") or streaming_repo_compare.get("copied_figures"):
            report_lines.extend(
                [
                    f"- streaming_repo_compare_dir: `{streaming_repo_compare.get('source_dir')}`",
                    f"- streaming_repo_compare_tables: `{streaming_repo_compare.get('copied_tables')}`",
                    f"- streaming_repo_compare_figures: `{streaming_repo_compare.get('copied_figures')}`",
                ]
            )
        else:
            report_lines.append("- streaming_repo_compare: source provided but artifacts missing.")
    if args.streaming_intervention_sweep_dir:
        if streaming_intervention_sweep.get("copied_files"):
            report_lines.extend(
                [
                    f"- streaming_intervention_sweep_dir: `{streaming_intervention_sweep.get('source_dir')}`",
                    f"- streaming_intervention_sweep_files: `{streaming_intervention_sweep.get('copied_files')}`",
                    f"- streaming_intervention_best: `{json.dumps(streaming_intervention_sweep.get('best_summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- streaming_intervention_sweep: source provided but artifacts missing.")
    if args.reranker_sweep_dir:
        if reranker_sweep.get("copied_files"):
            report_lines.extend(
                [
                    f"- reranker_sweep_dir: `{reranker_sweep.get('source_dir')}`",
                    f"- reranker_sweep_files: `{reranker_sweep.get('copied_files')}`",
                    f"- reranker_best: `{json.dumps(reranker_sweep.get('best_summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- reranker_sweep: source provided but artifacts missing.")
    if args.streaming_codec_sweep_dir:
        if streaming_codec_sweep.get("copied_files"):
            report_lines.extend(
                [
                    f"- streaming_codec_sweep_dir: `{streaming_codec_sweep.get('source_dir')}`",
                    f"- streaming_codec_sweep_files: `{streaming_codec_sweep.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- streaming_codec_sweep: source provided but artifacts missing.")
    if args.repo_policy_sweep_dir:
        if repo_policy_sweep.get("copied_files"):
            report_lines.extend(
                [
                    f"- repo_policy_sweep_dir: `{repo_policy_sweep.get('source_dir')}`",
                    f"- repo_policy_sweep_files: `{repo_policy_sweep.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- repo_policy_sweep: source provided but artifacts missing.")
    if args.repo_summary_sweep_dir:
        if repo_summary_sweep.get("copied_files"):
            report_lines.extend(
                [
                    f"- repo_summary_sweep_dir: `{repo_summary_sweep.get('source_dir')}`",
                    f"- repo_summary_sweep_files: `{repo_summary_sweep.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- repo_summary_sweep: source provided but artifacts missing.")
    if args.repo_summary_retrieval_compare_dir:
        if repo_summary_retrieval_compare.get("copied_files"):
            report_lines.extend(
                [
                    "",
                    "## Repo Summary Retrieval Compare",
                    f"- repo_summary_retrieval_compare_dir: `{repo_summary_retrieval_compare.get('source_dir')}`",
                    f"- repo_summary_retrieval_compare_files: `{repo_summary_retrieval_compare.get('copied_files')}`",
                    f"- repo_summary_retrieval_compare_summary: `{json.dumps(repo_summary_retrieval_compare.get('summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- repo_summary_retrieval_compare: source provided but artifacts missing.")
    if args.repo_query_selection_sweep_dir:
        if repo_query_selection_sweep.get("copied_files"):
            report_lines.extend(
                [
                    f"- repo_query_selection_sweep_dir: `{repo_query_selection_sweep.get('source_dir')}`",
                    f"- repo_query_selection_sweep_files: `{repo_query_selection_sweep.get('copied_files')}`",
                    f"- repo_query_selection_summary: `{json.dumps(repo_query_selection_sweep.get('best_summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- repo_query_selection_sweep: source provided but artifacts missing.")
    if args.component_attribution_dir:
        if component_attribution.get("copied_files"):
            report_lines.extend(
                [
                    f"- component_attribution_dir: `{component_attribution.get('source_dir')}`",
                    f"- component_attribution_files: `{component_attribution.get('copied_files')}`",
                    f"- component_attribution_summary: `{json.dumps(component_attribution.get('summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- component_attribution: source provided but artifacts missing.")
    if args.bye_report_compare_dir:
        if bye_report_compare.get("copied_files"):
            report_lines.extend(
                [
                    f"- bye_report_compare_dir: `{bye_report_compare.get('source_dir')}`",
                    f"- bye_report_compare_files: `{bye_report_compare.get('copied_files')}`",
                    f"- bye_report_compare_summary: `{json.dumps(bye_report_compare.get('summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- bye_report_compare: source provided but artifacts missing.")
    if args.decisions_backend_compare_dir:
        if decisions_backend_compare.get("copied_files"):
            report_lines.extend(
                [
                    "## Decisions Backend Compare",
                    "",
                    f"- decisions_backend_compare_dir: `{decisions_backend_compare.get('source_dir')}`",
                    f"- decisions_backend_compare_files: `{decisions_backend_compare.get('copied_files')}`",
                    f"- decisions_backend_compare_summary: `{json.dumps(decisions_backend_compare.get('summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- decisions_backend_compare: source provided but artifacts missing.")
    if args.planner_backend_compare_dir:
        if planner_backend_compare.get("copied_files"):
            report_lines.extend(
                [
                    "## Planner Backend Compare",
                    "",
                    f"- planner_backend_compare_dir: `{planner_backend_compare.get('source_dir')}`",
                    f"- planner_backend_compare_files: `{planner_backend_compare.get('copied_files')}`",
                    f"- planner_backend_compare_summary: `{json.dumps(planner_backend_compare.get('summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- planner_backend_compare: source provided but artifacts missing.")
    if args.model_stack_compare_dir:
        if model_stack_compare.get("copied_files"):
            report_lines.extend(
                [
                    "## Model Stack Compare",
                    "",
                    f"- model_stack_compare_dir: `{model_stack_compare.get('source_dir')}`",
                    f"- model_stack_compare_files: `{model_stack_compare.get('copied_files')}`",
                    f"- model_stack_compare_summary: `{json.dumps(model_stack_compare.get('summary', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- model_stack_compare: source provided but artifacts missing.")
    if resolved_model_cost_dir:
        if model_cost_compare.get("copied_files"):
            report_lines.extend(
                [
                    "## Model Cost and Structured Reliability",
                    "",
                    f"- model_cost_compare_dir: `{model_cost_compare.get('source_dir')}`",
                    f"- model_cost_compare_files: `{model_cost_compare.get('copied_files')}`",
                    "- includes cost_usd/latency/parse-fail telemetry panel for model-backed runs.",
                ]
            )
        else:
            report_lines.append("- model_cost_compare: source provided but artifacts missing.")
    if args.lost_object_panel_dir:
        if lost_object_panel.get("copied_files"):
            report_lines.extend(
                [
                    f"- lost_object_panel_dir: `{lost_object_panel.get('source_dir')}`",
                    f"- lost_object_panel_files: `{lost_object_panel.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- lost_object_panel: source provided but artifacts missing.")
    if args.chain_nlq_dir:
        if chain_nlq_panel.get("copied_files"):
            report_lines.extend(
                [
                    f"- chain_nlq_dir: `{chain_nlq_panel.get('source_dir')}`",
                    f"- chain_nlq_files: `{chain_nlq_panel.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- chain_nlq_panel: source provided but artifacts missing.")
    if args.chain_repo_compare_dir:
        if chain_repo_compare_panel.get("copied_files"):
            report_lines.extend(
                [
                    f"- chain_repo_compare_dir: `{chain_repo_compare_panel.get('source_dir')}`",
                    f"- chain_repo_compare_files: `{chain_repo_compare_panel.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- chain_repo_compare: source provided but artifacts missing.")
    if args.chain_attribution_dir:
        if chain_attribution_panel.get("copied_files"):
            report_lines.extend(
                [
                    f"- chain_attribution_dir: `{chain_attribution_panel.get('source_dir')}`",
                    f"- chain_attribution_files: `{chain_attribution_panel.get('copied_files')}`",
                    "- chain_attribution_backoff_metrics: available when table contains backoff_used_rate/backoff_mean_level/backoff_exhausted_rate.",
                ]
            )
        else:
            report_lines.append("- chain_attribution: source provided but artifacts missing.")
    if resolved_signal_selection_dir:
        if signal_selection_panel.get("copied_files"):
            report_lines.extend(
                [
                    "## Signal Coverage of Chosen UIDs",
                    "",
                    f"- signal_selection_dir: `{signal_selection_panel.get('source_dir')}`",
                    f"- signal_selection_files: `{signal_selection_panel.get('copied_files')}`",
                    "- this panel documents why selected UIDs carry non-empty place/interaction/object signals for AB plots.",
                ]
            )
        else:
            report_lines.append("- signal_selection: source provided but artifacts missing.")
    if args.significance_dir:
        if significance_panel.get("copied_files"):
            report_lines.extend(
                [
                    "## Statistical Significance",
                    "",
                    f"- significance_dir: `{significance_panel.get('source_dir')}`",
                    f"- significance_files: `{significance_panel.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- significance: source provided but artifacts missing.")
    if resolved_result_health_dir:
        if result_health_panel.get("copied_files"):
            report_lines.extend(
                [
                    "## Result Health",
                    "",
                    f"- result_health_dir: `{result_health_panel.get('source_dir')}`",
                    f"- result_health_files: `{result_health_panel.get('copied_files')}`",
                    f"- current_run_not_empty_reason: `{json.dumps(result_health_snapshot.get('overall_no_data_reason_counts', {}), ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- result_health: source provided but artifacts missing.")
    if resolved_freeze_dir:
        if benchmark_freeze_panel.get("copied_files"):
            report_lines.extend(
                [
                    "## Benchmark Freeze",
                    "",
                    f"- benchmark_freeze_dir: `{benchmark_freeze_panel.get('source_dir')}`",
                    f"- benchmark_freeze_files: `{benchmark_freeze_panel.get('copied_files')}`",
                    f"- current_run_frozen: `{json.dumps(freeze_manifest_payload, ensure_ascii=False, sort_keys=True)}`",
                ]
            )
        else:
            report_lines.append("- benchmark_freeze: source provided but artifacts missing.")
    if args.suite_dir:
        if suite_provenance_panel.get("manifest_files") or suite_provenance_panel.get("provenance_files"):
            report_lines.extend(
                [
                    "## Suite Provenance",
                    "",
                    f"- suite_dir: `{suite_provenance_panel.get('source_dir')}`",
                    f"- suite_manifest_files: `{suite_provenance_panel.get('manifest_files')}`",
                    f"- suite_provenance_files: `{suite_provenance_panel.get('provenance_files')}`",
                ]
            )
        else:
            report_lines.append("- suite_provenance: source provided but artifacts missing.")
    if args.prompt_registry or args.prompt_lock:
        if prompt_panel.get("copied_files"):
            report_lines.extend(
                [
                    "## Prompt Provenance",
                    "",
                    f"- prompt_registry: `{args.prompt_registry}`",
                    f"- prompt_lock: `{args.prompt_lock}`",
                    f"- prompt_files: `{prompt_panel.get('copied_files')}`",
                ]
            )
        else:
            report_lines.append("- prompt_provenance: source provided but artifacts missing.")
    report_lines.extend(
        [
        "",
        "## Artifacts",
        "",
        f"- panel table: `{panel_csv}`",
        f"- delta table: `{delta_csv}`",
        f"- figures: `{figure_paths}`",
        report_lines_submission,
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    snapshot_path = out_dir / "snapshot.json"
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "compare_dir": str(compare_dir),
            "label_a": str(args.label_a),
            "label_b": str(args.label_b),
            "primary_metrics_override": primary_override,
            "format": str(args.format),
            "with_figs": bool(args.with_figs),
            "formats": formats,
            "streaming_policy_compare_dir": str(args.streaming_policy_compare_dir)
            if args.streaming_policy_compare_dir
            else None,
            "streaming_chain_backoff_compare_dir": str(args.streaming_chain_backoff_compare_dir)
            if args.streaming_chain_backoff_compare_dir
            else None,
            "streaming_repo_compare_dir": str(args.streaming_repo_compare_dir)
            if args.streaming_repo_compare_dir
            else None,
            "streaming_intervention_sweep_dir": str(args.streaming_intervention_sweep_dir)
            if args.streaming_intervention_sweep_dir
            else None,
            "streaming_codec_sweep_dir": str(args.streaming_codec_sweep_dir)
            if args.streaming_codec_sweep_dir
            else None,
            "reranker_sweep_dir": str(args.reranker_sweep_dir) if args.reranker_sweep_dir else None,
            "repo_policy_sweep_dir": str(args.repo_policy_sweep_dir) if args.repo_policy_sweep_dir else None,
            "repo_summary_sweep_dir": str(args.repo_summary_sweep_dir) if args.repo_summary_sweep_dir else None,
            "repo_summary_retrieval_compare_dir": str(args.repo_summary_retrieval_compare_dir)
            if args.repo_summary_retrieval_compare_dir
            else None,
            "repo_query_selection_sweep_dir": str(args.repo_query_selection_sweep_dir)
            if args.repo_query_selection_sweep_dir
            else None,
            "component_attribution_dir": str(args.component_attribution_dir) if args.component_attribution_dir else None,
            "bye_report_compare_dir": str(args.bye_report_compare_dir) if args.bye_report_compare_dir else None,
            "decisions_backend_compare_dir": str(args.decisions_backend_compare_dir)
            if args.decisions_backend_compare_dir
            else None,
            "planner_backend_compare_dir": str(args.planner_backend_compare_dir)
            if args.planner_backend_compare_dir
            else None,
            "model_stack_compare_dir": str(args.model_stack_compare_dir)
            if args.model_stack_compare_dir
            else None,
            "model_cost_compare_dir": str(resolved_model_cost_dir) if resolved_model_cost_dir else None,
            "lost_object_panel_dir": str(args.lost_object_panel_dir) if args.lost_object_panel_dir else None,
            "chain_nlq_dir": str(args.chain_nlq_dir) if args.chain_nlq_dir else None,
            "chain_repo_compare_dir": str(args.chain_repo_compare_dir) if args.chain_repo_compare_dir else None,
            "chain_attribution_dir": str(args.chain_attribution_dir) if args.chain_attribution_dir else None,
            "signal_selection_dir": str(resolved_signal_selection_dir) if resolved_signal_selection_dir else None,
            "suite_dir": str(args.suite_dir) if args.suite_dir else None,
            "significance_dir": str(args.significance_dir) if args.significance_dir else None,
            "result_health_dir": str(resolved_result_health_dir) if resolved_result_health_dir else None,
            "benchmark_freeze_dir": str(resolved_freeze_dir) if resolved_freeze_dir else None,
            "prompt_registry": str(args.prompt_registry) if args.prompt_registry else None,
            "prompt_lock": str(args.prompt_lock) if args.prompt_lock else None,
            "export_submission_pack": bool(args.export_submission_pack),
            "submission_pack_dir": str(submission_pack_dir) if bool(args.export_submission_pack) else None,
        },
        "sources": {
            task: {side: str(path) for side, path in side_paths.items()}
            for task, side_paths in task_sources.items()
        },
        "missing_tasks": missing_tasks,
        "primary_metrics": chosen_primary,
        "outputs": {
            "table_budget_panel_csv": str(panel_csv),
            "table_budget_panel_md": str(panel_md),
            "table_budget_panel_delta_csv": str(delta_csv),
            "table_budget_panel_delta_md": str(delta_md),
            "figures": figure_paths,
            "safety_figures": safety_figure_paths,
            "streaming_policy_compare": streaming_policy_compare,
            "streaming_chain_backoff_compare": streaming_chain_backoff_compare,
            "streaming_repo_compare": streaming_repo_compare,
            "streaming_intervention_sweep": streaming_intervention_sweep,
            "streaming_codec_sweep": streaming_codec_sweep,
            "reranker_sweep": reranker_sweep,
            "repo_policy_sweep": repo_policy_sweep,
            "repo_summary_sweep": repo_summary_sweep,
            "repo_summary_retrieval_compare": repo_summary_retrieval_compare,
            "repo_query_selection_sweep": repo_query_selection_sweep,
            "component_attribution": component_attribution,
            "bye_report_compare": bye_report_compare,
            "decisions_backend_compare": decisions_backend_compare,
            "planner_backend_compare": planner_backend_compare,
            "model_stack_compare": model_stack_compare,
            "model_cost_compare": model_cost_compare,
            "lost_object_panel": lost_object_panel,
            "chain_nlq_panel": chain_nlq_panel,
            "chain_repo_compare_panel": chain_repo_compare_panel,
            "chain_attribution_panel": chain_attribution_panel,
            "signal_selection_panel": signal_selection_panel,
            "significance_panel": significance_panel,
            "result_health_panel": result_health_panel,
            "benchmark_freeze_panel": benchmark_freeze_panel,
            "suite_provenance_panel": suite_provenance_panel,
            "prompt_panel": prompt_panel,
            "report_md": str(report_path),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    submission_pack_saved: str | None = None
    if bool(args.export_submission_pack):
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "export_submission_pack.py"),
            "--paper-ready-dir",
            str(out_dir),
            "--out-dir",
            str(submission_pack_dir),
        ]
        if args.suite_dir:
            cmd.extend(["--suite-dir", str(args.suite_dir)])
        if args.significance_dir:
            cmd.extend(["--significance-dir", str(args.significance_dir)])
        if resolved_result_health_dir:
            cmd.extend(["--result-health-dir", str(resolved_result_health_dir)])
        if resolved_freeze_dir:
            cmd.extend(["--benchmark-freeze-dir", str(resolved_freeze_dir)])
        if args.prompt_registry:
            cmd.extend(["--prompt-registry", str(args.prompt_registry)])
        if args.prompt_lock:
            cmd.extend(["--prompt-lock", str(args.prompt_lock)])
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
            return int(proc.returncode)
        submission_pack_saved = str(submission_pack_dir)
        snapshot.setdefault("outputs", {})
        snapshot["outputs"]["submission_pack_dir"] = submission_pack_saved
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_table_panel={panel_csv}")
    print(f"saved_table_delta={delta_csv}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    if submission_pack_saved:
        print(f"saved_submission_pack={submission_pack_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
