from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unified paper-ready budget panel across BYE/NLQ/Streaming")
    parser.add_argument("--compare_dir", required=True, help="AB compare directory")
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
        "--reranker-sweep-dir",
        default=None,
        help="Optional directory from sweep_reranker.py output",
    )
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
    compare_dir = Path(args.compare_dir)
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
        summary_src = spc_dir / "compare_summary.json"

        copied_table_csv = _copy_if_exists(table_src_csv, tables_dir / table_src_csv.name)
        copied_table_md = _copy_if_exists(table_src_md, tables_dir / table_src_md.name)
        copied_summary = _copy_if_exists(summary_src, out_dir / summary_src.name)
        copied_figures = [
            _copy_if_exists(fig_src_png, figures_dir / fig_src_png.name),
            _copy_if_exists(fig_src_pdf, figures_dir / fig_src_pdf.name),
            _copy_if_exists(fig_delta_png, figures_dir / fig_delta_png.name),
            _copy_if_exists(fig_delta_pdf, figures_dir / fig_delta_pdf.name),
        ]
        streaming_policy_compare["copied_tables"] = [x for x in [copied_table_csv, copied_table_md] if x]
        streaming_policy_compare["copied_figures"] = [x for x in copied_figures if x]
        streaming_policy_compare["copied_summary"] = copied_summary
        figure_paths.extend(streaming_policy_compare["copied_figures"])

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
    report_lines.extend(
        [
        "",
        "## Artifacts",
        "",
        f"- panel table: `{panel_csv}`",
        f"- delta table: `{delta_csv}`",
        f"- figures: `{figure_paths}`",
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
            "repo_query_selection_sweep_dir": str(args.repo_query_selection_sweep_dir)
            if args.repo_query_selection_sweep_dir
            else None,
            "component_attribution_dir": str(args.component_attribution_dir) if args.component_attribution_dir else None,
            "bye_report_compare_dir": str(args.bye_report_compare_dir) if args.bye_report_compare_dir else None,
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
            "streaming_repo_compare": streaming_repo_compare,
            "streaming_intervention_sweep": streaming_intervention_sweep,
            "streaming_codec_sweep": streaming_codec_sweep,
            "reranker_sweep": reranker_sweep,
            "repo_policy_sweep": repo_policy_sweep,
            "repo_query_selection_sweep": repo_query_selection_sweep,
            "component_attribution": component_attribution,
            "bye_report_compare": bye_report_compare,
            "report_md": str(report_path),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_table_panel={panel_csv}")
    print(f"saved_table_delta={delta_csv}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
