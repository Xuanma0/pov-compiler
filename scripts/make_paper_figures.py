from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pov_compiler.bench.reporting.aggregate import (
    aggregate_over_videos,
    compute_deltas,
    load_csvs,
    pick_budget_slice,
)
from pov_compiler.bench.reporting.latex import (
    build_ablation_table,
    build_main_table,
    df_to_latex_table,
    df_to_markdown_table,
)
from pov_compiler.utils.media import get_duration_bucket


VARIANT_ORDER = [
    "raw_events_only",
    "highlights_only",
    "highlights_plus_tokens",
    "highlights_plus_decisions",
    "full",
]
VARIANT_ORDER_FOCUS = [
    "highlights_only",
    "highlights_plus_tokens",
    "highlights_plus_decisions",
    "full",
]
QUERY_ORDER = [
    "hard_pseudo_anchor",
    "hard_pseudo_token",
    "hard_pseudo_decision",
    "pseudo_time",
    "pseudo_anchor",
    "pseudo_hard_time",
    "pseudo_token",
    "pseudo_decision",
    "time",
    "anchor",
    "hard_time",
    "token",
    "decision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready figures/tables from cross eval outputs")
    parser.add_argument("--cross_dir", required=True, help="Directory containing per-video eval subdirs")
    parser.add_argument("--nlq_csv", required=True, help="Path to nlq_summary_all.csv")
    parser.add_argument("--label", default="run", help="Primary run label for compare outputs")
    parser.add_argument("--compare_dir", default=None, help="Compare run output dir (reads compare_dir/snapshot.json)")
    parser.add_argument("--compare_cross_dir", default=None, help="Compare run cross_dir (low-level override)")
    parser.add_argument("--compare_nlq_csv", default=None, help="Compare run nlq_summary_all.csv (low-level override)")
    parser.add_argument("--compare_label", default="compare", help="Compare run label")
    parser.add_argument("--ablation_terms_csv", default=None, help="Optional table_ablation_terms.csv path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--budget",
        default="max_total_s=40,max_tokens=100,max_decisions=8",
        help="Budget selector, e.g. max_total_s=40,max_tokens=100,max_decisions=8",
    )
    parser.add_argument("--budget_max_total_s", type=float, default=None, help="Alias: budget max_total_s")
    parser.add_argument("--budget_max_tokens", type=float, default=None, help="Alias: budget max_tokens")
    parser.add_argument("--budget_max_decisions", type=float, default=None, help="Alias: budget max_decisions")
    parser.add_argument("--macro_avg", "--macro-avg", dest="macro_avg", action="store_true", help="Macro average over video_uid")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Figure formats, e.g. --formats png pdf")
    return parser.parse_args()


def _parse_budget(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            out[key] = float(value)
        except Exception:
            continue
    return out


def _parse_formats(text: str | list[str]) -> list[str]:
    if isinstance(text, list):
        return [str(x).strip().lower() for x in text if str(x).strip()] or ["png", "pdf"]
    out: list[str] = []
    for chunk in str(text).split(","):
        fmt = chunk.strip().lower()
        if not fmt:
            continue
        out.append(fmt)
    return out or ["png", "pdf"]


def _discover_cross_csvs(cross_dir: Path) -> tuple[list[Path], list[Path]]:
    overall_paths: list[Path] = []
    by_type_paths: list[Path] = []
    if (cross_dir / "results_overall.csv").exists():
        overall_paths.append(cross_dir / "results_overall.csv")
    if (cross_dir / "results_by_query_type.csv").exists():
        by_type_paths.append(cross_dir / "results_by_query_type.csv")
    overall_paths.extend(sorted(cross_dir.glob("*/results_overall.csv")))
    by_type_paths.extend(sorted(cross_dir.glob("*/results_by_query_type.csv")))
    return overall_paths, by_type_paths


def _resolve_path_hint(raw: str | None, *, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidate_local = (base_dir / path).resolve()
    if candidate_local.exists():
        return candidate_local
    candidate_root = (ROOT / path).resolve()
    if candidate_root.exists():
        return candidate_root
    return candidate_root


def _load_compare_snapshot(compare_dir: Path) -> dict[str, Any]:
    snapshot_path = compare_dir / "snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"compare_dir snapshot not found: {snapshot_path}")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid snapshot payload in {snapshot_path}")

    args_part = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
    inputs_part = payload.get("inputs", {}) if isinstance(payload.get("inputs", {}), dict) else {}
    return {
        "snapshot_path": snapshot_path,
        "cross_dir": args_part.get("cross_dir"),
        "nlq_csv": args_part.get("nlq_csv") or inputs_part.get("nlq_summary_all"),
        "budget_used": args_part.get("budget_used"),
        "macro_avg": args_part.get("macro_avg"),
    }


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _simple_group_mean(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, *metric_cols])
    use_metrics = [col for col in metric_cols if col in df.columns]
    if not use_metrics:
        return pd.DataFrame(columns=[*group_cols, *metric_cols])
    return df.groupby(group_cols, dropna=False)[use_metrics].mean().reset_index()


def _agg(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str], macro_avg: bool) -> pd.DataFrame:
    if macro_avg:
        return aggregate_over_videos(df, group_cols=group_cols, metric_cols=metric_cols, agg="mean")
    return _simple_group_mean(df, group_cols=group_cols, metric_cols=metric_cols)


def _weighted_query_mean(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["video_uid", "variant", "budget_max_total_s", metric_col])

    work = df.copy()
    work = _coerce_numeric(work, [metric_col, "num_queries", "budget_max_total_s"])
    if "video_uid" not in work.columns:
        if "video_id" in work.columns:
            work["video_uid"] = work["video_id"]
        else:
            work["video_uid"] = "video_0000"
    if "num_queries" not in work.columns:
        work["num_queries"] = 1.0
    work["num_queries"] = work["num_queries"].fillna(1.0).clip(lower=1.0)
    work[metric_col] = work[metric_col].fillna(0.0)
    work["weighted"] = work[metric_col] * work["num_queries"]
    grouped = (
        work.groupby(["video_uid", "variant", "budget_max_total_s"], dropna=False)[["weighted", "num_queries"]]
        .sum()
        .reset_index()
    )
    grouped[metric_col] = grouped["weighted"] / grouped["num_queries"].replace(0.0, 1.0)
    return grouped[["video_uid", "variant", "budget_max_total_s", metric_col]]


def _weighted_metric_by_groups(
    df: pd.DataFrame,
    *,
    metric_col: str,
    group_cols: list[str],
    macro_avg: bool,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "variant", metric_col])
    if metric_col not in df.columns:
        return pd.DataFrame(columns=[*group_cols, "variant", metric_col])

    work = df.copy()
    work = _coerce_numeric(work, [metric_col, "num_queries"])
    if "video_uid" not in work.columns:
        if "video_id" in work.columns:
            work["video_uid"] = work["video_id"]
        else:
            work["video_uid"] = "video_0000"
    if "num_queries" not in work.columns:
        work["num_queries"] = 1.0
    work["num_queries"] = work["num_queries"].fillna(1.0).clip(lower=1.0)
    work[metric_col] = work[metric_col].fillna(0.0)
    work["weighted"] = work[metric_col] * work["num_queries"]

    per_video = (
        work.groupby(["video_uid", "variant", *group_cols], dropna=False)[["weighted", "num_queries"]]
        .sum()
        .reset_index()
    )
    per_video[metric_col] = per_video["weighted"] / per_video["num_queries"].replace(0.0, 1.0)
    per_video = per_video[["video_uid", "variant", *group_cols, metric_col]]

    if macro_avg:
        return aggregate_over_videos(
            per_video,
            group_cols=["variant", *group_cols],
            metric_cols=[metric_col],
            agg="mean",
        )
    return _simple_group_mean(per_video, ["variant", *group_cols], [metric_col])


def _order_variants(frame: pd.DataFrame, variants: list[str]) -> list[str]:
    present = [v for v in variants if v in set(frame.get("variant", []))]
    if present:
        return present
    discovered = sorted(set(str(v) for v in frame.get("variant", []) if str(v)))
    return discovered


def _order_query_types(frame: pd.DataFrame) -> list[str]:
    present = [q for q in QUERY_ORDER if q in set(frame.get("query_type", []))]
    if present:
        return present
    discovered = sorted(set(str(v) for v in frame.get("query_type", []) if str(v)))
    return discovered


def _save_figure(fig: Any, stem: Path, formats: list[str]) -> list[Path]:
    saved: list[Path] = []
    for fmt in formats:
        target = stem.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=200, bbox_inches="tight")
        saved.append(target)
    plt.close(fig)
    return saved


def _md_heading_table(title: str, table_md: str) -> str:
    return f"# {title}\n\n{table_md}\n"


def _closest_available_budget(df: pd.DataFrame, target_budget: dict[str, float]) -> dict[str, float] | None:
    if df is None or df.empty:
        return None
    cols = ["budget_max_total_s", "budget_max_tokens", "budget_max_decisions"]
    if any(col not in df.columns for col in cols):
        return None
    unique = df[cols].dropna().drop_duplicates()
    if unique.empty:
        return None
    target_total = float(target_budget.get("max_total_s", unique["budget_max_total_s"].iloc[0]))
    target_tokens = float(target_budget.get("max_tokens", unique["budget_max_tokens"].iloc[0]))
    target_decisions = float(target_budget.get("max_decisions", unique["budget_max_decisions"].iloc[0]))
    work = unique.copy()
    work["__dist"] = (
        (work["budget_max_total_s"] - target_total).abs()
        + (work["budget_max_tokens"] - target_tokens).abs()
        + (work["budget_max_decisions"] - target_decisions).abs()
    )
    best = work.sort_values(["__dist", "budget_max_total_s", "budget_max_tokens", "budget_max_decisions"]).iloc[0]
    return {
        "max_total_s": float(best["budget_max_total_s"]),
        "max_tokens": float(best["budget_max_tokens"]),
        "max_decisions": float(best["budget_max_decisions"]),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    commit = (result.stdout or "").strip()
    return commit or None


def main() -> int:
    args = parse_args()
    cross_dir = Path(args.cross_dir)
    nlq_csv = Path(args.nlq_csv)
    run_label = str(args.label).strip() or "run"
    compare_label = str(args.compare_label).strip() or "compare"
    compare_snapshot_path: Path | None = None
    compare_cross_dir: Path | None = None
    compare_nlq_csv: Path | None = None
    compare_budget_hint: dict[str, float] | None = None
    compare_macro_hint: bool | None = None
    if args.compare_dir:
        try:
            compare_info = _load_compare_snapshot(Path(args.compare_dir))
        except Exception as exc:
            print(f"error=invalid_compare_dir detail={exc}")
            return 1
        compare_snapshot_path = Path(compare_info["snapshot_path"])
        compare_cross_dir = _resolve_path_hint(compare_info.get("cross_dir"), base_dir=Path(args.compare_dir))
        compare_nlq_csv = _resolve_path_hint(compare_info.get("nlq_csv"), base_dir=Path(args.compare_dir))
        raw_budget_hint = compare_info.get("budget_used")
        if isinstance(raw_budget_hint, dict):
            compare_budget_hint = {
                str(k): float(v)
                for k, v in raw_budget_hint.items()
                if str(k) in {"max_total_s", "max_tokens", "max_decisions"}
            }
        if isinstance(compare_info.get("macro_avg"), bool):
            compare_macro_hint = bool(compare_info.get("macro_avg"))
    else:
        compare_cross_dir = _resolve_path_hint(args.compare_cross_dir, base_dir=ROOT)
        compare_nlq_csv = _resolve_path_hint(args.compare_nlq_csv, base_dir=ROOT)

    ablation_terms_csv = Path(args.ablation_terms_csv) if args.ablation_terms_csv else None
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    overall_paths, by_type_paths = _discover_cross_csvs(cross_dir)
    if not overall_paths or not by_type_paths:
        print("error=missing_cross_csvs")
        print(f"cross_dir={cross_dir}")
        return 1
    if not nlq_csv.exists():
        print(f"error=missing_nlq_csv path={nlq_csv}")
        return 1
    compare_enabled = compare_cross_dir is not None or compare_nlq_csv is not None
    if compare_enabled:
        if compare_cross_dir is None or compare_nlq_csv is None:
            print("error=compare_inputs_incomplete require compare cross_dir and nlq_csv")
            return 1
        if not compare_cross_dir.exists():
            print(f"error=missing_compare_cross_dir path={compare_cross_dir}")
            return 1
        if not compare_nlq_csv.exists():
            print(f"error=missing_compare_nlq_csv path={compare_nlq_csv}")
            return 1

    formats = _parse_formats(args.formats)
    budget = _parse_budget(args.budget)
    if args.budget_max_total_s is not None:
        budget["max_total_s"] = float(args.budget_max_total_s)
    if args.budget_max_tokens is not None:
        budget["max_tokens"] = float(args.budget_max_tokens)
    if args.budget_max_decisions is not None:
        budget["max_decisions"] = float(args.budget_max_decisions)
    if not budget:
        print(f"error=invalid_budget value={args.budget}")
        return 1
    if compare_budget_hint is not None:
        for key in ("max_total_s", "max_tokens", "max_decisions"):
            if key in compare_budget_hint and key not in budget:
                budget[key] = float(compare_budget_hint[key])
    if compare_macro_hint is not None and not bool(args.macro_avg):
        print(f"info=compare_snapshot_macro_avg={compare_macro_hint} current_macro_avg={bool(args.macro_avg)}")

    overall_df = load_csvs(overall_paths)
    by_type_df = load_csvs(by_type_paths)
    nlq_df = load_csvs([nlq_csv])
    overall_df = _coerce_numeric(
        overall_df,
        [
            "budget_max_total_s",
            "budget_max_tokens",
            "budget_max_decisions",
            "hit_at_k",
            "mrr",
            "coverage_ratio",
            "compression_ratio",
        ],
    )
    by_type_df = _coerce_numeric(
        by_type_df,
        [
            "budget_max_total_s",
            "budget_max_tokens",
            "budget_max_decisions",
            "hit_at_k",
            "mrr",
        ],
    )
    nlq_df = _coerce_numeric(
        nlq_df,
        [
            "budget_max_total_s",
            "budget_max_tokens",
            "budget_max_decisions",
            "hit_at_k",
            "hit_at_k_strict",
            "hit_at_1_strict",
            "mrr",
            "num_queries",
            "top1_in_distractor_rate",
            "fp_rate",
            "top1_kind_highlight_rate",
            "top1_kind_token_rate",
            "top1_kind_decision_rate",
            "top1_kind_event_rate",
        ],
    )
    compare_overall_df = pd.DataFrame()
    compare_by_type_df = pd.DataFrame()
    compare_nlq_df = pd.DataFrame()
    if compare_enabled and compare_cross_dir is not None and compare_nlq_csv is not None:
        compare_overall_paths, compare_by_type_paths = _discover_cross_csvs(compare_cross_dir)
        if not compare_overall_paths or not compare_by_type_paths:
            print(f"error=missing_compare_cross_csvs compare_cross_dir={compare_cross_dir}")
            return 1
        compare_overall_df = load_csvs(compare_overall_paths)
        compare_by_type_df = load_csvs(compare_by_type_paths)
        compare_nlq_df = load_csvs([compare_nlq_csv])
        compare_overall_df = _coerce_numeric(
            compare_overall_df,
            [
                "budget_max_total_s",
                "budget_max_tokens",
                "budget_max_decisions",
                "hit_at_k",
                "mrr",
                "coverage_ratio",
                "compression_ratio",
            ],
        )
        compare_by_type_df = _coerce_numeric(
            compare_by_type_df,
            [
                "budget_max_total_s",
                "budget_max_tokens",
                "budget_max_decisions",
                "hit_at_k",
                "mrr",
            ],
        )
        compare_nlq_df = _coerce_numeric(
            compare_nlq_df,
            [
                "budget_max_total_s",
                "budget_max_tokens",
                "budget_max_decisions",
                "hit_at_k",
                "hit_at_k_strict",
                "hit_at_1_strict",
                "mrr",
                "num_queries",
                "top1_in_distractor_rate",
                "fp_rate",
                "top1_kind_highlight_rate",
                "top1_kind_token_rate",
                "top1_kind_decision_rate",
                "top1_kind_event_rate",
            ],
        )
        if "duration_bucket" not in compare_nlq_df.columns and "duration_s" in compare_nlq_df.columns:
            compare_nlq_df["duration_bucket"] = compare_nlq_df["duration_s"].apply(get_duration_bucket)

    if "duration_bucket" not in nlq_df.columns and "duration_s" in nlq_df.columns:
        nlq_df["duration_bucket"] = nlq_df["duration_s"].apply(get_duration_bucket)

    # Fixed budget slice for ablation figures/tables.
    overall_budget_df = pick_budget_slice(overall_df, budget=budget)
    by_type_budget_df = pick_budget_slice(by_type_df, budget=budget)
    selected_budget = dict(budget)
    if overall_budget_df.empty or by_type_budget_df.empty:
        fallback = _closest_available_budget(overall_df, budget)
        if fallback is not None:
            selected_budget = fallback
            overall_budget_df = pick_budget_slice(overall_df, budget=selected_budget)
            by_type_budget_df = pick_budget_slice(by_type_df, budget=selected_budget)
            print(f"warn=budget_not_found requested={budget} fallback={selected_budget}")
    compare_overall_budget_df = pd.DataFrame()
    compare_by_type_budget_df = pd.DataFrame()
    compare_nlq_budget_df = pd.DataFrame()
    if compare_enabled:
        compare_overall_budget_df = pick_budget_slice(compare_overall_df, budget=selected_budget)
        compare_by_type_budget_df = pick_budget_slice(compare_by_type_df, budget=selected_budget)
        compare_nlq_budget_df = pick_budget_slice(compare_nlq_df, budget=selected_budget)
        if compare_overall_budget_df.empty or compare_by_type_budget_df.empty or compare_nlq_budget_df.empty:
            print(f"error=compare_budget_not_aligned budget_used={selected_budget}")
            return 1
    overall_agg = _agg(
        overall_budget_df,
        group_cols=["variant"],
        metric_cols=["hit_at_k", "mrr", "coverage_ratio", "compression_ratio"],
        macro_avg=bool(args.macro_avg),
    )
    by_type_agg = _agg(
        by_type_budget_df,
        group_cols=["query_type", "variant"],
        metric_cols=["hit_at_k", "mrr"],
        macro_avg=bool(args.macro_avg),
    )

    if overall_agg.empty or by_type_agg.empty:
        print("error=empty_budget_slice")
        print(f"budget={selected_budget}")
        return 1

    saved_figures: list[Path] = []

    # Figure 1/2: NLQ budget curves over max_total_s (fix max_tokens/max_decisions at selected budget).
    curve_budget = {
        "max_tokens": selected_budget.get("max_tokens"),
        "max_decisions": selected_budget.get("max_decisions"),
    }
    nlq_curve_source = pick_budget_slice(nlq_df, budget=curve_budget)
    hit_curve = _weighted_query_mean(nlq_curve_source, metric_col="hit_at_k")
    mrr_curve = _weighted_query_mean(nlq_curve_source, metric_col="mrr")
    hit_curve_agg = _agg(hit_curve, ["variant", "budget_max_total_s"], ["hit_at_k"], macro_avg=bool(args.macro_avg))
    mrr_curve_agg = _agg(mrr_curve, ["variant", "budget_max_total_s"], ["mrr"], macro_avg=bool(args.macro_avg))

    ordered_curve_variants = _order_variants(hit_curve_agg, VARIANT_ORDER_FOCUS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant in ordered_curve_variants:
        sub = hit_curve_agg[hit_curve_agg["variant"] == variant].sort_values("budget_max_total_s")
        if sub.empty:
            continue
        ax.plot(sub["budget_max_total_s"], sub["hit_at_k"], marker="o", label=variant)
    ax.set_title("NLQ Hit@K vs Time Budget (Macro Avg over Videos)" if args.macro_avg else "NLQ Hit@K vs Time Budget")
    ax.set_xlabel("max_total_s")
    ax.set_ylabel("hit@k")
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved_figures.extend(_save_figure(fig, fig_dir / "fig_budget_hitk_vs_seconds", formats))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant in ordered_curve_variants:
        sub = mrr_curve_agg[mrr_curve_agg["variant"] == variant].sort_values("budget_max_total_s")
        if sub.empty:
            continue
        ax.plot(sub["budget_max_total_s"], sub["mrr"], marker="o", label=variant)
    ax.set_title("NLQ MRR vs Time Budget (Macro Avg over Videos)" if args.macro_avg else "NLQ MRR vs Time Budget")
    ax.set_xlabel("max_total_s")
    ax.set_ylabel("mrr")
    ax.grid(True, alpha=0.3)
    ax.legend()
    saved_figures.extend(_save_figure(fig, fig_dir / "fig_budget_mrr_vs_seconds", formats))

    # Figure 3: Ablation overall at fixed budget.
    ordered_variants = _order_variants(overall_agg, VARIANT_ORDER)
    plot_overall = overall_agg.copy()
    plot_overall["__order"] = plot_overall["variant"].apply(
        lambda name: VARIANT_ORDER.index(name) if name in VARIANT_ORDER else 999
    )
    plot_overall = plot_overall.sort_values(["__order", "variant"])
    x = range(len(plot_overall))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(list(x), plot_overall["hit_at_k"])
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(plot_overall["variant"], rotation=20, ha="right")
    axes[0].set_ylabel("hit@k")
    axes[0].set_title("Overall Hit@K")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(list(x), plot_overall["mrr"])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(plot_overall["variant"], rotation=20, ha="right")
    axes[1].set_ylabel("mrr")
    axes[1].set_title("Overall MRR")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Ablation Overall at Fixed Budget")
    saved_figures.extend(_save_figure(fig, fig_dir / "fig_ablation_overall", formats))

    # Figure 4: Ablation by query type (focus variants).
    qplot = by_type_agg[by_type_agg["variant"].isin(VARIANT_ORDER_FOCUS)].copy()
    qtypes = _order_query_types(qplot)
    x_idx = list(range(len(qtypes)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, variant in enumerate(VARIANT_ORDER_FOCUS):
        sub = qplot[qplot["variant"] == variant]
        y = []
        for qtype in qtypes:
            row = sub[sub["query_type"] == qtype]
            y.append(float(row["hit_at_k"].iloc[0]) if len(row) else 0.0)
        offsets = [idx + (i - (len(VARIANT_ORDER_FOCUS) - 1) / 2.0) * width for idx in x_idx]
        ax.bar(offsets, y, width=width, label=variant)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(qtypes, rotation=20, ha="right")
    ax.set_ylabel("hit@k")
    ax.set_title("Ablation on Query Types (Token/Decision Gains)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    saved_figures.extend(_save_figure(fig, fig_dir / "fig_ablation_by_query_type", formats))

    # Figure 5: Coverage/Compression helper at fixed budget.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(list(x), plot_overall["coverage_ratio"])
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(plot_overall["variant"], rotation=20, ha="right")
    axes[0].set_ylabel("coverage_ratio")
    axes[0].set_title("Coverage at Fixed Budget")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(list(x), plot_overall["compression_ratio"])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(plot_overall["variant"], rotation=20, ha="right")
    axes[1].set_ylabel("compression_ratio")
    axes[1].set_title("Compression at Fixed Budget")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Coverage/Compression Comparison")
    saved_figures.extend(_save_figure(fig, fig_dir / "fig_coverage_compression", formats))

    # Figure 6: Performance vs duration bucket (macro average).
    if "duration_bucket" in nlq_df.columns:
        bucket_source = pick_budget_slice(
            nlq_df,
            budget={
                "max_total_s": selected_budget.get("max_total_s"),
                "max_tokens": selected_budget.get("max_tokens"),
                "max_decisions": selected_budget.get("max_decisions"),
            },
        ).copy()
        if not bucket_source.empty:
            if "num_queries" not in bucket_source.columns:
                bucket_source["num_queries"] = 1.0
            bucket_source["num_queries"] = pd.to_numeric(bucket_source["num_queries"], errors="coerce").fillna(1.0).clip(lower=1.0)
            bucket_source["hit_at_k"] = pd.to_numeric(bucket_source["hit_at_k"], errors="coerce").fillna(0.0)
            # If hard-pseudo query types exist, focus on token/decision slices.
            qtypes_present = set(str(x) for x in bucket_source.get("query_type", []) if str(x))
            hard_focus = {"hard_pseudo_token", "hard_pseudo_decision"}
            if hard_focus.intersection(qtypes_present):
                bucket_source = bucket_source[bucket_source["query_type"].isin(sorted(hard_focus))].copy()

            bucket_source["weighted"] = bucket_source["hit_at_k"] * bucket_source["num_queries"]
            if "video_uid" not in bucket_source.columns:
                if "video_id" in bucket_source.columns:
                    bucket_source["video_uid"] = bucket_source["video_id"]
                else:
                    bucket_source["video_uid"] = "video_0000"
            per_video_bucket = (
                bucket_source.groupby(["video_uid", "variant", "duration_bucket"], dropna=False)[["weighted", "num_queries"]]
                .sum()
                .reset_index()
            )
            per_video_bucket["hit_at_k"] = per_video_bucket["weighted"] / per_video_bucket["num_queries"].replace(0.0, 1.0)
            bucket_agg = _agg(
                per_video_bucket,
                group_cols=["variant", "duration_bucket"],
                metric_cols=["hit_at_k"],
                macro_avg=bool(args.macro_avg),
            )
            if not bucket_agg.empty:
                bucket_order = ["short", "medium", "long", "very_long", "unknown"]
                bucket_names = [b for b in bucket_order if b in set(bucket_agg["duration_bucket"])]
                other_buckets = sorted(set(bucket_agg["duration_bucket"]) - set(bucket_names))
                bucket_names.extend(other_buckets)

                x_bucket = list(range(len(bucket_names)))
                width = 0.18
                fig, ax = plt.subplots(figsize=(10.5, 4.8))
                for i, variant in enumerate(VARIANT_ORDER_FOCUS):
                    sub = bucket_agg[bucket_agg["variant"] == variant]
                    y = []
                    for bname in bucket_names:
                        row = sub[sub["duration_bucket"] == bname]
                        y.append(float(row["hit_at_k"].iloc[0]) if len(row) else 0.0)
                    offsets = [idx + (i - (len(VARIANT_ORDER_FOCUS) - 1) / 2.0) * width for idx in x_bucket]
                    ax.bar(offsets, y, width=width, label=variant)
                ax.set_xticks(x_bucket)
                ax.set_xticklabels(bucket_names)
                ax.set_ylabel("hit@k")
                ax.set_xlabel("duration_bucket")
                ax.set_title("NLQ Hit@K vs Video Duration Bucket (Macro Avg over Videos)")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend()
                saved_figures.extend(_save_figure(fig, fig_dir / "fig_by_duration_bucket_hitk", formats))

    # Figure 7/8: strict vs standard and distractor FP rate at fixed budget.
    strict_source = pick_budget_slice(
        nlq_df,
        budget={
            "max_total_s": selected_budget.get("max_total_s"),
            "max_tokens": selected_budget.get("max_tokens"),
            "max_decisions": selected_budget.get("max_decisions"),
        },
    ).copy()
    if not strict_source.empty and "hit_at_k_strict" in strict_source.columns:
        standard_df = _weighted_query_mean(strict_source, metric_col="hit_at_k")
        strict_df = _weighted_query_mean(strict_source, metric_col="hit_at_k_strict")
        std_agg = _agg(standard_df, ["variant"], ["hit_at_k"], macro_avg=bool(args.macro_avg))
        strict_agg = _agg(strict_df, ["variant"], ["hit_at_k_strict"], macro_avg=bool(args.macro_avg))

        merged = std_agg.merge(strict_agg, on=["variant"], how="outer").fillna(0.0)
        ordered_variants_strict = _order_variants(merged, VARIANT_ORDER_FOCUS)
        if not ordered_variants_strict:
            ordered_variants_strict = _order_variants(merged, VARIANT_ORDER)
        merged["__order"] = merged["variant"].apply(
            lambda name: ordered_variants_strict.index(name) if name in ordered_variants_strict else 999
        )
        merged = merged.sort_values(["__order", "variant"])

        x = list(range(len(merged)))
        width = 0.36
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar([i - width / 2 for i in x], merged["hit_at_k"], width=width, label="hit@k")
        ax.bar([i + width / 2 for i in x], merged["hit_at_k_strict"], width=width, label="hit@k_strict")
        ax.set_xticks(x)
        ax.set_xticklabels(merged["variant"], rotation=20, ha="right")
        ax.set_ylabel("score")
        ax.set_title("Strict vs Standard Retrieval (Macro Avg over Videos)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        saved_figures.extend(_save_figure(fig, fig_dir / "fig_strict_vs_standard", formats))

        if "top1_in_distractor_rate" in strict_source.columns:
            fp_df = _weighted_query_mean(strict_source, metric_col="top1_in_distractor_rate")
            fp_agg = _agg(fp_df, ["variant"], ["top1_in_distractor_rate"], macro_avg=bool(args.macro_avg))
            fp_agg["__order"] = fp_agg["variant"].apply(
                lambda name: ordered_variants_strict.index(name) if name in ordered_variants_strict else 999
            )
            fp_agg = fp_agg.sort_values(["__order", "variant"])
            fig, ax = plt.subplots(figsize=(9.5, 4.5))
            ax.bar(list(range(len(fp_agg))), fp_agg["top1_in_distractor_rate"])
            ax.set_xticks(list(range(len(fp_agg))))
            ax.set_xticklabels(fp_agg["variant"], rotation=20, ha="right")
            ax.set_ylabel("top1_in_distractor_rate")
            ax.set_title("Distractor Top-1 Error Rate (Lower is Better)")
            ax.grid(True, axis="y", alpha=0.3)
            saved_figures.extend(_save_figure(fig, fig_dir / "fig_fp_rate", formats))

        # Figure 9: top1 kind distribution (stacked bars).
        kind_cols = [
            "top1_kind_highlight_rate",
            "top1_kind_token_rate",
            "top1_kind_decision_rate",
            "top1_kind_event_rate",
        ]
        if all(col in strict_source.columns for col in kind_cols):
            kind_base = strict_source.copy()
            if "video_uid" not in kind_base.columns:
                if "video_id" in kind_base.columns:
                    kind_base["video_uid"] = kind_base["video_id"]
                else:
                    kind_base["video_uid"] = "video_0000"
            kind_agg = _agg(
                kind_base,
                group_cols=["variant"],
                metric_cols=kind_cols,
                macro_avg=bool(args.macro_avg),
            )
            if not kind_agg.empty:
                kind_order = _order_variants(kind_agg, VARIANT_ORDER_FOCUS)
                if not kind_order:
                    kind_order = _order_variants(kind_agg, VARIANT_ORDER)
                kind_agg["__order"] = kind_agg["variant"].apply(
                    lambda name: kind_order.index(name) if name in kind_order else 999
                )
                kind_agg = kind_agg.sort_values(["__order", "variant"])

                x = list(range(len(kind_agg)))
                fig, ax = plt.subplots(figsize=(10, 4.8))
                bottoms = [0.0] * len(kind_agg)
                labels = [
                    ("top1_kind_highlight_rate", "highlight"),
                    ("top1_kind_token_rate", "token"),
                    ("top1_kind_decision_rate", "decision"),
                    ("top1_kind_event_rate", "event"),
                ]
                for col, label in labels:
                    values = [float(v) for v in kind_agg[col].tolist()]
                    ax.bar(x, values, bottom=bottoms, label=label)
                    bottoms = [b + v for b, v in zip(bottoms, values)]
                ax.set_xticks(x)
                ax.set_xticklabels(kind_agg["variant"], rotation=20, ha="right")
                ax.set_ylabel("ratio")
                ax.set_title("Top-1 Kind Distribution by Variant")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend()
                saved_figures.extend(_save_figure(fig, fig_dir / "fig_top1_kind_distribution", formats))

        # Optional: best vs default reranker config comparison if multiple configs exist.
        if "rerank_cfg_name" in strict_source.columns:
            cfg_values = sorted({str(x) for x in strict_source["rerank_cfg_name"] if str(x)})
            if len(cfg_values) >= 2:
                strict_cfg_df = _weighted_query_mean(strict_source, metric_col="hit_at_k_strict")
                strict_cfg_agg = _agg(strict_cfg_df, ["variant", "rerank_cfg_name"], ["hit_at_k_strict"], macro_avg=bool(args.macro_avg))
                if not strict_cfg_agg.empty:
                    # Pick default and best (max strict) per variant.
                    rows_plot: list[dict[str, Any]] = []
                    for variant in set(strict_cfg_agg["variant"]):
                        sub = strict_cfg_agg[strict_cfg_agg["variant"] == variant].copy()
                        if sub.empty:
                            continue
                        default_row = sub[sub["rerank_cfg_name"] == "default"]
                        if default_row.empty:
                            default_row = sub.sort_values("hit_at_k_strict", ascending=False).head(1)
                        best_row = sub.sort_values("hit_at_k_strict", ascending=False).head(1)
                        rows_plot.append(
                            {
                                "variant": str(variant),
                                "default": float(default_row["hit_at_k_strict"].iloc[0]),
                                "best": float(best_row["hit_at_k_strict"].iloc[0]),
                                "best_cfg": str(best_row["rerank_cfg_name"].iloc[0]),
                            }
                        )
                    if rows_plot:
                        plot_df = pd.DataFrame(rows_plot)
                        order = _order_variants(plot_df, VARIANT_ORDER_FOCUS)
                        if not order:
                            order = _order_variants(plot_df, VARIANT_ORDER)
                        plot_df["__order"] = plot_df["variant"].apply(lambda n: order.index(n) if n in order else 999)
                        plot_df = plot_df.sort_values(["__order", "variant"])
                        x = list(range(len(plot_df)))
                        width = 0.36
                        fig, ax = plt.subplots(figsize=(10, 4.8))
                        ax.bar([i - width / 2 for i in x], plot_df["default"], width=width, label="default")
                        ax.bar([i + width / 2 for i in x], plot_df["best"], width=width, label="best")
                        ax.set_xticks(x)
                        ax.set_xticklabels(plot_df["variant"], rotation=20, ha="right")
                        ax.set_ylabel("hit@k_strict")
                        ax.set_title("Best vs Default Reranker (Strict)")
                        ax.grid(True, axis="y", alpha=0.3)
                        ax.legend()
                        saved_figures.extend(_save_figure(fig, fig_dir / "fig_best_vs_default_strict", formats))

    # Tables.
    table_main_df = build_main_table(overall_agg)
    table_ablation_df = build_ablation_table(by_type_agg)
    table_main_md = table_dir / "table_main.md"
    table_ablation_md = table_dir / "table_ablation.md"
    table_main_tex = table_dir / "table_main.tex"
    table_ablation_tex = table_dir / "table_ablation.tex"

    table_main_md.write_text(
        _md_heading_table("Main Table", df_to_markdown_table(table_main_df, float_format="%.4f")),
        encoding="utf-8",
    )
    table_ablation_md.write_text(
        _md_heading_table("Ablation Table", df_to_markdown_table(table_ablation_df, float_format="%.4f")),
        encoding="utf-8",
    )
    table_main_tex.write_text(
        df_to_latex_table(
            table_main_df,
            caption="Overall macro-average performance at fixed budget.",
            label="tab:main_overall",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )
    table_ablation_tex.write_text(
        df_to_latex_table(
            table_ablation_df,
            caption="Ablation by query type (hit@k).",
            label="tab:ablation_query_type",
            float_format="%.4f",
        ),
        encoding="utf-8",
    )

    table_compare_md: Path | None = None
    table_compare_tex: Path | None = None
    if compare_enabled and compare_cross_dir is not None and compare_nlq_csv is not None:
        compare_overall_agg = _agg(
            compare_overall_budget_df,
            group_cols=["variant"],
            metric_cols=["hit_at_k", "mrr", "coverage_ratio", "compression_ratio"],
            macro_avg=bool(args.macro_avg),
        )
        if compare_overall_agg.empty:
            print(f"error=empty_compare_budget_slice budget={selected_budget}")
            return 1

        # Build compare table with run/compare values and deltas.
        strict_metric = "hit_at_k_strict"
        strict1_metric = "hit_at_1_strict"
        fp_metric = "top1_in_distractor_rate" if "top1_in_distractor_rate" in nlq_df.columns else "fp_rate"
        compare_metrics = [strict_metric, strict1_metric, "mrr", fp_metric]

        run_metric_frames: dict[str, pd.DataFrame] = {}
        cmp_metric_frames: dict[str, pd.DataFrame] = {}
        for metric in compare_metrics:
            run_metric_frames[metric] = _weighted_metric_by_groups(
                pick_budget_slice(nlq_df, budget=selected_budget).copy(),
                metric_col=metric,
                group_cols=[],
                macro_avg=bool(args.macro_avg),
            )
            cmp_metric_frames[metric] = _weighted_metric_by_groups(
                compare_nlq_budget_df.copy(),
                metric_col=metric,
                group_cols=[],
                macro_avg=bool(args.macro_avg),
            )

        variants_all = sorted(
            set(str(v) for v in overall_agg.get("variant", []))
            | set(str(v) for v in compare_overall_agg.get("variant", []))
            | set(str(v) for v in pick_budget_slice(nlq_df, budget=selected_budget).get("variant", []))
            | set(str(v) for v in compare_nlq_budget_df.get("variant", []))
        )
        table_compare_df = pd.DataFrame({"variant": variants_all})

        def _attach_pair(metric_name: str, run_df: pd.DataFrame, cmp_df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
            run_col = f"{run_label}_{metric_name}"
            cmp_col = f"{compare_label}_{metric_name}"
            delta_col = f"delta_{metric_name}"
            run_slice = run_df[["variant", metric_name]].rename(columns={metric_name: run_col}) if metric_name in run_df.columns else pd.DataFrame(columns=["variant", run_col])
            cmp_slice = cmp_df[["variant", metric_name]].rename(columns={metric_name: cmp_col}) if metric_name in cmp_df.columns else pd.DataFrame(columns=["variant", cmp_col])
            out = base.merge(run_slice, on="variant", how="left").merge(cmp_slice, on="variant", how="left")
            if run_col not in out.columns:
                out[run_col] = pd.NA
            if cmp_col not in out.columns:
                out[cmp_col] = pd.NA
            out[delta_col] = pd.to_numeric(out[run_col], errors="coerce") - pd.to_numeric(out[cmp_col], errors="coerce")
            return out

        # strict/mrr/fp from NLQ summaries
        for metric in compare_metrics:
            table_compare_df = _attach_pair(metric, run_metric_frames[metric], cmp_metric_frames[metric], table_compare_df)
        # coverage/compression from cross overall
        table_compare_df = _attach_pair("coverage_ratio", overall_agg, compare_overall_agg, table_compare_df)
        table_compare_df = _attach_pair("compression_ratio", overall_agg, compare_overall_agg, table_compare_df)

        table_compare_df["__order"] = table_compare_df["variant"].apply(
            lambda name: VARIANT_ORDER.index(name) if name in VARIANT_ORDER else 999
        )
        table_compare_df = table_compare_df.sort_values(["__order", "variant"]).drop(columns=["__order"]).reset_index(drop=True)

        table_compare_md = table_dir / "table_compare.md"
        table_compare_tex = table_dir / "table_compare.tex"
        table_compare_md.write_text(
            _md_heading_table("Compare Table", df_to_markdown_table(table_compare_df, float_format="%.4f")),
            encoding="utf-8",
        )
        table_compare_tex.write_text(
            df_to_latex_table(
                table_compare_df,
                caption=f"Comparison between {run_label} and {compare_label} at aligned budget.",
                label="tab:compare_overall",
                float_format="%.4f",
            ),
            encoding="utf-8",
        )

        # Compare figure 1: strict deltas (run - compare) by variant.
        delta_metrics = [strict_metric, strict1_metric, "mrr", fp_metric]
        display_names = {
            strict_metric: "delta hit@k_strict",
            strict1_metric: "delta hit@1_strict",
            "mrr": "delta mrr",
            fp_metric: f"delta {fp_metric}",
        }
        plot_variants = _order_variants(table_compare_df, VARIANT_ORDER_FOCUS)
        if not plot_variants:
            plot_variants = _order_variants(table_compare_df, VARIANT_ORDER)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
        axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        for idx, metric in enumerate(delta_metrics):
            ax = axes_list[idx]
            delta_col = f"delta_{metric}"
            sub = table_compare_df[table_compare_df["variant"].isin(plot_variants)].copy()
            y = [float(sub.loc[sub["variant"] == v, delta_col].iloc[0]) if delta_col in sub.columns and len(sub.loc[sub["variant"] == v, delta_col]) else 0.0 for v in plot_variants]
            ax.bar(list(range(len(plot_variants))), y)
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            ax.set_xticks(list(range(len(plot_variants))))
            ax.set_xticklabels(plot_variants, rotation=20, ha="right")
            ax.set_title(display_names.get(metric, metric))
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"{run_label} - {compare_label} Delta (Strict Metrics)")
        saved_figures.extend(_save_figure(fig, fig_dir / "fig_compare_delta_strict", formats))

        # Compare figure 2: hit@k_strict delta by query_type.
        run_q = _weighted_metric_by_groups(
            pick_budget_slice(nlq_df, budget=selected_budget).copy(),
            metric_col="hit_at_k_strict",
            group_cols=["query_type"],
            macro_avg=bool(args.macro_avg),
        )
        cmp_q = _weighted_metric_by_groups(
            compare_nlq_budget_df.copy(),
            metric_col="hit_at_k_strict",
            group_cols=["query_type"],
            macro_avg=bool(args.macro_avg),
        )
        if not run_q.empty and not cmp_q.empty:
            q_merged = run_q.merge(cmp_q, on=["variant", "query_type"], how="outer", suffixes=(f"_{run_label}", f"_{compare_label}"))
            run_col = f"hit_at_k_strict_{run_label}"
            cmp_col = f"hit_at_k_strict_{compare_label}"
            q_merged["delta_hit_at_k_strict"] = pd.to_numeric(q_merged.get(run_col), errors="coerce").fillna(0.0) - pd.to_numeric(q_merged.get(cmp_col), errors="coerce").fillna(0.0)
            qtypes = _order_query_types(q_merged)
            variants_focus = [v for v in VARIANT_ORDER_FOCUS if v in set(q_merged["variant"])]
            if variants_focus and qtypes:
                x_idx = list(range(len(qtypes)))
                width = 0.18
                fig, ax = plt.subplots(figsize=(12, 5.2))
                for i, variant in enumerate(variants_focus):
                    sub = q_merged[q_merged["variant"] == variant]
                    y = []
                    for qtype in qtypes:
                        row = sub[sub["query_type"] == qtype]
                        y.append(float(row["delta_hit_at_k_strict"].iloc[0]) if len(row) else 0.0)
                    offsets = [idx + (i - (len(variants_focus) - 1) / 2.0) * width for idx in x_idx]
                    ax.bar(offsets, y, width=width, label=variant)
                ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
                ax.set_xticks(x_idx)
                ax.set_xticklabels(qtypes, rotation=20, ha="right")
                ax.set_ylabel("delta hit@k_strict")
                ax.set_title(f"{run_label} - {compare_label} by Query Type")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend()
                saved_figures.extend(_save_figure(fig, fig_dir / "fig_compare_by_query_type", formats))

        # Compare figure 3: hit@k_strict delta by duration bucket.
        run_b = _weighted_metric_by_groups(
            pick_budget_slice(nlq_df, budget=selected_budget).copy(),
            metric_col="hit_at_k_strict",
            group_cols=["duration_bucket"],
            macro_avg=bool(args.macro_avg),
        )
        cmp_b = _weighted_metric_by_groups(
            compare_nlq_budget_df.copy(),
            metric_col="hit_at_k_strict",
            group_cols=["duration_bucket"],
            macro_avg=bool(args.macro_avg),
        )
        if not run_b.empty and not cmp_b.empty and "duration_bucket" in run_b.columns and "duration_bucket" in cmp_b.columns:
            b_merged = run_b.merge(cmp_b, on=["variant", "duration_bucket"], how="outer", suffixes=(f"_{run_label}", f"_{compare_label}"))
            run_col = f"hit_at_k_strict_{run_label}"
            cmp_col = f"hit_at_k_strict_{compare_label}"
            b_merged["delta_hit_at_k_strict"] = pd.to_numeric(b_merged.get(run_col), errors="coerce").fillna(0.0) - pd.to_numeric(b_merged.get(cmp_col), errors="coerce").fillna(0.0)
            bucket_order = ["short", "medium", "long", "very_long", "unknown"]
            bucket_names = [b for b in bucket_order if b in set(b_merged["duration_bucket"])]
            bucket_names.extend(sorted(set(str(x) for x in b_merged["duration_bucket"] if str(x) and str(x) not in bucket_names)))
            variants_focus = [v for v in VARIANT_ORDER_FOCUS if v in set(b_merged["variant"])]
            if variants_focus and bucket_names:
                x_idx = list(range(len(bucket_names)))
                width = 0.18
                fig, ax = plt.subplots(figsize=(11, 4.8))
                for i, variant in enumerate(variants_focus):
                    sub = b_merged[b_merged["variant"] == variant]
                    y = []
                    for bname in bucket_names:
                        row = sub[sub["duration_bucket"] == bname]
                        y.append(float(row["delta_hit_at_k_strict"].iloc[0]) if len(row) else 0.0)
                    offsets = [idx + (i - (len(variants_focus) - 1) / 2.0) * width for idx in x_idx]
                    ax.bar(offsets, y, width=width, label=variant)
                ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
                ax.set_xticks(x_idx)
                ax.set_xticklabels(bucket_names)
                ax.set_ylabel("delta hit@k_strict")
                ax.set_xlabel("duration_bucket")
                ax.set_title(f"{run_label} - {compare_label} by Duration Bucket")
                ax.grid(True, axis="y", alpha=0.3)
                ax.legend()
                saved_figures.extend(_save_figure(fig, fig_dir / "fig_compare_by_duration_bucket", formats))

    # Optional terms ablation figure.
    if ablation_terms_csv is not None and ablation_terms_csv.exists():
        terms_df = load_csvs([ablation_terms_csv])
        terms_df = _coerce_numeric(terms_df, ["hit_at_k_strict", "delta_hit_at_k_strict", "fp_rate", "delta_fp_rate"])
        if not terms_df.empty and "term" in terms_df.columns:
            terms_df = terms_df.copy()
            terms_df["__order"] = terms_df["term"].apply(
                lambda t: ["baseline", "intent_bonus", "distractor_penalty", "match_mismatch", "constraint_hard_filter"].index(t)
                if t in ["baseline", "intent_bonus", "distractor_penalty", "match_mismatch", "constraint_hard_filter"]
                else 999
            )
            terms_df = terms_df.sort_values(["__order", "term"])
            x = list(range(len(terms_df)))
            width = 0.36
            fig, ax = plt.subplots(figsize=(10.5, 4.8))
            ax.bar([i - width / 2 for i in x], terms_df["delta_hit_at_k_strict"], width=width, label="delta_hit@k_strict")
            ax.bar([i + width / 2 for i in x], terms_df["delta_fp_rate"], width=width, label="delta_fp_rate")
            ax.set_xticks(x)
            ax.set_xticklabels(terms_df["term"], rotation=20, ha="right")
            ax.set_ylabel("delta")
            ax.set_title("Reranker Terms Ablation")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
            saved_figures.extend(_save_figure(fig, fig_dir / "fig_terms_ablation", formats))

    # Delta summary for snapshot/debug.
    delta_df = compute_deltas(
        by_type_agg,
        baseline_variant="highlights_only",
        target_variants=["highlights_plus_tokens", "highlights_plus_decisions", "full"],
    )

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "args": {
            "label": run_label,
            "compare_label": compare_label if compare_enabled else None,
            "cross_dir": str(cross_dir),
            "nlq_csv": str(nlq_csv),
            "out_dir": str(out_dir),
            "budget_requested": budget,
            "budget_used": selected_budget,
            "macro_avg": bool(args.macro_avg),
            "formats": formats,
            "compare_dir": str(args.compare_dir) if args.compare_dir else None,
            "compare_cross_dir": str(compare_cross_dir) if compare_cross_dir is not None else None,
            "compare_nlq_csv": str(compare_nlq_csv) if compare_nlq_csv is not None else None,
            "compare_snapshot": str(compare_snapshot_path) if compare_snapshot_path is not None else None,
        },
        "inputs": {
            "results_overall": [str(p) for p in overall_paths],
            "results_by_query_type": [str(p) for p in by_type_paths],
            "nlq_summary_all": str(nlq_csv),
            "compare_results_overall": [str(p) for p in _discover_cross_csvs(compare_cross_dir)[0]] if compare_enabled and compare_cross_dir is not None else [],
            "compare_results_by_query_type": [str(p) for p in _discover_cross_csvs(compare_cross_dir)[1]] if compare_enabled and compare_cross_dir is not None else [],
            "compare_nlq_summary_all": str(compare_nlq_csv) if compare_enabled and compare_nlq_csv is not None else None,
        },
        "rows": {
            "overall_rows": int(len(overall_df)),
            "by_query_type_rows": int(len(by_type_df)),
            "nlq_rows": int(len(nlq_df)),
            "overall_budget_rows": int(len(overall_budget_df)),
            "by_query_type_budget_rows": int(len(by_type_budget_df)),
            "delta_rows": int(len(delta_df)),
            "compare_overall_rows": int(len(compare_overall_df)) if compare_enabled else 0,
            "compare_by_query_type_rows": int(len(compare_by_type_df)) if compare_enabled else 0,
            "compare_nlq_rows": int(len(compare_nlq_df)) if compare_enabled else 0,
            "compare_overall_budget_rows": int(len(compare_overall_budget_df)) if compare_enabled else 0,
            "compare_by_query_type_budget_rows": int(len(compare_by_type_budget_df)) if compare_enabled else 0,
            "compare_nlq_budget_rows": int(len(compare_nlq_budget_df)) if compare_enabled else 0,
        },
        "outputs": {
            "figures": [str(p) for p in saved_figures],
            "tables": [
                str(table_main_md),
                str(table_ablation_md),
                str(table_main_tex),
                str(table_ablation_tex),
                str(table_compare_md) if table_compare_md is not None else "",
                str(table_compare_tex) if table_compare_tex is not None else "",
            ],
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"cross_overall_files={len(overall_paths)}")
    print(f"cross_by_type_files={len(by_type_paths)}")
    print(f"rows_overall={len(overall_df)}")
    print(f"rows_by_query_type={len(by_type_df)}")
    print(f"rows_nlq={len(nlq_df)}")
    print(f"budget_requested={budget}")
    print(f"budget_used={selected_budget}")
    print(f"macro_avg={bool(args.macro_avg)}")
    print(f"label={run_label}")
    print(f"saved_figures={len(saved_figures)}")
    print(f"saved_table_main={table_main_md}")
    print(f"saved_table_ablation={table_ablation_md}")
    if table_compare_md is not None:
        print(f"saved_table_compare={table_compare_md}")
    if compare_enabled:
        print(f"compare_enabled=true")
        print(f"compare_label={compare_label}")
        print(f"compare_cross_dir={compare_cross_dir}")
        print(f"compare_nlq_csv={compare_nlq_csv}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
