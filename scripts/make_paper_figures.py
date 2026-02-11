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
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--budget",
        default="max_total_s=40,max_tokens=100,max_decisions=8",
        help="Budget selector, e.g. max_total_s=40,max_tokens=100,max_decisions=8",
    )
    parser.add_argument("--macro_avg", action="store_true", help="Macro average over video_uid")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated figure formats")
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


def _parse_formats(text: str) -> list[str]:
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

    formats = _parse_formats(args.formats)
    budget = _parse_budget(args.budget)
    if not budget:
        print(f"error=invalid_budget value={args.budget}")
        return 1

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
            "mrr",
            "num_queries",
            "top1_in_distractor_rate",
        ],
    )
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
            "cross_dir": str(cross_dir),
            "nlq_csv": str(nlq_csv),
            "out_dir": str(out_dir),
            "budget_requested": budget,
            "budget_used": selected_budget,
            "macro_avg": bool(args.macro_avg),
            "formats": formats,
        },
        "inputs": {
            "results_overall": [str(p) for p in overall_paths],
            "results_by_query_type": [str(p) for p in by_type_paths],
            "nlq_summary_all": str(nlq_csv),
        },
        "rows": {
            "overall_rows": int(len(overall_df)),
            "by_query_type_rows": int(len(by_type_df)),
            "nlq_rows": int(len(nlq_df)),
            "overall_budget_rows": int(len(overall_budget_df)),
            "by_query_type_budget_rows": int(len(by_type_budget_df)),
            "delta_rows": int(len(delta_df)),
        },
        "outputs": {
            "figures": [str(p) for p in saved_figures],
            "tables": [
                str(table_main_md),
                str(table_ablation_md),
                str(table_main_tex),
                str(table_ablation_tex),
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
    print(f"saved_figures={len(saved_figures)}")
    print(f"saved_table_main={table_main_md}")
    print(f"saved_table_ablation={table_ablation_md}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
