from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection

from pov_compiler.bench.manifest import ExperimentManifest
from pov_compiler.bench.reporting.latex import df_to_markdown_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for significance reporting.")
    return pd


def _write_csv(path: Path, df: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _budget_order(manifest: ExperimentManifest) -> dict[str, int]:
    return {point.key: idx for idx, point in enumerate(manifest.budgets.points)}


def _to_numeric(series: Any) -> Any:
    lib = _require_pandas()
    return lib.to_numeric(series, errors="coerce")


def _preferred_group(group_df: Any) -> Any:
    if "source_kind" not in group_df.columns:
        return group_df
    pair_rows = group_df.loc[group_df["source_kind"].astype(str).str.lower().isin(["pair", "detail", "unit"])]
    return pair_rows if len(pair_rows) > 0 else group_df


def paired_bootstrap_ci(
    baseline: list[float] | np.ndarray,
    treatment: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    baseline_arr = np.asarray(baseline, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)
    mask = np.isfinite(baseline_arr) & np.isfinite(treatment_arr)
    baseline_arr = baseline_arr[mask]
    treatment_arr = treatment_arr[mask]
    if baseline_arr.size < 2:
        return {"status": "insufficient_pairs", "ci_low": None, "ci_high": None}
    diffs = treatment_arr - baseline_arr
    rng = np.random.default_rng(int(seed))
    sample_idx = rng.integers(0, diffs.size, size=(int(n_boot), diffs.size))
    sample_means = diffs[sample_idx].mean(axis=1)
    alpha = 1.0 - float(confidence)
    return {
        "status": "ok",
        "ci_low": float(np.quantile(sample_means, alpha / 2.0)),
        "ci_high": float(np.quantile(sample_means, 1.0 - alpha / 2.0)),
    }


def wilcoxon_result(baseline: list[float] | np.ndarray, treatment: list[float] | np.ndarray) -> dict[str, Any]:
    baseline_arr = np.asarray(baseline, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)
    mask = np.isfinite(baseline_arr) & np.isfinite(treatment_arr)
    baseline_arr = baseline_arr[mask]
    treatment_arr = treatment_arr[mask]
    if baseline_arr.size < 2:
        return {"status": "insufficient_pairs", "p_value": None}
    diffs = treatment_arr - baseline_arr
    if not np.any(np.abs(diffs) > 0):
        return {"status": "all_zero_diffs", "p_value": 1.0}
    try:
        result = wilcoxon(treatment_arr, baseline_arr, zero_method="wilcox", correction=False)
    except Exception:
        return {"status": "wilcoxon_failed", "p_value": None}
    return {"status": "ok", "p_value": float(result.pvalue)}


def mcnemar_result(baseline: list[int] | np.ndarray, treatment: list[int] | np.ndarray) -> dict[str, Any]:
    baseline_arr = np.asarray(baseline, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)
    mask = np.isfinite(baseline_arr) & np.isfinite(treatment_arr)
    baseline_arr = baseline_arr[mask]
    treatment_arr = treatment_arr[mask]
    if baseline_arr.size < 2:
        return {"status": "insufficient_pairs", "p_value": None, "discordant_pairs": 0}
    baseline_bin = np.asarray(np.rint(baseline_arr), dtype=int)
    treatment_bin = np.asarray(np.rint(treatment_arr), dtype=int)
    if not np.isin(baseline_bin, [0, 1]).all() or not np.isin(treatment_bin, [0, 1]).all():
        return {"status": "binary_metric_unavailable", "p_value": None, "discordant_pairs": 0}
    table = np.array(
        [
            [
                int(np.sum((baseline_bin == 0) & (treatment_bin == 0))),
                int(np.sum((baseline_bin == 0) & (treatment_bin == 1))),
            ],
            [
                int(np.sum((baseline_bin == 1) & (treatment_bin == 0))),
                int(np.sum((baseline_bin == 1) & (treatment_bin == 1))),
            ],
        ],
        dtype=int,
    )
    discordant = int(table[0, 1] + table[1, 0])
    if discordant == 0:
        return {"status": "all_agree", "p_value": 1.0, "discordant_pairs": 0}
    exact = bool(discordant < 25)
    result = mcnemar(table, exact=exact, correction=not exact)
    return {"status": "ok", "p_value": float(result.pvalue), "discordant_pairs": discordant}


def _extract_pairs(
    group_df: Any,
    *,
    metric_col: str,
    baseline_key: str,
    treatment_key: str,
    binary_metric_col: str | None,
) -> dict[str, Any]:
    lib = _require_pandas()
    work = _preferred_group(group_df).copy()
    if "label_key" not in work.columns:
        work["label_key"] = ""
    if metric_col not in work.columns:
        return {"status": "metric_missing", "pairs": lib.DataFrame(), "binary_pairs": lib.DataFrame()}
    unit_col = "sample_unit" if "sample_unit" in work.columns else "__sample_unit"
    if unit_col == "__sample_unit":
        work[unit_col] = "aggregate"
    work[metric_col] = _to_numeric(work[metric_col])
    work = work.loc[work["label_key"].astype(str).isin([str(baseline_key), str(treatment_key)])]
    metric_pairs = (
        work.groupby([unit_col, "label_key"], dropna=False)[metric_col].mean().reset_index()
        .pivot(index=unit_col, columns="label_key", values=metric_col)
        .reset_index()
    )
    keep_cols = [col for col in [baseline_key, treatment_key] if col in metric_pairs.columns]
    if len(keep_cols) < 2:
        return {"status": "insufficient_pairs", "pairs": lib.DataFrame(), "binary_pairs": lib.DataFrame()}
    metric_pairs = metric_pairs.loc[:, [unit_col, baseline_key, treatment_key]].dropna().reset_index(drop=True)
    binary_pairs = lib.DataFrame()
    if binary_metric_col and binary_metric_col in work.columns:
        work[binary_metric_col] = _to_numeric(work[binary_metric_col])
        binary_pairs = (
            work.groupby([unit_col, "label_key"], dropna=False)[binary_metric_col].mean().reset_index()
            .pivot(index=unit_col, columns="label_key", values=binary_metric_col)
            .reset_index()
        )
        if baseline_key in binary_pairs.columns and treatment_key in binary_pairs.columns:
            binary_pairs = binary_pairs.loc[:, [unit_col, baseline_key, treatment_key]].dropna().reset_index(drop=True)
        else:
            binary_pairs = lib.DataFrame()
    elif not metric_pairs.empty:
        valid_values = np.unique(metric_pairs.loc[:, [baseline_key, treatment_key]].to_numpy(dtype=float))
        if set(np.asarray(valid_values[~np.isnan(valid_values)], dtype=int).tolist()).issubset({0, 1}):
            binary_pairs = metric_pairs.copy()
    status = "ok" if len(metric_pairs) > 0 else "insufficient_pairs"
    return {"status": status, "pairs": metric_pairs, "binary_pairs": binary_pairs}


def build_significance_tables(results_long: Any, manifest: ExperimentManifest) -> tuple[Any, Any]:
    lib = _require_pandas()
    if results_long is None or len(results_long) == 0:
        empty = lib.DataFrame(
            [
                {
                    "task": "n/a",
                    "budget_key": "n/a",
                    "budget_seconds": 0.0,
                    "primary_metric": "n/a",
                    "baseline_label": manifest.selection.labels.get(manifest.variants.baseline, manifest.variants.baseline),
                    "treatment_label": manifest.selection.labels.get(manifest.variants.treatment, manifest.variants.treatment),
                    "n_pairs": 0,
                    "status": "insufficient_pairs",
                }
            ]
        )
        return empty.copy(), empty.copy()

    rows: list[dict[str, Any]] = []
    baseline_key = manifest.variants.baseline
    treatment_key = manifest.variants.treatment
    label_a = manifest.selection.labels.get(baseline_key, baseline_key)
    label_b = manifest.selection.labels.get(treatment_key, treatment_key)
    order_map = _budget_order(manifest)

    for task in manifest.selection.tasks:
        metric_col = manifest.metrics.primary.get(task, "")
        binary_col = manifest.metrics.binary.get(task, "")
        task_df = results_long.loc[results_long["task"].astype(str) == str(task)].copy()
        if task_df.empty:
            rows.append(
                {
                    "task": task,
                    "budget_key": "n/a",
                    "budget_seconds": np.nan,
                    "primary_metric": metric_col,
                    "baseline_label": label_a,
                    "treatment_label": label_b,
                    "n_pairs": 0,
                    "baseline_mean": np.nan,
                    "treatment_mean": np.nan,
                    "mean_delta": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "wilcoxon_p": np.nan,
                    "mcnemar_p": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "primary_test": "none",
                    "status": "task_missing",
                }
            )
            continue
        budgets = sorted(task_df["budget_key"].astype(str).dropna().unique().tolist(), key=lambda key: order_map.get(key, 999))
        if not budgets:
            budgets = ["n/a"]
        for budget_key in budgets:
            group_df = task_df.loc[task_df["budget_key"].astype(str) == str(budget_key)].copy()
            extracted = _extract_pairs(
                group_df,
                metric_col=metric_col,
                baseline_key=baseline_key,
                treatment_key=treatment_key,
                binary_metric_col=binary_col or None,
            )
            pairs = extracted["pairs"]
            binary_pairs = extracted["binary_pairs"]
            if len(group_df) > 0 and "budget_seconds" in group_df.columns:
                budget_seconds = float(_to_numeric(group_df["budget_seconds"]).dropna().iloc[0]) if len(_to_numeric(group_df["budget_seconds"]).dropna()) > 0 else np.nan
            else:
                budget_seconds = np.nan
            if extracted["status"] != "ok" or pairs.empty:
                rows.append(
                    {
                        "task": task,
                        "budget_key": budget_key,
                        "budget_seconds": budget_seconds,
                        "primary_metric": metric_col,
                        "baseline_label": label_a,
                        "treatment_label": label_b,
                        "n_pairs": 0,
                        "baseline_mean": np.nan,
                        "treatment_mean": np.nan,
                        "mean_delta": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "wilcoxon_p": np.nan,
                        "mcnemar_p": np.nan,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "primary_test": "none",
                        "status": extracted["status"],
                    }
                )
                continue
            baseline_vals = pairs[baseline_key].to_numpy(dtype=float)
            treatment_vals = pairs[treatment_key].to_numpy(dtype=float)
            ci = paired_bootstrap_ci(baseline_vals, treatment_vals, seed=int(manifest.seed))
            wil = wilcoxon_result(baseline_vals, treatment_vals)
            mcn = (
                mcnemar_result(binary_pairs[baseline_key].to_numpy(dtype=float), binary_pairs[treatment_key].to_numpy(dtype=float))
                if not binary_pairs.empty
                else {"status": "binary_metric_unavailable", "p_value": None, "discordant_pairs": 0}
            )
            p_value = wil.get("p_value")
            primary_test = "wilcoxon"
            if p_value is None and mcn.get("p_value") is not None:
                p_value = mcn.get("p_value")
                primary_test = "mcnemar"
            if p_value is None:
                primary_test = "none"
            status = "ok"
            if len(pairs) < 2:
                status = "insufficient_pairs"
            elif primary_test == "none":
                status = wil.get("status") or mcn.get("status") or "insufficient_pairs"
            rows.append(
                {
                    "task": task,
                    "budget_key": budget_key,
                    "budget_seconds": budget_seconds,
                    "primary_metric": metric_col,
                    "baseline_label": label_a,
                    "treatment_label": label_b,
                    "n_pairs": int(len(pairs)),
                    "baseline_mean": float(np.mean(baseline_vals)),
                    "treatment_mean": float(np.mean(treatment_vals)),
                    "mean_delta": float(np.mean(treatment_vals - baseline_vals)),
                    "ci_low": ci.get("ci_low"),
                    "ci_high": ci.get("ci_high"),
                    "wilcoxon_p": wil.get("p_value"),
                    "mcnemar_p": mcn.get("p_value"),
                    "p_value": p_value,
                    "q_value": np.nan,
                    "primary_test": primary_test,
                    "wilcoxon_status": wil.get("status"),
                    "mcnemar_status": mcn.get("status"),
                    "status": status,
                }
            )

    out = lib.DataFrame(rows)
    valid_p = out["p_value"].notna() if "p_value" in out.columns else lib.Series(dtype=bool)
    if len(out) > 0 and valid_p.any():
        _, q_vals = fdrcorrection(out.loc[valid_p, "p_value"].astype(float).to_numpy())
        out.loc[valid_p, "q_value"] = q_vals
    ci_table = out.loc[
        :,
        [
            col
            for col in (
                "task",
                "budget_key",
                "budget_seconds",
                "primary_metric",
                "n_pairs",
                "mean_delta",
                "ci_low",
                "ci_high",
                "status",
            )
            if col in out.columns
        ],
    ].copy()
    return out, ci_table


def _save_placeholder_figure(path_base: Path, title: str, message: str, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_paths: list[str] = []
    plt.figure(figsize=(8.0, 4.2))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    for ext in formats:
        target = path_base.with_suffix(f".{ext}")
        plt.savefig(target)
        out_paths.append(str(target))
    plt.close()
    return out_paths


def _make_significance_figures(table_df: Any, out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    rows = table_df.loc[table_df["status"].astype(str) == "ok"].copy() if len(table_df) > 0 else table_df.copy()
    if rows.empty:
        paths = _save_placeholder_figure(
            out_dir / "fig_significance_delta_vs_budget_seconds",
            "Significance Delta vs Budget Seconds",
            "No valid paired rows available",
            formats,
        )
        paths.extend(
            _save_placeholder_figure(
                out_dir / "fig_effect_size_forest",
                "Effect Size Forest",
                "No valid paired rows available",
                formats,
            )
        )
        return paths

    figure_paths: list[str] = []
    rows["budget_seconds"] = _to_numeric(rows["budget_seconds"])
    rows["mean_delta"] = _to_numeric(rows["mean_delta"])
    rows["ci_low"] = _to_numeric(rows["ci_low"])
    rows["ci_high"] = _to_numeric(rows["ci_high"])

    fig1 = out_dir / "fig_significance_delta_vs_budget_seconds"
    plt.figure(figsize=(8.2, 4.6))
    for task in sorted(rows["task"].astype(str).unique().tolist()):
        task_rows = rows.loc[rows["task"].astype(str) == str(task)].sort_values("budget_seconds")
        xs = task_rows["budget_seconds"].to_numpy(dtype=float)
        ys = task_rows["mean_delta"].to_numpy(dtype=float)
        yerr_low = ys - task_rows["ci_low"].to_numpy(dtype=float)
        yerr_high = task_rows["ci_high"].to_numpy(dtype=float) - ys
        yerr = np.vstack([yerr_low, yerr_high])
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label=str(task))
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("Budget Seconds")
    plt.ylabel("Mean Delta")
    plt.title("Paired Delta vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        target = fig1.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    fig2 = out_dir / "fig_effect_size_forest"
    forest = rows.sort_values(["task", "budget_seconds"]).copy()
    labels = [f"{task}:{budget}" for task, budget in zip(forest["task"], forest["budget_key"])]
    ys = np.arange(len(forest))
    means = forest["mean_delta"].to_numpy(dtype=float)
    low = forest["ci_low"].to_numpy(dtype=float)
    high = forest["ci_high"].to_numpy(dtype=float)
    plt.figure(figsize=(8.4, max(3.2, 0.5 * len(forest) + 1.5)))
    plt.errorbar(means, ys, xerr=np.vstack([means - low, high - means]), fmt="o", capsize=4)
    plt.axvline(x=0.0, linewidth=1.0)
    plt.yticks(ys, labels)
    plt.xlabel("Mean Delta")
    plt.title("Effect Size Forest")
    plt.grid(True, axis="x", alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        target = fig2.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()
    return figure_paths


def write_significance_outputs(
    *,
    results_long: Any,
    manifest: ExperimentManifest,
    out_dir: str | Path,
    table_name: str = "table_significance_main",
    ci_table_name: str = "table_confidence_intervals",
) -> dict[str, Any]:
    lib = _require_pandas()
    out_root = Path(out_dir)
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_df, ci_df = build_significance_tables(results_long, manifest)
    table_csv = tables_dir / f"{table_name}.csv"
    table_md = tables_dir / f"{table_name}.md"
    ci_csv = tables_dir / f"{ci_table_name}.csv"
    ci_md = tables_dir / f"{ci_table_name}.md"
    _write_csv(table_csv, table_df)
    _write_text(table_md, "# Statistical Significance\n\n" + df_to_markdown_table(table_df))
    _write_csv(ci_csv, ci_df)
    _write_text(ci_md, "# Confidence Intervals\n\n" + df_to_markdown_table(ci_df))

    figure_paths = _make_significance_figures(table_df, figures_dir, list(manifest.output.figure_formats))
    status_counts = (
        table_df["status"].astype(str).value_counts(dropna=False).to_dict()
        if "status" in table_df.columns and len(table_df) > 0
        else {}
    )
    report_lines = [
        "# Statistical Significance",
        "",
        f"- suite_id: `{manifest.suite_id}`",
        f"- baseline: `{manifest.selection.labels.get(manifest.variants.baseline, manifest.variants.baseline)}`",
        f"- treatment: `{manifest.selection.labels.get(manifest.variants.treatment, manifest.variants.treatment)}`",
        f"- rows_total: `{len(table_df)}`",
        f"- status_counts: `{json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Main Table",
        "",
        df_to_markdown_table(table_df),
    ]
    report_path = out_root / "report.md"
    _write_text(report_path, "\n".join(report_lines))

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_id": manifest.suite_id,
        "tables": {
            "main_csv": str(table_csv),
            "main_md": str(table_md),
            "ci_csv": str(ci_csv),
            "ci_md": str(ci_md),
        },
        "figures": figure_paths,
        "rows_total": int(len(table_df)),
        "status_counts": status_counts,
    }
    snapshot_path = out_root / "snapshot.json"
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
    return {
        "table_df": table_df,
        "ci_df": ci_df,
        "table_csv": table_csv,
        "table_md": table_md,
        "ci_csv": ci_csv,
        "ci_md": ci_md,
        "figure_paths": figure_paths,
        "report_md": report_path,
        "snapshot_json": snapshot_path,
    }
