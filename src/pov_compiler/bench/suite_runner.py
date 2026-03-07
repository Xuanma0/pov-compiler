from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

from pov_compiler.bench.manifest import ExperimentManifest, load_manifest, write_resolved_manifest
from pov_compiler.bench.reporting.latex import df_to_markdown_table
from pov_compiler.bench.reporting.significance import build_significance_tables, write_significance_outputs
from pov_compiler.prompts.registry import PromptRegistry, write_prompt_lock


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for the benchmark suite runner.")
    return pd


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, df: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _read_csv(path: Path) -> Any:
    lib = _require_pandas()
    if not path.exists():
        return lib.DataFrame()
    try:
        return lib.read_csv(path)
    except Exception:
        return lib.DataFrame()


def _normalize_budget_fields(row: dict[str, Any]) -> tuple[str, float | None]:
    def _pick(*keys: str) -> float | None:
        for key in keys:
            if key not in row:
                continue
            try:
                value = float(row.get(key))
            except Exception:
                continue
            if value == value:
                return value
        return None

    budget_key = str(row.get("budget_key", row.get("budget_tag", ""))).strip()
    seconds = _pick("budget_seconds", "budget_max_total_s", "max_total_s")
    tokens = _pick("budget_max_tokens", "max_tokens")
    decisions = _pick("budget_max_decisions", "max_decisions")
    if not budget_key and seconds is not None and tokens is not None and decisions is not None:
        budget_key = f"{int(round(seconds))}/{int(tokens)}/{int(decisions)}"
    if not budget_key:
        budget_key = "unknown_budget"
    return budget_key, seconds


def _infer_sample_unit(row: dict[str, Any], source_kind: str, index: int) -> str:
    for key in ("sample_unit", "video_uid", "video_id", "uid", "qid", "query_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "aggregate" if str(source_kind) == "aggregate" else f"row_{int(index):06d}"


def _collect_source_rows(
    *,
    csv_path: Path,
    task: str,
    label_key: str,
    label_name: str,
    source_kind: str,
    manifest: ExperimentManifest,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = _read_csv(csv_path)
    run_entry = {
        "task": task,
        "label_key": label_key,
        "label_name": label_name,
        "source_kind": source_kind,
        "source_path": str(csv_path),
        "exists": bool(csv_path.exists()),
        "row_count": int(len(frame)),
        "status": "ok" if len(frame) > 0 else ("missing" if not csv_path.exists() else "empty"),
        "command": "",
    }
    rows: list[dict[str, Any]] = []
    if len(frame) == 0:
        return rows, run_entry

    primary_metric = str(manifest.metrics.primary.get(task, "")).strip()
    binary_metric = str(manifest.metrics.binary.get(task, "")).strip()
    for idx, payload in enumerate(frame.to_dict(orient="records")):
        row = dict(payload)
        budget_key, budget_seconds = _normalize_budget_fields(row)
        row.update(
            {
                "suite_id": manifest.suite_id,
                "task": task,
                "label_key": label_key,
                "variant_label": label_name,
                "source_kind": source_kind,
                "source_path": str(csv_path),
                "budget_key": budget_key,
                "budget_seconds": budget_seconds,
                "sample_unit": _infer_sample_unit(row, source_kind, idx),
                "primary_metric_name": primary_metric,
                "primary_metric_value": row.get(primary_metric) if primary_metric else None,
                "binary_metric_name": binary_metric,
                "binary_metric_value": row.get(binary_metric) if binary_metric else None,
            }
        )
        rows.append(row)
    return rows, run_entry


def _budget_keys_for_task(results_long: Any, manifest: ExperimentManifest, task: str) -> list[str]:
    configured = [point.key for point in manifest.budgets.points]
    seen = configured.copy()
    if len(results_long) == 0:
        return configured
    extra = (
        results_long.loc[results_long["task"].astype(str) == str(task), "budget_key"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    for key in extra:
        if key not in seen:
            seen.append(key)
    return seen


def build_main_results_table(results_long: Any, manifest: ExperimentManifest) -> Any:
    lib = _require_pandas()
    label_a_key = manifest.variants.baseline
    label_b_key = manifest.variants.treatment
    label_a = manifest.selection.labels.get(label_a_key, label_a_key)
    label_b = manifest.selection.labels.get(label_b_key, label_b_key)
    rows: list[dict[str, Any]] = []

    for task in manifest.selection.tasks:
        metric = str(manifest.metrics.primary.get(task, "")).strip()
        task_df = results_long.loc[results_long["task"].astype(str) == str(task)].copy()
        if len(task_df) > 0 and "source_kind" in task_df.columns:
            aggregate_rows = task_df.loc[task_df["source_kind"].astype(str) == "aggregate"]
            if len(aggregate_rows) > 0:
                task_df = aggregate_rows
        for budget_key in _budget_keys_for_task(results_long, manifest, task):
            group = task_df.loc[task_df["budget_key"].astype(str) == str(budget_key)].copy()
            point = manifest.budget_index().get(budget_key)
            budget_seconds = float(point.budget_seconds) if point and point.budget_seconds is not None else np.nan
            if len(group) > 0 and "budget_seconds" in group.columns:
                numeric_seconds = pd.to_numeric(group["budget_seconds"], errors="coerce").dropna()
                if len(numeric_seconds) > 0:
                    budget_seconds = float(numeric_seconds.iloc[0])
            row = {
                "task": task,
                "budget_key": budget_key,
                "budget_seconds": budget_seconds,
                "primary_metric": metric,
                "label_a": label_a,
                "label_b": label_b,
                "value_a": np.nan,
                "value_b": np.nan,
                "delta": np.nan,
                "n_rows_a": 0,
                "n_rows_b": 0,
                "status": "missing_rows",
            }
            if not metric:
                row["status"] = "metric_missing"
                rows.append(row)
                continue
            for key, out_col, count_col in (
                (label_a_key, "value_a", "n_rows_a"),
                (label_b_key, "value_b", "n_rows_b"),
            ):
                side = group.loc[group["label_key"].astype(str) == str(key)].copy()
                row[count_col] = int(len(side))
                if metric in side.columns and len(side) > 0:
                    values = pd.to_numeric(side[metric], errors="coerce").dropna()
                    if len(values) > 0:
                        row[out_col] = float(values.mean())
            if row["value_a"] == row["value_a"] and row["value_b"] == row["value_b"]:
                row["delta"] = float(row["value_b"] - row["value_a"])
                row["status"] = "ok"
            elif row["n_rows_a"] > 0 or row["n_rows_b"] > 0:
                row["status"] = "metric_missing"
            rows.append(row)

    if not rows:
        rows.append(
            {
                "task": "n/a",
                "budget_key": "n/a",
                "budget_seconds": 0.0,
                "primary_metric": "n/a",
                "label_a": label_a,
                "label_b": label_b,
                "value_a": np.nan,
                "value_b": np.nan,
                "delta": np.nan,
                "n_rows_a": 0,
                "n_rows_b": 0,
                "status": "missing_rows",
            }
        )
    return lib.DataFrame(rows)


def build_failure_attribution_table(results_long: Any, manifest: ExperimentManifest) -> Any:
    lib = _require_pandas()
    rows: list[dict[str, Any]] = []
    for task in manifest.selection.tasks:
        failure_metrics = list(manifest.metrics.failure.get(task, []))
        if not failure_metrics:
            continue
        task_df = results_long.loc[results_long["task"].astype(str) == str(task)].copy()
        if len(task_df) > 0 and "source_kind" in task_df.columns:
            aggregate_rows = task_df.loc[task_df["source_kind"].astype(str) == "aggregate"]
            if len(aggregate_rows) > 0:
                task_df = aggregate_rows
        for budget_key in _budget_keys_for_task(results_long, manifest, task):
            group = task_df.loc[task_df["budget_key"].astype(str) == str(budget_key)].copy()
            point = manifest.budget_index().get(budget_key)
            budget_seconds = float(point.budget_seconds) if point and point.budget_seconds is not None else np.nan
            for label_key, label_name in manifest.selection.labels.items():
                side = group.loc[group["label_key"].astype(str) == str(label_key)].copy()
                for metric in failure_metrics:
                    row = {
                        "task": task,
                        "budget_key": budget_key,
                        "budget_seconds": budget_seconds,
                        "label_key": label_key,
                        "label": label_name,
                        "failure_metric": metric,
                        "failure_reason": metric.replace("safety_reason_", "").replace("_rate", ""),
                        "rate": np.nan,
                        "status": "missing_rows",
                    }
                    if metric in side.columns and len(side) > 0:
                        values = pd.to_numeric(side[metric], errors="coerce").dropna()
                        if len(values) > 0:
                            row["rate"] = float(values.mean())
                            row["status"] = "ok"
                        else:
                            row["status"] = "metric_missing"
                    elif len(side) > 0:
                        row["status"] = "metric_missing"
                    rows.append(row)
    if not rows:
        rows.append(
            {
                "task": "n/a",
                "budget_key": "n/a",
                "budget_seconds": 0.0,
                "label_key": "n/a",
                "label": "n/a",
                "failure_metric": "n/a",
                "failure_reason": "insufficient_pairs",
                "rate": np.nan,
                "status": "no_failure_metrics",
            }
        )
    return lib.DataFrame(rows)


def _save_placeholder_figure(path_base: Path, title: str, message: str, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    figure_paths: list[str] = []
    plt.figure(figsize=(8.2, 4.6))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    for ext in formats:
        target = path_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()
    return figure_paths


def write_main_figures(main_df: Any, failure_df: Any, out_dir: Path, formats: list[str]) -> dict[str, list[str]]:
    import matplotlib.pyplot as plt

    figures: dict[str, list[str]] = {}
    valid_main = main_df.loc[main_df["status"].astype(str) == "ok"].copy() if len(main_df) > 0 else main_df.copy()
    if valid_main.empty:
        figures["budget_frontier"] = _save_placeholder_figure(
            out_dir / "fig_main_budget_frontier",
            "Main Budget Frontier",
            "No valid main-result rows available",
            formats,
        )
        figures["variant_delta"] = _save_placeholder_figure(
            out_dir / "fig_main_variant_delta",
            "Main Variant Delta",
            "No valid main-result rows available",
            formats,
        )
    else:
        valid_main["budget_seconds"] = pd.to_numeric(valid_main["budget_seconds"], errors="coerce")
        fig1 = out_dir / "fig_main_budget_frontier"
        plt.figure(figsize=(8.4, 4.8))
        for task in sorted(valid_main["task"].astype(str).unique().tolist()):
            task_rows = valid_main.loc[valid_main["task"].astype(str) == str(task)].sort_values("budget_seconds")
            plt.plot(task_rows["budget_seconds"], task_rows["value_a"], marker="o", linestyle="-", label=f"{task}:{task_rows['label_a'].iloc[0]}")
            plt.plot(task_rows["budget_seconds"], task_rows["value_b"], marker="o", linestyle="--", label=f"{task}:{task_rows['label_b'].iloc[0]}")
        plt.xlabel("Budget Seconds")
        plt.ylabel("Primary Metric")
        plt.title("Main Budget Frontier")
        plt.grid(True, alpha=0.35)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        figures["budget_frontier"] = []
        for ext in formats:
            target = fig1.with_suffix(f".{ext}")
            plt.savefig(target)
            figures["budget_frontier"].append(str(target))
        plt.close()

        fig2 = out_dir / "fig_main_variant_delta"
        plt.figure(figsize=(8.4, 4.8))
        for task in sorted(valid_main["task"].astype(str).unique().tolist()):
            task_rows = valid_main.loc[valid_main["task"].astype(str) == str(task)].sort_values("budget_seconds")
            plt.plot(task_rows["budget_seconds"], task_rows["delta"], marker="o", label=str(task))
        plt.axhline(y=0.0, linewidth=1.0)
        plt.xlabel("Budget Seconds")
        plt.ylabel("Delta")
        plt.title("Main Variant Delta")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        figures["variant_delta"] = []
        for ext in formats:
            target = fig2.with_suffix(f".{ext}")
            plt.savefig(target)
            figures["variant_delta"].append(str(target))
        plt.close()

    valid_failure = failure_df.loc[failure_df["status"].astype(str) == "ok"].copy() if len(failure_df) > 0 else failure_df.copy()
    if valid_failure.empty:
        figures["failure_attribution"] = _save_placeholder_figure(
            out_dir / "fig_main_failure_attribution",
            "Main Failure Attribution",
            "No valid failure-attribution rows available",
            formats,
        )
    else:
        valid_failure["budget_seconds"] = pd.to_numeric(valid_failure["budget_seconds"], errors="coerce")
        pivot = (
            valid_failure.groupby(["budget_seconds", "label", "failure_reason"], dropna=False)["rate"]
            .mean()
            .reset_index()
            .pivot_table(index=["budget_seconds", "label"], columns="failure_reason", values="rate", aggfunc="mean")
            .fillna(0.0)
            .reset_index()
        )
        labels = [f"{int(row['budget_seconds'])}:{row['label']}" for _, row in pivot.iterrows()]
        fig3 = out_dir / "fig_main_failure_attribution"
        plt.figure(figsize=(9.0, 4.8))
        bottom = np.zeros(len(pivot))
        reason_cols = [col for col in pivot.columns if col not in {"budget_seconds", "label"}]
        x = np.arange(len(pivot))
        for reason in reason_cols:
            values = pivot[reason].to_numpy(dtype=float)
            plt.bar(x, values, bottom=bottom, label=str(reason))
            bottom += values
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Failure Rate")
        plt.title("Main Failure Attribution")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        figures["failure_attribution"] = []
        for ext in formats:
            target = fig3.with_suffix(f".{ext}")
            plt.savefig(target)
            figures["failure_attribution"].append(str(target))
        plt.close()
    return figures


def _copy_manifest(original_manifest_path: Path, manifest_dir: Path) -> tuple[Path, Path]:
    import shutil

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_copy = manifest_dir / "experiment_manifest.yaml"
    shutil.copyfile(original_manifest_path, manifest_copy)
    return manifest_copy, manifest_dir / "manifest_resolved.json"


def _build_runs_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_compare_readme(
    *,
    manifest: ExperimentManifest,
    compare_dir: Path,
    results_long_path: Path,
    main_table_path: Path,
    significance_path: Path,
    failure_table_path: Path,
) -> str:
    lines = [
        "# Benchmark Suite Compare",
        "",
        f"- suite_id: `{manifest.suite_id}`",
        f"- suite_version: `{manifest.suite_version}`",
        f"- compare_dir: `{compare_dir}`",
        f"- labels: `{json.dumps(manifest.selection.labels, ensure_ascii=False, sort_keys=True)}`",
        f"- tasks: `{manifest.selection.tasks}`",
        "",
        "## Key Outputs",
        "",
        f"- ledger: `{results_long_path}`",
        f"- main table: `{main_table_path}`",
        f"- significance table: `{significance_path}`",
        f"- failure attribution table: `{failure_table_path}`",
        "",
        "This suite runner only collects, aggregates, and exports from existing compare/sweep outputs.",
    ]
    return "\n".join(lines)


def run_suite(
    *,
    manifest_path: str | Path,
    out_dir: str | Path | None = None,
    compare_dir_override: str | Path | None = None,
    mode: str = "collect-only",
) -> dict[str, Any]:
    lib = _require_pandas()
    manifest_path = Path(manifest_path).resolve()
    manifest, resolved = load_manifest(manifest_path)
    compare_dir = Path(compare_dir_override).resolve() if compare_dir_override else Path(str(resolved["selection"]["compare_dir"]))
    out_root = Path(out_dir).resolve() if out_dir else Path(str(resolved["output"]["root"]))
    manifest_dir = out_root / "manifest"
    ledger_dir = out_root / "ledger"
    compare_out_dir = out_root / "compare"
    tables_dir = compare_out_dir / "tables"
    figures_dir = compare_out_dir / "figures"
    significance_dir = out_root / "significance"
    for path in (manifest_dir, ledger_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_copy, resolved_path = _copy_manifest(manifest_path, manifest_dir)
    write_resolved_manifest(resolved, resolved_path)

    prompt_lock_path: Path | None = None
    registry_path_raw = resolved.get("prompts", {}).get("registry")
    if registry_path_raw:
        registry_path = Path(str(registry_path_raw))
        registry = PromptRegistry.from_path(registry_path)
        prompt_lock = registry.build_prompt_lock(registry_path, manifest.prompts.profile)
        prompt_lock_path = write_prompt_lock(prompt_lock, manifest_dir / "prompt_lock.json")

    rows: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []
    task_sources = dict(manifest.selection.task_sources)
    pair_sources = dict(manifest.selection.pair_sources)

    for task in manifest.selection.tasks:
        task_pattern = str(task_sources.get(task, "")).strip()
        pair_pattern = str(pair_sources.get(task, "")).strip()
        for label_key, label_name in manifest.selection.labels.items():
            if task_pattern:
                source_path = compare_dir / Path(task_pattern.format(label=label_name, label_key=label_key, task=task))
                source_rows, run_entry = _collect_source_rows(
                    csv_path=source_path,
                    task=task,
                    label_key=label_key,
                    label_name=label_name,
                    source_kind="aggregate",
                    manifest=manifest,
                )
                rows.extend(source_rows)
                run_entries.append(run_entry)
            if pair_pattern:
                source_path = compare_dir / Path(pair_pattern.format(label=label_name, label_key=label_key, task=task))
                source_rows, run_entry = _collect_source_rows(
                    csv_path=source_path,
                    task=task,
                    label_key=label_key,
                    label_name=label_name,
                    source_kind="pair",
                    manifest=manifest,
                )
                rows.extend(source_rows)
                run_entries.append(run_entry)

    results_long = lib.DataFrame(rows)
    if len(results_long) == 0:
        results_long = lib.DataFrame(
            columns=[
                "suite_id",
                "task",
                "label_key",
                "variant_label",
                "source_kind",
                "source_path",
                "budget_key",
                "budget_seconds",
                "sample_unit",
                "primary_metric_name",
                "primary_metric_value",
            ]
        )
    results_long_path = ledger_dir / "results_long.csv"
    _write_csv(results_long_path, results_long)
    runs_jsonl_path = ledger_dir / "runs.jsonl"
    _build_runs_jsonl(runs_jsonl_path, run_entries)

    main_table = build_main_results_table(results_long, manifest)
    failure_table = build_failure_attribution_table(results_long, manifest)
    significance_table, _ = build_significance_tables(results_long, manifest)

    table_main_csv = tables_dir / "table_main_results.csv"
    table_main_md = tables_dir / "table_main_results.md"
    table_sig_csv = tables_dir / "table_significance.csv"
    table_sig_md = tables_dir / "table_significance.md"
    table_fail_csv = tables_dir / "table_failure_attribution.csv"
    table_fail_md = tables_dir / "table_failure_attribution.md"
    _write_csv(table_main_csv, main_table)
    _write_text(table_main_md, "# Main Results\n\n" + df_to_markdown_table(main_table))
    _write_csv(table_sig_csv, significance_table)
    _write_text(table_sig_md, "# Statistical Significance\n\n" + df_to_markdown_table(significance_table))
    _write_csv(table_fail_csv, failure_table)
    _write_text(table_fail_md, "# Failure Attribution\n\n" + df_to_markdown_table(failure_table))

    figure_paths = write_main_figures(main_table, failure_table, figures_dir, list(manifest.output.figure_formats))
    significance_outputs = write_significance_outputs(results_long=results_long, manifest=manifest, out_dir=significance_dir)

    main_valid = main_table.loc[main_table["status"].astype(str) == "ok"].copy()
    task_summary: dict[str, Any] = {}
    for task in manifest.selection.tasks:
        task_rows = main_valid.loc[main_valid["task"].astype(str) == str(task)].copy()
        if task_rows.empty:
            task_summary[task] = {"rows": 0, "best_budget_key": "", "best_delta": None}
            continue
        task_rows["delta"] = pd.to_numeric(task_rows["delta"], errors="coerce")
        best_idx = task_rows["delta"].idxmax()
        best = task_rows.loc[best_idx]
        task_summary[task] = {
            "rows": int(len(task_rows)),
            "best_budget_key": str(best.get("budget_key", "")),
            "best_delta": float(best.get("delta")) if best.get("delta") == best.get("delta") else None,
        }

    compare_summary = {
        "suite_id": manifest.suite_id,
        "suite_version": manifest.suite_version,
        "mode": mode,
        "compare_dir": str(compare_dir),
        "results_rows": int(len(results_long)),
        "run_entries": int(len(run_entries)),
        "missing_sources": int(sum(1 for row in run_entries if str(row.get("status", "")) != "ok")),
        "tasks": task_summary,
        "prompt_lock": str(prompt_lock_path) if prompt_lock_path is not None else None,
        "significance_status_counts": (
            significance_table["status"].astype(str).value_counts(dropna=False).to_dict()
            if "status" in significance_table.columns and len(significance_table) > 0
            else {}
        ),
    }
    compare_summary_path = compare_out_dir / "compare_summary.json"
    _write_text(compare_summary_path, json.dumps(compare_summary, ensure_ascii=False, indent=2))

    commands_lines = [
        "#!/usr/bin/env bash",
        f"# Generated by run_benchmark_suite.py for suite `{manifest.suite_id}`",
        f"# mode={mode}",
    ]
    if mode == "collect-only":
        commands_lines.append("# No producer commands were executed; artifacts were collected from existing compare outputs.")

    paper_ready_dir: Path | None = None
    if str(mode).lower() == "export":
        paper_ready_dir = out_root / "paper_ready"
        export_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / "scripts" / "export_paper_ready.py"),
            "--compare_dir",
            str(compare_dir),
            "--out_dir",
            str(paper_ready_dir),
            "--suite-dir",
            str(out_root),
            "--significance-dir",
            str(significance_dir),
            "--export-submission-pack",
        ]
        if registry_path_raw:
            export_cmd.extend(["--prompt-registry", str(registry_path_raw)])
        if prompt_lock_path is not None:
            export_cmd.extend(["--prompt-lock", str(prompt_lock_path)])
        commands_lines.append(" ".join(export_cmd))
        subprocess.run(export_cmd, cwd=str(Path(__file__).resolve().parents[3]), check=False)
    for row in run_entries:
        commands_lines.append(f"# source[{row['task']}][{row['label_name']}][{row['source_kind']}]={row['source_path']}")

    commands_path = compare_out_dir / "commands.sh"
    _write_text(commands_path, "\n".join(commands_lines) + "\n")

    compare_readme = _build_compare_readme(
        manifest=manifest,
        compare_dir=compare_dir,
        results_long_path=results_long_path,
        main_table_path=table_main_csv,
        significance_path=table_sig_csv,
        failure_table_path=table_fail_csv,
    )
    compare_readme_path = compare_out_dir / "README.md"
    _write_text(compare_readme_path, compare_readme)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_id": manifest.suite_id,
        "suite_version": manifest.suite_version,
        "mode": mode,
        "manifest_copy": str(manifest_copy),
        "manifest_resolved": str(resolved_path),
        "compare_dir": str(compare_dir),
        "ledger": {
            "results_long_csv": str(results_long_path),
            "runs_jsonl": str(runs_jsonl_path),
        },
        "compare": {
            "table_main_results_csv": str(table_main_csv),
            "table_significance_csv": str(table_sig_csv),
            "table_failure_attribution_csv": str(table_fail_csv),
            "figures": figure_paths,
            "compare_summary_json": str(compare_summary_path),
            "commands_sh": str(commands_path),
            "readme_md": str(compare_readme_path),
        },
        "significance": {
            "table_main_csv": str(significance_outputs["table_csv"]),
            "ci_csv": str(significance_outputs["ci_csv"]),
            "snapshot_json": str(significance_outputs["snapshot_json"]),
        },
        "prompt_lock": str(prompt_lock_path) if prompt_lock_path is not None else None,
        "paper_ready_dir": str(paper_ready_dir) if paper_ready_dir is not None else None,
    }
    snapshot_path = compare_out_dir / "snapshot.json"
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))

    return {
        "manifest_copy": manifest_copy,
        "resolved_manifest": resolved_path,
        "prompt_lock": prompt_lock_path,
        "results_long_csv": results_long_path,
        "runs_jsonl": runs_jsonl_path,
        "table_main_csv": table_main_csv,
        "table_significance_csv": table_sig_csv,
        "table_failure_csv": table_fail_csv,
        "figure_paths": figure_paths,
        "compare_summary_json": compare_summary_path,
        "snapshot_json": snapshot_path,
        "commands_sh": commands_path,
        "readme_md": compare_readme_path,
        "significance_dir": significance_dir,
        "paper_ready_dir": paper_ready_dir,
    }
