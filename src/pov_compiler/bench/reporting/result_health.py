from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

from pov_compiler.bench.query_bank import selection_stats
from pov_compiler.bench.reporting.health_gate import evaluate_health_gate
from pov_compiler.bench.reporting.latex import df_to_markdown_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for result-health reporting.")
    return pd


def _read_csv(path: Path) -> Any:
    lib = _require_pandas()
    if not path.exists():
        return lib.DataFrame()
    try:
        return lib.read_csv(path)
    except Exception:
        return lib.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - dependency should exist in runtime env.
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, df: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _placeholder_figure(path_base: Path, title: str, message: str, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_paths: list[str] = []
    plt.figure(figsize=(8.2, 4.6))
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


def _load_runs_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _effect_nonzero(frame: Any, epsilon: float) -> Any:
    lib = _require_pandas()
    work = frame.copy()
    if "mean_delta" in work.columns:
        values = lib.to_numeric(work["mean_delta"], errors="coerce")
    elif "delta" in work.columns:
        values = lib.to_numeric(work["delta"], errors="coerce")
    else:
        values = lib.Series([None] * len(work))
    return values.abs() > float(epsilon)


def build_result_health_table(
    *,
    suite_dir: str | Path,
    epsilon: float = 1e-9,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    compare_dir = suite_root / "compare"
    manifest_path = suite_root / "manifest" / "experiment_manifest.yaml"
    compare_summary = _read_json(compare_dir / "compare_summary.json")
    compare_snapshot = _read_json(compare_dir / "snapshot.json")
    main_df = _read_csv(compare_dir / "tables" / "table_main_results.csv")
    significance_df = _read_csv(compare_dir / "tables" / "table_significance.csv")
    detail_sig_df = _read_csv(suite_root / "significance" / "tables" / "table_significance_main.csv")
    runs = _load_runs_jsonl(suite_root / "ledger" / "runs.jsonl")
    selection = selection_stats(compare_dir=compare_summary.get("compare_dir", compare_dir), manifest_path=manifest_path)

    if detail_sig_df.empty:
        detail_sig_df = significance_df.copy()
    if main_df.empty:
        main_df = lib.DataFrame(
            [
                {
                    "task": "overall",
                    "budget_key": "n/a",
                    "budget_seconds": 0.0,
                    "status": "missing_rows",
                    "delta": None,
                }
            ]
        )

    tasks = [str(value) for value in main_df["task"].astype(str).unique().tolist() if str(value).strip()]
    tasks = [task for task in tasks if task != "n/a"] or ["overall"]

    rows: list[dict[str, Any]] = []
    overall_reason_counts: dict[str, int] = {}

    for task in [*tasks, "overall"]:
        task_main = main_df if task == "overall" else main_df.loc[main_df["task"].astype(str) == str(task)].copy()
        task_sig = detail_sig_df if task == "overall" else detail_sig_df.loc[detail_sig_df["task"].astype(str) == str(task)].copy()
        task_runs = runs if task == "overall" else [row for row in runs if str(row.get("task", "")) == str(task)]
        rows_total = int(len(task_main))
        if rows_total <= 0:
            rows_total = 1

        missing_sources_count = int(sum(1 for row in task_runs if str(row.get("status", "")) == "missing"))
        empty_sources_count = int(sum(1 for row in task_runs if str(row.get("status", "")) == "empty"))
        missing_metric_count = int((task_main.get("status", lib.Series(dtype=str)).astype(str) == "metric_missing").sum()) if len(task_main) > 0 else 0
        missing_rows_count = int((task_main.get("status", lib.Series(dtype=str)).astype(str) == "missing_rows").sum()) if len(task_main) > 0 else rows_total
        insufficient_pairs_count = int((task_sig.get("status", lib.Series(dtype=str)).astype(str) == "insufficient_pairs").sum()) if len(task_sig) > 0 else 0
        zero_delta_count = 0
        if len(task_main) > 0 and "delta" in task_main.columns:
            delta_vals = lib.to_numeric(task_main["delta"], errors="coerce")
            status_vals = task_main.get("status", lib.Series([""] * len(task_main)))
            zero_delta_count = int(((status_vals.astype(str) == "ok") & (delta_vals.abs() <= float(epsilon))).sum())

        significance_available_count = int(len(task_sig.loc[task_sig.get("status", lib.Series(dtype=str)).astype(str) != "task_missing"])) if len(task_sig) > 0 else 0
        effect_size_nonzero_count = int(_effect_nonzero(task_sig if len(task_sig) > 0 else task_main, epsilon).sum()) if rows_total > 0 else 0

        reason_counts = {
            "source_missing": missing_sources_count,
            "source_empty": empty_sources_count,
            "missing_rows": missing_rows_count,
            "missing_metric": missing_metric_count,
            "insufficient_pairs": insufficient_pairs_count,
            "ok": max(rows_total - max(missing_rows_count, missing_metric_count) - insufficient_pairs_count, 0),
        }
        for key, value in reason_counts.items():
            overall_reason_counts[key] = int(overall_reason_counts.get(key, 0) + int(value))

        if missing_sources_count > 0:
            no_data_reason = "source_missing"
        elif empty_sources_count > 0:
            no_data_reason = "source_empty"
        elif missing_metric_count > 0:
            no_data_reason = "missing_metric"
        elif missing_rows_count > 0:
            no_data_reason = "missing_rows"
        elif insufficient_pairs_count > 0:
            no_data_reason = "insufficient_pairs"
        elif zero_delta_count > 0:
            no_data_reason = "zero_delta"
        else:
            no_data_reason = "ok"

        coverage_stats = selection.get("coverage_score_stats", {})
        row = {
            "task": task,
            "rows_total": int(rows_total),
            "missing_metric_rate": float(missing_metric_count / rows_total),
            "zero_delta_rate": float(zero_delta_count / rows_total),
            "no_data_reason": no_data_reason,
            "no_data_reason_breakdown": json.dumps(reason_counts, ensure_ascii=False, sort_keys=True),
            "significance_available_rate": float(min(significance_available_count, rows_total) / rows_total),
            "effect_size_nonzero_rate": float(min(effect_size_nonzero_count, rows_total) / rows_total),
            "selected_uids_count": int(selection.get("selected_uids_count", 0) or 0),
            "coverage_score_stats": json.dumps(coverage_stats, ensure_ascii=False, sort_keys=True),
            "missing_sources_count": int(missing_sources_count),
            "insufficient_pairs_count": int(insufficient_pairs_count),
            "selection_mode": str(selection.get("selection_mode", "")),
            "missing_signal_breakdown": json.dumps(selection.get("missing_signal_breakdown", {}), ensure_ascii=False, sort_keys=True),
            "health_status": "ok" if no_data_reason == "ok" else ("degraded" if no_data_reason in {"zero_delta", "insufficient_pairs"} else "no_data"),
        }
        rows.append(row)

    out_df = lib.DataFrame(rows)
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "compare_summary": compare_summary,
        "compare_snapshot_keys": sorted(compare_snapshot.keys()),
        "rows_total": int(len(out_df)),
        "overall_no_data_reason_counts": overall_reason_counts,
        "selection": selection,
    }
    return out_df, snapshot


def write_result_health_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
    epsilon: float = 1e-9,
    figure_formats: list[str] | None = None,
    gate_profile: str | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    lib = _require_pandas()
    formats = figure_formats or ["png", "pdf"]
    out_root = Path(out_dir)
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_df, snapshot = build_result_health_table(suite_dir=suite_dir, epsilon=epsilon)
    suite_root = Path(suite_dir).resolve()
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    resolved_gate_profile = str(gate_profile or manifest_payload.get("health_gate_profile", "")).strip() or None

    table_csv = tables_dir / "table_result_health.csv"
    table_md = tables_dir / "table_result_health.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv(table_csv, out_df)
    _write_text(table_md, "# Result Health\n\n" + df_to_markdown_table(out_df))
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))

    if out_df.empty:
        figure_paths = _placeholder_figure(
            figures_dir / "fig_result_health_breakdown",
            "Result Health Breakdown",
            "No health rows available",
            formats,
        )
    else:
        plot_df = out_df.loc[out_df["task"].astype(str) != "overall"].copy()
        if plot_df.empty:
            plot_df = out_df.copy()
        plot_df["missing_metric_rate"] = lib.to_numeric(plot_df["missing_metric_rate"], errors="coerce").fillna(0.0)
        plot_df["zero_delta_rate"] = lib.to_numeric(plot_df["zero_delta_rate"], errors="coerce").fillna(0.0)
        plot_df["significance_available_rate"] = lib.to_numeric(plot_df["significance_available_rate"], errors="coerce").fillna(0.0)
        labels = plot_df["task"].astype(str).tolist()
        x = list(range(len(plot_df)))
        width = 0.25
        fig_base = figures_dir / "fig_result_health_breakdown"
        plt.figure(figsize=(8.8, 4.8))
        plt.bar([item - width for item in x], plot_df["missing_metric_rate"], width=width, label="missing_metric_rate")
        plt.bar(x, plot_df["zero_delta_rate"], width=width, label="zero_delta_rate")
        plt.bar([item + width for item in x], 1.0 - plot_df["significance_available_rate"], width=width, label="1-significance_available_rate")
        plt.xticks(x, labels)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Rate")
        plt.title("Result Health Breakdown")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        figure_paths = []
        for ext in formats:
            target = fig_base.with_suffix(f".{ext}")
            plt.savefig(target)
            figure_paths.append(str(target))
        plt.close()

    gate = evaluate_health_gate(out_df, snapshot, resolved_gate_profile)
    snapshot["outputs"] = {
        "table_csv": str(table_csv),
        "table_md": str(table_md),
        "figures": figure_paths,
        "snapshot_json": str(snapshot_path),
    }
    snapshot["gate"] = gate
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "figure_paths": figure_paths,
        "snapshot_json": snapshot_path,
        "rows_total": int(len(out_df)),
        "no_data_reason_counts": snapshot.get("overall_no_data_reason_counts", {}),
        "gate_status": gate.get("gate_status", "skipped"),
        "gate_fail_reasons": gate.get("gate_fail_reasons", []),
        "gate_profile": gate.get("profile", ""),
    }
