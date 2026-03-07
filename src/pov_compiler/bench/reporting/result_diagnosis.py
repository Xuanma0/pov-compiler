from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

from pov_compiler.bench.reporting.latex import df_to_markdown_table
from pov_compiler.bench.reporting.result_health import build_result_health_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for result-diagnosis reporting.")
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


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _provider_noise_summary(results_df: Any, task: str, telemetry_df: Any | None = None) -> dict[str, Any]:
    lib = _require_pandas()
    frame = results_df.copy()
    if task != "overall" and "task" in frame.columns:
        frame = frame.loc[frame["task"].astype(str) == task].copy()
    extra = telemetry_df if telemetry_df is not None else lib.DataFrame()
    if task != "overall" and len(extra) > 0 and "task" in extra.columns:
        extra = extra.loc[extra["task"].astype(str) == task].copy()

    metric_map = {
        "model_cost_known_rate": ["model_cost_known_rate", "cost_known_rate", "model_cost_rate"],
        "model_latency_p95_ms_mean": ["model_latency_p95_ms_mean", "model_latency_p95_ms", "latency_p95_ms"],
        "structured_parse_fail_rate_mean": [
            "structured_parse_fail_rate_mean",
            "structured_parse_fail_rate",
            "parse_fail_rate",
        ],
        "planner_fallback_rate": ["planner_fallback_rate", "fallback_rate"],
    }
    out_metrics: dict[str, Any] = {}
    available_total = 0
    sources = [frame, extra]
    for metric_name, candidates in metric_map.items():
        chosen = None
        chosen_frame = None
        for candidate in candidates:
            for source in sources:
                if len(source) > 0 and candidate in source.columns:
                    chosen = candidate
                    chosen_frame = source
                    break
            if chosen is not None:
                break
        if chosen is None or chosen_frame is None:
            out_metrics[metric_name] = {"available": False, "value": None, "source_column": None}
            continue
        values = lib.to_numeric(chosen_frame[chosen], errors="coerce").dropna()
        if values.empty:
            out_metrics[metric_name] = {"available": False, "value": None, "source_column": chosen}
            continue
        available_total += 1
        out_metrics[metric_name] = {
            "available": True,
            "value": float(values.mean()),
            "source_column": chosen,
        }

    availability = "unavailable" if available_total == 0 else ("partial" if available_total < len(metric_map) else "ok")
    return {
        "availability": availability,
        "metrics": out_metrics,
    }


def _recommendations_for_row(
    *,
    row: dict[str, Any],
    provider_noise_summary: dict[str, Any],
    near_zero_threshold: float,
    effect_size_threshold: float,
    significance_threshold: float,
) -> list[str]:
    recommendations: list[str] = []
    selected_uids_count = int(_to_float(row.get("selected_uids_count")) or 0)
    missing_metric_rate = float(_to_float(row.get("missing_metric_rate")) or 0.0)
    near_zero_delta_rate = float(_to_float(row.get("near_zero_delta_rate")) or 0.0)
    effect_size_nonzero_rate = float(_to_float(row.get("effect_size_nonzero_rate")) or 0.0)
    significance_available_rate = float(_to_float(row.get("significance_available_rate")) or 0.0)
    no_data_reason = str(row.get("no_data_reason", "")).strip()
    if selected_uids_count < 3:
        recommendations.append("increase selected_uids_count")
    if no_data_reason in {"source_missing", "source_empty", "missing_rows"}:
        recommendations.append("verify compare producer outputs before main_real pilot")
    if no_data_reason == "missing_metric" or missing_metric_rate > 0.0:
        recommendations.append("audit metric schema for missing primary metrics")
    if significance_available_rate < significance_threshold:
        recommendations.append("increase paired sample coverage")
    if near_zero_delta_rate >= 0.5 and effect_size_nonzero_rate <= effect_size_threshold:
        recommendations.append("switch query bank to chain-heavy set")
    if near_zero_delta_rate >= 0.5 and no_data_reason == "ok":
        recommendations.append(f"raise signal_min_score or tighten pilot budget selection (near_zero<{near_zero_threshold:g})")
    metrics = provider_noise_summary.get("metrics", {}) if isinstance(provider_noise_summary, dict) else {}
    parse_fail = _to_float(((metrics.get("structured_parse_fail_rate_mean") or {}).get("value")))
    fallback = _to_float(((metrics.get("planner_fallback_rate") or {}).get("value")))
    if (parse_fail is not None and parse_fail > 0.1) or (fallback is not None and fallback > 0.1):
        recommendations.append("real provider noise dominates deltas")
    if not recommendations:
        recommendations.append("pilot looks healthy; keep the current manifest and freeze outputs")
    return recommendations


def build_result_diagnosis_table(
    *,
    suite_dir: str | Path,
    near_zero_threshold: float = 1e-6,
    effect_size_threshold: float = 1e-9,
    significance_threshold: float = 0.50,
    provider_telemetry_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    compare_dir = suite_root / "compare"
    health_df, health_snapshot = build_result_health_table(suite_dir=suite_root, epsilon=effect_size_threshold)
    main_df = _read_csv(compare_dir / "tables" / "table_main_results.csv")
    results_df = _read_csv(suite_root / "ledger" / "results_long.csv")
    health_snapshot_file = _read_json(suite_root / "result_health" / "snapshot.json")
    detail_sig_df = _read_csv(suite_root / "significance" / "tables" / "table_significance_main.csv")

    telemetry_df = lib.DataFrame()
    if provider_telemetry_dir:
        telemetry_root = Path(provider_telemetry_dir).resolve()
        csv_files = sorted(telemetry_root.glob("*.csv"))
        frames = [_read_csv(path) for path in csv_files]
        frames = [frame for frame in frames if len(frame) > 0]
        if frames:
            telemetry_df = lib.concat(frames, ignore_index=True)

    if health_df.empty:
        health_df = lib.DataFrame(
            [
                {
                    "task": "overall",
                    "rows_total": 1,
                    "missing_metric_rate": 1.0,
                    "zero_delta_rate": 0.0,
                    "no_data_reason": "source_missing",
                    "no_data_reason_breakdown": json.dumps({"source_missing": 1}, ensure_ascii=False),
                    "significance_available_rate": 0.0,
                    "effect_size_nonzero_rate": 0.0,
                    "selected_uids_count": 0,
                    "coverage_score_stats": json.dumps({}, ensure_ascii=False),
                    "missing_sources_count": 1,
                    "insufficient_pairs_count": 0,
                    "selection_mode": "",
                    "missing_signal_breakdown": json.dumps({}, ensure_ascii=False),
                    "health_status": "no_data",
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    for _, health_row in health_df.iterrows():
        task = str(health_row.get("task", "")).strip() or "overall"
        task_main = main_df.copy()
        if task != "overall" and "task" in task_main.columns:
            task_main = task_main.loc[task_main["task"].astype(str) == task].copy()
        if len(task_main) > 0 and "status" in task_main.columns:
            task_main = task_main.loc[task_main["status"].astype(str).isin(["ok", "metric_missing", "missing_rows"])]
        rows_total = int(_to_float(health_row.get("rows_total")) or len(task_main) or 1)
        if len(task_main) > 0 and "delta" in task_main.columns:
            delta_vals = lib.to_numeric(task_main["delta"], errors="coerce")
            near_zero_count = int((delta_vals.abs() < float(near_zero_threshold)).sum())
        else:
            near_zero_count = rows_total
        near_zero_delta_rate = float(near_zero_count / rows_total) if rows_total > 0 else 1.0

        provider_summary = _provider_noise_summary(results_df, task, telemetry_df=telemetry_df)
        row_payload = health_row.to_dict()
        row_payload["near_zero_delta_rate"] = near_zero_delta_rate
        recommendations = _recommendations_for_row(
            row=row_payload,
            provider_noise_summary=provider_summary,
            near_zero_threshold=near_zero_threshold,
            effect_size_threshold=effect_size_threshold,
            significance_threshold=significance_threshold,
        )
        rows.append(
            {
                "task": task,
                "rows_total": rows_total,
                "no_data_reason": str(health_row.get("no_data_reason", "")),
                "no_data_reason_breakdown": str(health_row.get("no_data_reason_breakdown", "")),
                "missing_metric_rate": float(_to_float(health_row.get("missing_metric_rate")) or 0.0),
                "near_zero_delta_rate": near_zero_delta_rate,
                "effect_size_nonzero_rate": float(_to_float(health_row.get("effect_size_nonzero_rate")) or 0.0),
                "significance_available_rate": float(_to_float(health_row.get("significance_available_rate")) or 0.0),
                "selected_uids_count": int(_to_float(health_row.get("selected_uids_count")) or 0),
                "coverage_score_stats": str(health_row.get("coverage_score_stats", "")),
                "missing_sources_count": int(_to_float(health_row.get("missing_sources_count")) or 0),
                "insufficient_pairs_count": int(_to_float(health_row.get("insufficient_pairs_count")) or 0),
                "provider_noise_summary": json.dumps(provider_summary, ensure_ascii=False, sort_keys=True),
                "diagnosis_recommendations": json.dumps(recommendations, ensure_ascii=False),
            }
        )

    out_df = lib.DataFrame(rows)
    overall_row = out_df.loc[out_df["task"].astype(str) == "overall"].copy()
    overall_payload = overall_row.iloc[0].to_dict() if len(overall_row) > 0 else (out_df.iloc[0].to_dict() if len(out_df) > 0 else {})
    overall_provider_summary = _parse_json_dict(overall_payload.get("provider_noise_summary"))
    overall_recommendations = []
    raw_recommendations = overall_payload.get("diagnosis_recommendations")
    if isinstance(raw_recommendations, str) and raw_recommendations.strip():
        try:
            parsed = json.loads(raw_recommendations)
        except Exception:
            parsed = []
        if isinstance(parsed, list):
            overall_recommendations = [str(item) for item in parsed]
    provider_fields = {}
    metrics = overall_provider_summary.get("metrics", {}) if isinstance(overall_provider_summary, dict) else {}
    for key in (
        "model_cost_known_rate",
        "model_latency_p95_ms_mean",
        "structured_parse_fail_rate_mean",
        "planner_fallback_rate",
    ):
        metric_payload = metrics.get(key, {}) if isinstance(metrics, dict) else {}
        provider_fields[key] = metric_payload.get("value", "unavailable") if metric_payload.get("available") else "unavailable"

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "thresholds": {
            "near_zero_threshold": float(near_zero_threshold),
            "effect_size_threshold": float(effect_size_threshold),
            "significance_threshold": float(significance_threshold),
        },
        "rows_total": int(len(out_df)),
        "overall_no_data_reason_counts": health_snapshot.get("overall_no_data_reason_counts", {}),
        "selected_uids_count": int(_to_float(overall_payload.get("selected_uids_count")) or 0),
        "coverage_score_stats": _parse_json_dict(overall_payload.get("coverage_score_stats")),
        "near_zero_delta_rate": float(_to_float(overall_payload.get("near_zero_delta_rate")) or 0.0),
        "effect_size_nonzero_rate": float(_to_float(overall_payload.get("effect_size_nonzero_rate")) or 0.0),
        "significance_available_rate": float(_to_float(overall_payload.get("significance_available_rate")) or 0.0),
        "provider_noise_summary": {
            "availability": overall_provider_summary.get("availability", "unavailable"),
            **provider_fields,
        },
        "diagnosis_recommendations": overall_recommendations,
        "health_gate": health_snapshot_file.get("gate", {}),
        "detail_significance_rows": int(len(detail_sig_df)),
    }
    return out_df, snapshot


def write_result_diagnosis_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
    near_zero_threshold: float = 1e-6,
    effect_size_threshold: float = 1e-9,
    significance_threshold: float = 0.50,
    provider_telemetry_dir: str | Path | None = None,
    figure_formats: list[str] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    lib = _require_pandas()
    formats = figure_formats or ["png", "pdf"]
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_df, snapshot = build_result_diagnosis_table(
        suite_dir=suite_dir,
        near_zero_threshold=near_zero_threshold,
        effect_size_threshold=effect_size_threshold,
        significance_threshold=significance_threshold,
        provider_telemetry_dir=provider_telemetry_dir,
    )

    table_csv = tables_dir / "table_result_diagnosis.csv"
    table_md = tables_dir / "table_result_diagnosis.md"
    report_md = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv(table_csv, out_df)
    _write_text(table_md, "# Result Diagnosis\n\n" + df_to_markdown_table(out_df))

    if out_df.empty:
        figure_paths = _placeholder_figure(
            figures_dir / "fig_result_diagnosis_breakdown",
            "Result Diagnosis Breakdown",
            "No diagnosis rows available",
            formats,
        )
    else:
        plot_df = out_df.loc[out_df["task"].astype(str) != "overall"].copy()
        if plot_df.empty:
            plot_df = out_df.copy()
        plot_df["near_zero_delta_rate"] = lib.to_numeric(plot_df["near_zero_delta_rate"], errors="coerce").fillna(0.0)
        plot_df["missing_metric_rate"] = lib.to_numeric(plot_df["missing_metric_rate"], errors="coerce").fillna(0.0)
        plot_df["significance_available_rate"] = lib.to_numeric(plot_df["significance_available_rate"], errors="coerce").fillna(0.0)
        labels = plot_df["task"].astype(str).tolist()
        x = list(range(len(plot_df)))
        width = 0.25
        fig_base = figures_dir / "fig_result_diagnosis_breakdown"
        plt.figure(figsize=(9.0, 4.8))
        plt.bar([item - width for item in x], plot_df["near_zero_delta_rate"], width=width, label="near_zero_delta_rate")
        plt.bar(x, plot_df["missing_metric_rate"], width=width, label="missing_metric_rate")
        plt.bar([item + width for item in x], 1.0 - plot_df["significance_available_rate"], width=width, label="1-significance_available_rate")
        plt.xticks(x, labels)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Rate")
        plt.title("Result Diagnosis Breakdown")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        figure_paths = []
        for ext in formats:
            target = fig_base.with_suffix(f".{ext}")
            plt.savefig(target)
            figure_paths.append(str(target))
        plt.close()

    report_lines = [
        "# Result Diagnosis",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- rows_total: `{snapshot.get('rows_total', 0)}`",
        f"- selected_uids_count: `{snapshot.get('selected_uids_count', 0)}`",
        f"- significance_available_rate: `{snapshot.get('significance_available_rate', 0.0)}`",
        f"- effect_size_nonzero_rate: `{snapshot.get('effect_size_nonzero_rate', 0.0)}`",
        f"- near_zero_delta_rate: `{snapshot.get('near_zero_delta_rate', 0.0)}`",
        f"- provider_noise_summary: `{json.dumps(snapshot.get('provider_noise_summary', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- diagnosis_recommendations: `{snapshot.get('diagnosis_recommendations', [])}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- figures: `{figure_paths}`",
        f"- snapshot_json: `{snapshot_path}`",
    ]
    _write_text(report_md, "\n".join(report_lines))

    snapshot["outputs"] = {
        "table_csv": str(table_csv),
        "table_md": str(table_md),
        "figures": figure_paths,
        "report_md": str(report_md),
        "snapshot_json": str(snapshot_path),
    }
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "figure_paths": figure_paths,
        "report_md": report_md,
        "snapshot_json": snapshot_path,
        "rows_total": int(len(out_df)),
        "overall_no_data_reason_counts": snapshot.get("overall_no_data_reason_counts", {}),
        "provider_noise_summary": snapshot.get("provider_noise_summary", {}),
        "diagnosis_recommendations": snapshot.get("diagnosis_recommendations", []),
    }
