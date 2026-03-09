from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

from pov_compiler.bench.reporting.latex import df_to_markdown_table
from pov_compiler.bench.reporting.provider_telemetry import load_provider_telemetry_outputs


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for delta-audit reporting.")
    return pd


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


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


def _coverage_mean(payload: dict[str, Any]) -> float:
    coverage_stats = payload.get("coverage_score_stats", {})
    if not isinstance(coverage_stats, dict):
        coverage_stats = _parse_json_dict(coverage_stats)
    return float(_to_float(coverage_stats.get("mean")) or 0.0)


def _build_health_map(health_df: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if health_df is None or len(health_df) == 0:
        return out
    for _, row in health_df.iterrows():
        out[str(row.get("task", "")).strip() or "overall"] = row.to_dict()
    return out


def _build_significance_map(detail_sig_df: Any) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if detail_sig_df is None or len(detail_sig_df) == 0:
        return out
    for _, row in detail_sig_df.iterrows():
        key = (
            str(row.get("task", "")).strip(),
            str(row.get("budget_key", "")).strip(),
            str(row.get("primary_metric", "")).strip(),
        )
        if key not in out:
            out[key] = row.to_dict()
    return out


def _provider_noise_flag(provider_summary: dict[str, Any], manifest_payload: dict[str, Any]) -> bool:
    parse_fail = _to_float(provider_summary.get("structured_parse_fail_rate_mean"))
    fallback = _to_float(provider_summary.get("planner_fallback_rate"))
    availability = str(provider_summary.get("availability", "unavailable")).strip()
    parse_fail_max = float(_to_float(manifest_payload.get("max_provider_noise_parse_fail_rate")) or 0.10)
    fallback_max = float(_to_float(manifest_payload.get("max_provider_noise_fallback_rate")) or 0.10)
    return bool(
        availability in {"provider_unavailable", "no_real_call", "partial", "unavailable"}
        or (parse_fail is not None and parse_fail > parse_fail_max)
        or (fallback is not None and fallback > fallback_max)
    )


def _recommended_action(
    *,
    task: str,
    metric: str,
    no_data_reason: str,
    near_zero: bool,
    significance_available: bool,
    effect_size_nonzero: bool,
    provider_noise_flag: bool,
    coverage_score_mean: float,
    selected_uids_count: int,
    provider_summary: dict[str, Any],
) -> str:
    parse_fail = _to_float(provider_summary.get("structured_parse_fail_rate_mean"))
    fallback = _to_float(provider_summary.get("planner_fallback_rate"))
    if no_data_reason in {"source_missing", "source_empty", "missing_rows"}:
        return "increase_signal_coverage"
    if no_data_reason == "insufficient_pairs" or (not significance_available and selected_uids_count < 3):
        return "increase_selected_uids"
    if provider_noise_flag and fallback is not None and fallback > 0.10:
        return "tune_planner_backend"
    if "repo" in task.lower() or "repo" in metric.lower():
        return "tune_repo_policy"
    if provider_noise_flag and parse_fail is not None and parse_fail > 0.10:
        return "reduce_provider_noise"
    if coverage_score_mean < 2.0:
        return "increase_signal_coverage"
    if near_zero and not significance_available:
        return "increase_selected_uids"
    if near_zero and provider_noise_flag:
        return "reduce_provider_noise"
    if near_zero and not effect_size_nonzero:
        return "strengthen_query_bank"
    return "algorithm_no_effect_detected"


def build_delta_audit_table(
    *,
    suite_dir: str | Path,
    near_zero_threshold: float = 1e-6,
    effect_size_threshold: float = 1e-9,
    provider_telemetry_dir: str | Path | None = None,
    admission_profile: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    compare_dir = suite_root / "compare"
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    compare_summary = _read_json(compare_dir / "compare_summary.json")
    main_df = _read_csv(compare_dir / "tables" / "table_main_results.csv")
    if main_df.empty:
        main_df = lib.DataFrame(
            [
                {
                    "task": "overall",
                    "budget_key": "n/a",
                    "budget_seconds": 0.0,
                    "primary_metric": "",
                    "delta": 0.0,
                    "status": "source_missing",
                }
            ]
        )
    health_df = _read_csv(suite_root / "result_health" / "tables" / "table_result_health.csv")
    detail_sig_df = _read_csv(suite_root / "significance" / "tables" / "table_significance_main.csv")
    health_snapshot = _read_json(suite_root / "result_health" / "snapshot.json")
    diagnosis_snapshot = _read_json(suite_root / "result_diagnosis" / "snapshot.json")
    resolved_provider_dir = Path(provider_telemetry_dir).resolve() if provider_telemetry_dir else (suite_root / "provider_telemetry")
    _, provider_summary = load_provider_telemetry_outputs(resolved_provider_dir)

    health_map = _build_health_map(health_df)
    sig_map = _build_significance_map(detail_sig_df)
    label_a = str(compare_summary.get("label_a", manifest_payload.get("selection", {}).get("labels", {}).get("a", "a"))).strip() or "a"
    label_b = str(compare_summary.get("label_b", manifest_payload.get("selection", {}).get("labels", {}).get("b", "b"))).strip() or "b"
    provider_noise_flag = _provider_noise_flag(provider_summary, manifest_payload)

    rows: list[dict[str, Any]] = []
    for _, row in main_df.iterrows():
        row_payload = row.to_dict()
        task = str(row_payload.get("task", "")).strip() or "overall"
        budget_key = str(row_payload.get("budget_key", "")).strip() or "n/a"
        metric = str(row_payload.get("primary_metric", "")).strip()
        delta_value = float(_to_float(row_payload.get("delta")) or 0.0)
        row_status = str(row_payload.get("status", "")).strip()
        health_row = health_map.get(task) or health_map.get("overall", {})
        sig_payload = sig_map.get((task, budget_key, metric), {})
        sig_status = str(sig_payload.get("status", "")).strip()
        significance_available = sig_status not in {"", "task_missing", "insufficient_pairs"}
        effect_size_value = _to_float(sig_payload.get("mean_delta"))
        if effect_size_value is None:
            effect_size_value = delta_value
        effect_size_nonzero = abs(float(effect_size_value or 0.0)) > float(effect_size_threshold)
        if row_status == "missing_rows":
            no_data_reason = "missing_rows"
        elif row_status == "metric_missing":
            no_data_reason = "missing_metric"
        elif sig_status == "insufficient_pairs":
            no_data_reason = "insufficient_pairs"
        elif str(health_row.get("no_data_reason", "")).strip() in {"source_missing", "source_empty"}:
            no_data_reason = str(health_row.get("no_data_reason", "")).strip()
        else:
            no_data_reason = "ok"
        near_zero = abs(delta_value) < float(near_zero_threshold)
        selected_uids_count = int(
            _to_float(health_row.get("selected_uids_count")) or diagnosis_snapshot.get("selected_uids_count") or 0
        )
        coverage_score_mean = _coverage_mean(health_row) or _coverage_mean(diagnosis_snapshot)
        recommended_action = _recommended_action(
            task=task,
            metric=metric,
            no_data_reason=no_data_reason,
            near_zero=near_zero,
            significance_available=significance_available,
            effect_size_nonzero=effect_size_nonzero,
            provider_noise_flag=provider_noise_flag,
            coverage_score_mean=coverage_score_mean,
            selected_uids_count=selected_uids_count,
            provider_summary=provider_summary,
        )
        rows.append(
            {
                "variant": f"{label_a}->{label_b}",
                "baseline_label": label_a,
                "treatment_label": label_b,
                "task": task,
                "metric": metric,
                "budget": budget_key,
                "budget_seconds": float(_to_float(row_payload.get("budget_seconds")) or 0.0),
                "delta_value": delta_value,
                "near_zero": bool(near_zero),
                "significance_available": bool(significance_available),
                "effect_size_nonzero": bool(effect_size_nonzero),
                "no_data_reason": no_data_reason,
                "provider_noise_flag": bool(provider_noise_flag),
                "selected_uids_count": selected_uids_count,
                "coverage_score_mean": coverage_score_mean,
                "recommended_action": recommended_action,
            }
        )

    out_df = lib.DataFrame(rows)
    if out_df.empty:
        out_df = lib.DataFrame(
            [
                {
                    "variant": f"{label_a}->{label_b}",
                    "baseline_label": label_a,
                    "treatment_label": label_b,
                    "task": "overall",
                    "metric": "",
                    "budget": "n/a",
                    "budget_seconds": 0.0,
                    "delta_value": 0.0,
                    "near_zero": True,
                    "significance_available": False,
                    "effect_size_nonzero": False,
                    "no_data_reason": "source_missing",
                    "provider_noise_flag": provider_noise_flag,
                    "selected_uids_count": 0,
                    "coverage_score_mean": 0.0,
                    "recommended_action": "increase_signal_coverage",
                }
            ]
        )

    recommended_counter = Counter(str(value) for value in out_df["recommended_action"].tolist())
    task_counter = Counter(str(value) for value in out_df["task"].tolist())
    main_recommendation = ""
    if recommended_counter:
        main_recommendation = sorted(
            recommended_counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[0][0]
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "provider_telemetry_dir": str(resolved_provider_dir) if resolved_provider_dir.exists() else None,
        "admission_profile": str(admission_profile or manifest_payload.get("admission_profile", "")).strip(),
        "thresholds": {
            "near_zero_threshold": float(near_zero_threshold),
            "effect_size_threshold": float(effect_size_threshold),
        },
        "rows_total": int(len(out_df)),
        "near_zero_rate": float(out_df["near_zero"].astype(bool).mean()) if len(out_df) > 0 else 1.0,
        "provider_noise_flag_rate": float(out_df["provider_noise_flag"].astype(bool).mean()) if len(out_df) > 0 else 0.0,
        "recommended_action_counts": {key: int(value) for key, value in sorted(recommended_counter.items())},
        "rows_by_task": {key: int(value) for key, value in sorted(task_counter.items())},
        "main_recommendation": main_recommendation,
        "provider_noise_summary": provider_summary,
        "health_gate": health_snapshot.get("gate", {}),
        "diagnosis_recommendations": diagnosis_snapshot.get("diagnosis_recommendations", []),
    }
    return out_df, snapshot


def write_delta_audit_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
    near_zero_threshold: float = 1e-6,
    effect_size_threshold: float = 1e-9,
    provider_telemetry_dir: str | Path | None = None,
    admission_profile: str | None = None,
    figure_formats: list[str] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    formats = figure_formats or ["png", "pdf"]
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_df, snapshot = build_delta_audit_table(
        suite_dir=suite_dir,
        near_zero_threshold=near_zero_threshold,
        effect_size_threshold=effect_size_threshold,
        provider_telemetry_dir=provider_telemetry_dir,
        admission_profile=admission_profile,
    )
    table_csv = tables_dir / "table_delta_audit.csv"
    table_md = tables_dir / "table_delta_audit.md"
    report_md = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Delta Audit\n\n" + df_to_markdown_table(out_df))

    fig_base = figures_dir / "fig_delta_audit_breakdown"
    action_counter = snapshot.get("recommended_action_counts", {})
    figure_paths: list[str] = []
    if action_counter:
        ordered = [(key, int(action_counter[key])) for key in sorted(action_counter.keys())]
        labels = [item[0] for item in ordered]
        values = [item[1] for item in ordered]
        plt.figure(figsize=(9.0, 4.8))
        plt.bar(labels, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Rows")
        plt.title("Delta Audit Breakdown")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        for ext in formats:
            target = fig_base.with_suffix(f".{ext}")
            plt.savefig(target)
            figure_paths.append(str(target))
        plt.close()
    else:
        plt.figure(figsize=(8.2, 4.6))
        plt.text(0.5, 0.5, "No delta-audit rows available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        for ext in formats:
            target = fig_base.with_suffix(f".{ext}")
            plt.savefig(target)
            figure_paths.append(str(target))
        plt.close()

    report_lines = [
        "# Delta Audit",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- provider_telemetry_dir: `{snapshot.get('provider_telemetry_dir')}`",
        f"- admission_profile: `{snapshot.get('admission_profile')}`",
        f"- rows_total: `{snapshot.get('rows_total', 0)}`",
        f"- near_zero_rate: `{snapshot.get('near_zero_rate', 0.0)}`",
        f"- provider_noise_flag_rate: `{snapshot.get('provider_noise_flag_rate', 0.0)}`",
        f"- main_recommendation: `{snapshot.get('main_recommendation', '')}`",
        f"- recommended_action_counts: `{snapshot.get('recommended_action_counts', {})}`",
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
        "report_md": report_md,
        "snapshot_json": snapshot_path,
        "figure_paths": figure_paths,
        "rows_total": int(len(out_df)),
        "delta_audit_summary": {
            "main_recommendation": snapshot.get("main_recommendation", ""),
            "recommended_action_counts": snapshot.get("recommended_action_counts", {}),
        },
    }
