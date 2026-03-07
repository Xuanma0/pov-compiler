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


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for admission calibration reporting.")
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


def _to_int(value: Any) -> int:
    out = _to_float(value)
    return int(round(out)) if out is not None else 0


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
    coverage = payload.get("coverage_score_stats", {})
    if not isinstance(coverage, dict):
        coverage = _parse_json_dict(coverage)
    return float(_to_float(coverage.get("mean")) or 0.0)


def _query_group_count(manifest_payload: dict[str, Any], suite_root: Path) -> int:
    groups = manifest_payload.get("queries", {}).get("groups", [])
    if isinstance(groups, list) and groups:
        return len([str(item).strip() for item in groups if str(item).strip()])
    lock_payload = _read_json(suite_root / "manifest" / "query_bank_lock.json")
    primary = lock_payload.get("primary", {})
    if isinstance(primary, dict):
        lock_groups = primary.get("groups", [])
        if isinstance(lock_groups, list):
            return len([str(item).strip() for item in lock_groups if str(item).strip()])
    return 0


def _no_data_rate(health_snapshot: dict[str, Any]) -> float:
    gate_metrics = health_snapshot.get("gate", {}).get("metrics", {})
    if isinstance(gate_metrics, dict):
        gate_value = _to_float(gate_metrics.get("no_data_rate"))
        if gate_value is not None:
            return float(gate_value)
    counts = health_snapshot.get("overall_no_data_reason_counts", {})
    if not isinstance(counts, dict):
        return 1.0
    total = 0
    ok_count = 0
    for key, raw in counts.items():
        value = _to_int(raw)
        total += value
        if str(key) == "ok":
            ok_count += value
    if total <= 0:
        return 1.0
    return float(max(total - ok_count, 0) / total)


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_admission_calibration_table(*, suite_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    health_snapshot = _read_json(suite_root / "result_health" / "snapshot.json")
    diagnosis_snapshot = _read_json(suite_root / "result_diagnosis" / "snapshot.json")
    delta_audit_snapshot = _read_json(suite_root / "delta_audit" / "snapshot.json")
    provider_summary = _read_json(suite_root / "provider_telemetry" / "summary.json")
    compare_summary = _read_json(suite_root / "compare" / "compare_summary.json")
    admission_snapshot = _read_json(suite_root / "admission_control" / "snapshot.json")

    gate_metrics = health_snapshot.get("gate", {}).get("metrics", {})
    if not isinstance(gate_metrics, dict):
        gate_metrics = {}

    profile = str(admission_snapshot.get("profile") or manifest_payload.get("admission_profile") or "").strip()
    thresholds = admission_snapshot.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}

    selected_uids_count = int(
        _to_float(gate_metrics.get("selected_uids_count"))
        or _to_float(diagnosis_snapshot.get("selected_uids_count"))
        or 0
    )
    coverage_score_mean = float(
        _to_float(gate_metrics.get("coverage_score_mean"))
        or _coverage_mean(diagnosis_snapshot)
        or _coverage_mean(health_snapshot.get("selection", {}))
    )
    no_data_rate = float(_no_data_rate(health_snapshot))
    significance_available_rate = float(
        _to_float(gate_metrics.get("significance_available_rate"))
        or _to_float(diagnosis_snapshot.get("significance_available_rate"))
        or 0.0
    )
    effect_size_nonzero_rate = float(
        _to_float(gate_metrics.get("effect_size_nonzero_rate"))
        or _to_float(diagnosis_snapshot.get("effect_size_nonzero_rate"))
        or 0.0
    )
    parse_fail_rate = float(_to_float(provider_summary.get("structured_parse_fail_rate_mean")) or 0.0)
    fallback_rate = float(_to_float(provider_summary.get("planner_fallback_rate")) or 0.0)
    usage_present_rate = float(_to_float(provider_summary.get("usage_present_rate")) or 0.0)
    query_groups = _query_group_count(manifest_payload, suite_root)
    detail_rows = int(_to_float(diagnosis_snapshot.get("detail_significance_rows")) or 0)
    available_runs = int(_to_float(compare_summary.get("results_rows")) or 0)

    current_min_selected_uids = int(_to_float(thresholds.get("min_selected_uids")) or 0)
    current_max_no_data_rate = float(_to_float(thresholds.get("max_no_data_rate")) or 0.25)
    current_min_effect_rate = float(_to_float(thresholds.get("min_effect_size_nonzero_rate")) or 0.34)
    current_max_parse_fail = float(_to_float(thresholds.get("max_provider_noise_parse_fail_rate")) or 0.10)
    current_max_fallback = float(_to_float(thresholds.get("max_provider_noise_fallback_rate")) or 0.10)

    if selected_uids_count <= 0 or available_runs <= 0 or query_groups <= 0:
        calibration_status = "weak"
    elif selected_uids_count < max(current_min_selected_uids, 2) or detail_rows < 2:
        calibration_status = "partial"
    elif available_runs < 6 or significance_available_rate < 0.50:
        calibration_status = "partial"
    else:
        calibration_status = "ok"

    calibration_confidence = {
        "ok": "high",
        "partial": "medium",
        "weak": "low",
    }.get(calibration_status, "low")

    if calibration_status == "weak":
        recommended_min_selected_uids = max(current_min_selected_uids, max(selected_uids_count + 1, 3))
    elif calibration_status == "partial":
        recommended_min_selected_uids = max(current_min_selected_uids, selected_uids_count + 1)
    else:
        recommended_min_selected_uids = max(current_min_selected_uids, selected_uids_count)

    recommended_max_no_data_rate = _bounded(
        max(no_data_rate + 0.05, current_max_no_data_rate if calibration_status == "weak" else no_data_rate + 0.02),
        0.05,
        0.60,
    )
    recommended_min_effect_size_nonzero_rate = _bounded(
        max(current_min_effect_rate if calibration_status == "weak" else effect_size_nonzero_rate * 0.8, 0.10),
        0.10,
        1.0,
    )
    recommended_max_parse_fail_rate = _bounded(
        max(parse_fail_rate * 1.25, current_max_parse_fail if calibration_status == "weak" else parse_fail_rate + 0.02),
        0.02,
        0.25,
    )
    recommended_max_fallback_rate = _bounded(
        max(fallback_rate * 1.25, current_max_fallback if calibration_status == "weak" else fallback_rate + 0.02),
        0.02,
        0.25,
    )

    basis = {
        "available_runs": available_runs,
        "detail_significance_rows": detail_rows,
        "query_groups": query_groups,
        "selected_uids_count": selected_uids_count,
        "coverage_score_mean": coverage_score_mean,
        "significance_available_rate": significance_available_rate,
        "effect_size_nonzero_rate": effect_size_nonzero_rate,
        "provider_availability": str(provider_summary.get("availability", "unavailable")),
        "usage_present_rate": usage_present_rate,
    }
    row = {
        "admission_profile": profile,
        "calibration_status": calibration_status,
        "calibration_confidence": calibration_confidence,
        "recommended_min_selected_uids": int(recommended_min_selected_uids),
        "recommended_max_no_data_rate": float(recommended_max_no_data_rate),
        "recommended_min_effect_size_nonzero_rate": float(recommended_min_effect_size_nonzero_rate),
        "recommended_max_parse_fail_rate": float(recommended_max_parse_fail_rate),
        "recommended_max_fallback_rate": float(recommended_max_fallback_rate),
        "calibration_basis": json.dumps(basis, ensure_ascii=False, sort_keys=True),
        "current_min_selected_uids": int(current_min_selected_uids),
        "current_max_no_data_rate": float(current_max_no_data_rate),
        "current_min_effect_size_nonzero_rate": float(current_min_effect_rate),
        "current_max_parse_fail_rate": float(current_max_parse_fail),
        "current_max_fallback_rate": float(current_max_fallback),
        "selected_uids_count": int(selected_uids_count),
        "coverage_score_mean": float(coverage_score_mean),
        "no_data_rate": float(no_data_rate),
        "significance_available_rate": float(significance_available_rate),
        "effect_size_nonzero_rate": float(effect_size_nonzero_rate),
        "parse_fail_rate": float(parse_fail_rate),
        "fallback_rate": float(fallback_rate),
        "usage_present_rate": float(usage_present_rate),
    }
    out_df = lib.DataFrame([row])
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "profile": profile,
        "calibration_status": calibration_status,
        "calibration_confidence": calibration_confidence,
        "calibration_basis": basis,
        "recommended_profile": {
            "min_selected_uids": int(recommended_min_selected_uids),
            "max_no_data_rate": float(recommended_max_no_data_rate),
            "min_effect_size_nonzero_rate": float(recommended_min_effect_size_nonzero_rate),
            "max_parse_fail_rate": float(recommended_max_parse_fail_rate),
            "max_fallback_rate": float(recommended_max_fallback_rate),
        },
        "current_profile": {
            "min_selected_uids": int(current_min_selected_uids),
            "max_no_data_rate": float(current_max_no_data_rate),
            "min_effect_size_nonzero_rate": float(current_min_effect_rate),
            "max_parse_fail_rate": float(current_max_parse_fail),
            "max_fallback_rate": float(current_max_fallback),
        },
        "sources": {
            "result_health_snapshot": str(suite_root / "result_health" / "snapshot.json"),
            "result_diagnosis_snapshot": str(suite_root / "result_diagnosis" / "snapshot.json"),
            "delta_audit_snapshot": str(suite_root / "delta_audit" / "snapshot.json"),
            "provider_telemetry_summary": str(suite_root / "provider_telemetry" / "summary.json"),
        },
        "delta_audit_summary": {
            "main_recommendation": delta_audit_snapshot.get("main_recommendation", ""),
            "recommended_action_counts": delta_audit_snapshot.get("recommended_action_counts", {}),
        },
        "provider_noise_summary": provider_summary,
    }
    return out_df, snapshot


def write_admission_calibration_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    out_df, snapshot = build_admission_calibration_table(suite_dir=suite_dir)
    table_csv = tables_dir / "table_admission_calibration.csv"
    table_md = tables_dir / "table_admission_calibration.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Admission Calibration\n\n" + df_to_markdown_table(out_df))
    report_lines = [
        "# Admission Calibration",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- profile: `{snapshot.get('profile', '')}`",
        f"- calibration_status: `{snapshot.get('calibration_status', 'weak')}`",
        f"- calibration_confidence: `{snapshot.get('calibration_confidence', 'low')}`",
        f"- calibration_basis: `{json.dumps(snapshot.get('calibration_basis', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- recommended_profile: `{json.dumps(snapshot.get('recommended_profile', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- current_profile: `{json.dumps(snapshot.get('current_profile', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- report_md: `{report_md}`",
        f"- snapshot_json: `{snapshot_json}`",
    ]
    _write_text(report_md, "\n".join(report_lines))
    snapshot["outputs"] = {
        "table_csv": str(table_csv),
        "table_md": str(table_md),
        "report_md": str(report_md),
        "snapshot_json": str(snapshot_json),
    }
    _write_text(snapshot_json, json.dumps(snapshot, ensure_ascii=False, indent=2))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "report_md": report_md,
        "snapshot_json": snapshot_json,
        "rows_total": int(len(out_df)),
        "calibration_status": snapshot.get("calibration_status", "weak"),
        "calibration_summary": {
            "calibration_confidence": snapshot.get("calibration_confidence", "low"),
            "recommended_profile": snapshot.get("recommended_profile", {}),
        },
    }
