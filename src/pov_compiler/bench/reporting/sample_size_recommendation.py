from __future__ import annotations

import json
import math
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
        raise ImportError("pandas is required for sample-size recommendation reporting.")
    return pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _recommendation_status(
    *,
    runs_count: int,
    coefficient_of_variation: float,
    effect_size_nonzero_rate: float,
    repeatability_status: str,
) -> str:
    if runs_count < 3 or repeatability_status == "weak":
        return "weak"
    if coefficient_of_variation > 0.25 or effect_size_nonzero_rate < 0.50:
        return "range_only"
    return "ok"


def build_sample_size_recommendation_table(
    *,
    suite_dir: str | Path,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    repeat_snapshot = _read_json(suite_root / "repeatability_audit" / "snapshot.json")
    repeat_df = _read_csv(suite_root / "repeatability_audit" / "tables" / "table_repeatability_audit.csv")
    diagnosis_snapshot = _read_json(suite_root / "result_diagnosis" / "snapshot.json")
    calibration_snapshot = _read_json(suite_root / "admission_calibration" / "snapshot.json")
    query_strength_snapshot = _read_json(suite_root / "query_strength_audit" / "snapshot.json")
    repeat_profile = (
        repeat_snapshot.get("repeat_profile", {}).get("profile", {})
        if isinstance(repeat_snapshot.get("repeat_profile"), dict)
        else {}
    )
    growth_factor = float(_to_float(repeat_profile.get("sample_size_growth_factor")) or 2.0)
    range_growth_factor = float(_to_float(repeat_profile.get("sample_size_range_growth_factor")) or 3.0)

    selected_uids_count = int(_to_float(repeat_snapshot.get("selected_uids_count")) or diagnosis_snapshot.get("selected_uids_count") or 0)
    effect_size_nonzero_rate = float(_to_float(diagnosis_snapshot.get("effect_size_nonzero_rate")) or 0.0)
    repeatability_status = str(repeat_snapshot.get("repeatability_status", "weak")).strip() or "weak"
    detail_rows_total = int(_to_float(diagnosis_snapshot.get("detail_significance_rows")) or 0)
    promoted_groups = int(len(query_strength_snapshot.get("promoted_query_groups", []))) if isinstance(query_strength_snapshot.get("promoted_query_groups"), list) else 0

    rows: list[dict[str, Any]] = []
    if repeat_df.empty:
        repeat_df = lib.DataFrame(
            [
                {
                    "variant": "stub->real",
                    "metric": "overall:primary_metric",
                    "runs_count": int(repeat_snapshot.get("repeat_runs_total", 1) or 1),
                    "coefficient_of_variation": None,
                    "stability_flag": "weak_evidence",
                }
            ]
        )

    for metric_name, metric_df in repeat_df.groupby("metric", dropna=False):
        metric_rows = metric_df.copy()
        runs_count = int(_safe_max([float(_to_float(value) or 0.0) for value in metric_rows.get("runs_count", []).tolist()]))
        coefficient_of_variation = _safe_mean(
            [float(_to_float(value) or 0.0) for value in metric_rows.get("coefficient_of_variation", []).tolist()]
        )
        stability_flags = sorted({str(value).strip() for value in metric_rows.get("stability_flag", []).tolist() if str(value).strip()})
        current_n_pairs = max(int(detail_rows_total), max(selected_uids_count * max(runs_count, 1), 1))
        status = _recommendation_status(
            runs_count=runs_count,
            coefficient_of_variation=coefficient_of_variation,
            effect_size_nonzero_rate=effect_size_nonzero_rate,
            repeatability_status=repeatability_status,
        )
        if status == "weak":
            recommended_n_pairs_min = int(max(current_n_pairs * growth_factor, current_n_pairs + 4))
            recommended_n_uids_min = int(max(selected_uids_count + 2, math.ceil(recommended_n_pairs_min / max(runs_count, 1))))
            confidence_level = "low"
            notes = "too few repeated runs to calibrate precisely"
        elif status == "range_only":
            recommended_n_pairs_min = int(max(current_n_pairs * range_growth_factor, current_n_pairs + 6))
            recommended_n_uids_min = int(max(selected_uids_count + 2, math.ceil(recommended_n_pairs_min / max(runs_count, 1))))
            confidence_level = "medium"
            notes = "variance/effect mix only supports range guidance"
        else:
            variance_factor = max(1.25, 1.0 + coefficient_of_variation)
            effect_factor = max(1.1, 1.0 + max(0.0, 0.75 - effect_size_nonzero_rate))
            recommended_n_pairs_min = int(max(math.ceil(current_n_pairs * variance_factor * effect_factor), current_n_pairs + 2))
            recommended_n_uids_min = int(max(selected_uids_count, math.ceil(recommended_n_pairs_min / max(runs_count, 1))))
            confidence_level = "high"
            notes = "repeatability and effect-rate support a tighter recommendation"
        rows.append(
            {
                "metric": str(metric_name),
                "current_n_pairs": int(current_n_pairs),
                "recommended_n_pairs_min": int(recommended_n_pairs_min),
                "recommended_n_uids_min": int(recommended_n_uids_min),
                "confidence_level": confidence_level,
                "recommendation_status": status,
                "based_on_variance": float(coefficient_of_variation),
                "based_on_effect_size": float(effect_size_nonzero_rate),
                "notes": notes,
                "runs_count": int(runs_count),
                "stability_flags": json.dumps(stability_flags, ensure_ascii=False),
                "promoted_query_groups": int(promoted_groups),
            }
        )

    out_df = lib.DataFrame(rows)
    status_counts = (
        out_df["recommendation_status"].astype(str).value_counts().sort_index().to_dict()
        if not out_df.empty
        else {}
    )
    status_rank = {"weak": 0, "range_only": 1, "ok": 2}
    overall_status = "weak"
    if not out_df.empty:
        overall_status = sorted(
            {str(value) for value in out_df["recommendation_status"].astype(str).tolist()},
            key=lambda item: status_rank.get(item, -1),
        )[0]
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "repeatability_status": repeatability_status,
        "rows_total": int(len(out_df)),
        "sample_size_recommendation_status": overall_status,
        "recommendation_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "selected_uids_count": int(selected_uids_count),
        "effect_size_nonzero_rate": float(effect_size_nonzero_rate),
        "growth_factor": float(growth_factor),
        "range_growth_factor": float(range_growth_factor),
        "sources": {
            "repeatability_snapshot": str(suite_root / "repeatability_audit" / "snapshot.json"),
            "repeatability_table": str(suite_root / "repeatability_audit" / "tables" / "table_repeatability_audit.csv"),
            "result_diagnosis_snapshot": str(suite_root / "result_diagnosis" / "snapshot.json"),
            "admission_calibration_snapshot": str(suite_root / "admission_calibration" / "snapshot.json"),
            "query_strength_snapshot": str(suite_root / "query_strength_audit" / "snapshot.json"),
        },
        "calibration_status": str(calibration_snapshot.get("calibration_status", "unknown")),
    }
    return out_df, summary


def write_sample_size_recommendation_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    out_df, summary = build_sample_size_recommendation_table(suite_dir=suite_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_csv = tables_dir / "table_sample_size_recommendation.csv"
    table_md = tables_dir / "table_sample_size_recommendation.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    out_df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Sample Size Recommendation\n\n" + df_to_markdown_table(out_df))
    report_lines = [
        "# Sample Size Recommendation",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- repeatability_status: `{summary.get('repeatability_status', 'weak')}`",
        f"- sample_size_recommendation_status: `{summary.get('sample_size_recommendation_status', 'weak')}`",
        f"- recommendation_status_counts: `{summary.get('recommendation_status_counts', {})}`",
        f"- selected_uids_count: `{summary.get('selected_uids_count', 0)}`",
        f"- effect_size_nonzero_rate: `{summary.get('effect_size_nonzero_rate', 0.0)}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- report_md: `{report_md}`",
        f"- snapshot_json: `{snapshot_json}`",
    ]
    _write_text(report_md, "\n".join(report_lines))
    summary["outputs"] = {
        "table_csv": str(table_csv),
        "table_md": str(table_md),
        "report_md": str(report_md),
        "snapshot_json": str(snapshot_json),
    }
    _write_text(snapshot_json, json.dumps(summary, ensure_ascii=False, indent=2))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "report_md": report_md,
        "snapshot_json": snapshot_json,
        "rows_total": int(len(out_df)),
        "summary": summary,
    }
