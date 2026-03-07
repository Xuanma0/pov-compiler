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
from pov_compiler.bench.reporting.provider_telemetry import load_provider_telemetry_outputs


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for provider normalization reporting.")
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


def _to_int(value: Any) -> int | None:
    out = _to_float(value)
    if out is None:
        return None
    return int(round(out))


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "ok"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [float(item) for item in values if item is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _safe_sum(values: list[int | None]) -> int:
    clean = [int(item) for item in values if item is not None]
    return int(sum(clean)) if clean else 0


def _pick_value(row: dict[str, Any], aliases: list[str]) -> Any:
    for alias in aliases:
        if alias in row and row.get(alias) not in (None, ""):
            return row.get(alias)
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_normalization_config(config_path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    default_path = _repo_root() / "configs" / "telemetry" / "provider_normalization_v1.yaml"
    resolved = Path(config_path).resolve() if config_path else default_path.resolve()
    payload = _load_yaml(resolved)
    return resolved, payload


def _real_call_status(provider: str, availability: str, calls_total: int) -> str:
    provider_text = str(provider or "").strip().lower()
    if provider_text.startswith("fake") or provider_text == "fake":
        return "simulated"
    if availability in {"provider_unavailable", "no_real_call", "unavailable"} or calls_total <= 0:
        return "missing_or_unavailable"
    return "observed"


def _missing_fields(
    *,
    row: dict[str, Any],
    config: dict[str, Any],
    usage_present: bool,
    cost_known: bool,
) -> list[str]:
    missing: list[str] = []
    required_fields = list(config.get("required_fields", [])) if isinstance(config.get("required_fields"), list) else []
    conditional_required = config.get("conditional_required", {})
    if not isinstance(conditional_required, dict):
        conditional_required = {}
    for field in required_fields:
        value = row.get(field)
        if value in (None, ""):
            missing.append(str(field))
    usage_required = conditional_required.get("usage_present", {})
    if isinstance(usage_required, dict) and usage_present:
        for field in usage_required.get("when_true", []) if isinstance(usage_required.get("when_true"), list) else []:
            if row.get(str(field)) in (None, ""):
                missing.append(str(field))
    cost_required = conditional_required.get("cost_known", {})
    if isinstance(cost_required, dict) and cost_known:
        for field in cost_required.get("when_true", []) if isinstance(cost_required.get("when_true"), list) else []:
            if row.get(str(field)) in (None, ""):
                missing.append(str(field))
    return sorted(dict.fromkeys(missing))


def _normalize_row(
    *,
    row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    targets = config.get("targets", {})
    if not isinstance(targets, dict):
        targets = {}
    normalized: dict[str, Any] = {
        "variant_label": str(row.get("variant_label", "")).strip(),
        "variant_code": str(row.get("variant_code", "")).strip(),
    }
    for target, aliases in targets.items():
        alias_list = [str(item) for item in aliases] if isinstance(aliases, list) else [str(aliases)]
        raw_value = _pick_value(row, alias_list)
        if target in {"usage_present", "cost_known"}:
            normalized[target] = _to_bool(raw_value)
        elif target in {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "calls_total",
            "calls_with_usage",
            "calls_with_cost",
        }:
            normalized[target] = _to_int(raw_value)
        elif target in {
            "model_cost_usd_total",
            "model_cost_usd_mean_per_query",
            "latency_p50_ms",
            "latency_p95_ms",
            "structured_parse_fail_rate",
            "planner_fallback_rate",
        }:
            normalized[target] = _to_float(raw_value)
        else:
            normalized[target] = str(raw_value).strip() if raw_value is not None else ""

    usage_present = bool(normalized.get("usage_present"))
    cost_known = bool(normalized.get("cost_known"))
    availability = str(row.get("availability", "")).strip() or "unavailable"
    calls_total = int(normalized.get("calls_total") or 0)
    missing_fields = _missing_fields(
        row=normalized,
        config=config,
        usage_present=usage_present,
        cost_known=cost_known,
    )
    real_call_status = _real_call_status(
        provider=str(normalized.get("provider", "")),
        availability=availability,
        calls_total=calls_total,
    )
    if not str(normalized.get("provider", "")).strip() and not str(normalized.get("model", "")).strip():
        normalization_status = "unavailable"
    elif missing_fields or real_call_status == "missing_or_unavailable" or availability == "partial":
        normalization_status = "partial"
    else:
        normalization_status = "ok"
    normalized["availability"] = availability
    normalized["availability_reason"] = str(row.get("availability_reason", "")).strip()
    normalized["real_call_status"] = real_call_status
    normalized["normalization_status"] = normalization_status
    normalized["missing_fields"] = json.dumps(missing_fields, ensure_ascii=False)
    normalized["telemetry_source_paths"] = str(row.get("telemetry_source_paths", "[]"))
    return normalized


def _overall_status(rows: list[dict[str, Any]]) -> str:
    statuses = [str(row.get("normalization_status", "")).strip() for row in rows if str(row.get("normalization_status", "")).strip()]
    if not statuses:
        return "unavailable"
    if all(item == "ok" for item in statuses):
        return "ok"
    if any(item == "partial" for item in statuses):
        return "partial"
    return statuses[0]


def _overall_real_call_status(rows: list[dict[str, Any]]) -> str:
    values = [str(row.get("real_call_status", "")).strip() for row in rows if str(row.get("real_call_status", "")).strip()]
    real_values = [value for value in values if value != "simulated"]
    if any(value == "observed" for value in real_values):
        return "observed"
    if any(value == "missing_or_unavailable" for value in real_values):
        return "missing_or_unavailable"
    if values:
        return values[0]
    return "missing_or_unavailable"


def build_provider_normalization_table(
    *,
    suite_dir: str | Path,
    provider_telemetry_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    resolved_provider_dir = Path(provider_telemetry_dir).resolve() if provider_telemetry_dir else (suite_root / "provider_telemetry")
    config_file, config = _load_normalization_config(config_path)
    telemetry_df, telemetry_summary = load_provider_telemetry_outputs(resolved_provider_dir)
    compare_summary = _read_json(suite_root / "compare" / "compare_summary.json")

    rows: list[dict[str, Any]] = []
    if len(telemetry_df) > 0:
        for _, payload in telemetry_df.sort_values(["variant_label"]).iterrows():
            rows.append(_normalize_row(row=payload.to_dict(), config=config))
    elif telemetry_summary:
        rows.append(
            _normalize_row(
                row={
                    "variant_label": "overall",
                    "provider": telemetry_summary.get("provider", ""),
                    "model": telemetry_summary.get("model", ""),
                    "api_mode_used": telemetry_summary.get("api_mode_used", ""),
                    "usage_present": telemetry_summary.get("usage_present_rate"),
                    "cost_known": telemetry_summary.get("model_cost_known_rate"),
                    "latency_p95_ms": telemetry_summary.get("model_latency_p95_ms_mean"),
                    "structured_parse_fail_rate": telemetry_summary.get("structured_parse_fail_rate_mean"),
                    "planner_fallback_rate": telemetry_summary.get("planner_fallback_rate"),
                    "calls_total": telemetry_summary.get("calls_total"),
                    "calls_with_usage": telemetry_summary.get("calls_with_usage"),
                    "calls_with_cost": telemetry_summary.get("calls_with_cost"),
                    "availability": telemetry_summary.get("availability", "unavailable"),
                    "telemetry_source_paths": json.dumps(telemetry_summary.get("telemetry_source_paths", []), ensure_ascii=False),
                },
                config=config,
            )
        )

    out_df = lib.DataFrame(rows)
    missing_fields_union = sorted(
        {
            item
            for row in rows
            for item in json.loads(str(row.get("missing_fields", "[]")) or "[]")
            if str(item).strip()
        }
    )
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "provider_telemetry_dir": str(resolved_provider_dir) if resolved_provider_dir.exists() else None,
        "normalization_config": str(config_file),
        "normalization_status": _overall_status(rows),
        "real_call_status": _overall_real_call_status(rows),
        "rows_total": int(len(out_df)),
        "providers_total": int(len({str(row.get('provider', '')).strip() for row in rows if str(row.get('provider', '')).strip()})),
        "usage_present_rate": _safe_mean([1.0 if row.get("usage_present") else 0.0 for row in rows]),
        "model_cost_known_rate": _safe_mean([1.0 if row.get("cost_known") else 0.0 for row in rows]),
        "model_latency_p95_ms_mean": _safe_mean([_to_float(row.get("latency_p95_ms")) for row in rows]),
        "structured_parse_fail_rate_mean": _safe_mean([_to_float(row.get("structured_parse_fail_rate")) for row in rows]),
        "planner_fallback_rate": _safe_mean([_to_float(row.get("planner_fallback_rate")) for row in rows]),
        "calls_total": _safe_sum([_to_int(row.get("calls_total")) for row in rows]),
        "calls_with_usage": _safe_sum([_to_int(row.get("calls_with_usage")) for row in rows]),
        "calls_with_cost": _safe_sum([_to_int(row.get("calls_with_cost")) for row in rows]),
        "missing_fields_union": missing_fields_union,
        "telemetry_summary_availability": str(telemetry_summary.get("availability", "unavailable")) if telemetry_summary else "unavailable",
        "telemetry_source_paths": telemetry_summary.get("telemetry_source_paths", []) if isinstance(telemetry_summary.get("telemetry_source_paths"), list) else [],
        "query_bank_id": str(compare_summary.get("query_bank_id", "")),
        "query_bank_hash": str(compare_summary.get("query_bank_hash", "")),
    }
    return out_df, summary


def write_provider_normalization_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
    provider_telemetry_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_df, summary = build_provider_normalization_table(
        suite_dir=suite_dir,
        provider_telemetry_dir=provider_telemetry_dir,
        config_path=config_path,
    )
    table_csv = tables_dir / "table_provider_normalization.csv"
    table_md = tables_dir / "table_provider_normalization.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    out_df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Provider Normalization\n\n" + df_to_markdown_table(out_df))
    report_lines = [
        "# Provider Normalization",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- provider_telemetry_dir: `{summary.get('provider_telemetry_dir')}`",
        f"- normalization_status: `{summary.get('normalization_status', 'unavailable')}`",
        f"- real_call_status: `{summary.get('real_call_status', 'missing_or_unavailable')}`",
        f"- usage_present_rate: `{summary.get('usage_present_rate')}`",
        f"- model_cost_known_rate: `{summary.get('model_cost_known_rate')}`",
        f"- model_latency_p95_ms_mean: `{summary.get('model_latency_p95_ms_mean')}`",
        f"- structured_parse_fail_rate_mean: `{summary.get('structured_parse_fail_rate_mean')}`",
        f"- planner_fallback_rate: `{summary.get('planner_fallback_rate')}`",
        f"- missing_fields_union: `{summary.get('missing_fields_union', [])}`",
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


def load_provider_normalization_outputs(normalization_dir: str | Path | None) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    if normalization_dir is None:
        return lib.DataFrame(), {}
    root = Path(normalization_dir).resolve()
    if not root.exists():
        return lib.DataFrame(), {}
    table_csv = root / "tables" / "table_provider_normalization.csv"
    try:
        df = lib.read_csv(table_csv) if table_csv.exists() else lib.DataFrame()
    except Exception:
        df = lib.DataFrame()
    summary = _read_json(root / "snapshot.json")
    return df, summary
