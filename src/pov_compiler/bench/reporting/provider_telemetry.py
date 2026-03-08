from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for provider telemetry reporting.")
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


def _to_int(value: Any) -> int | None:
    out = _to_float(value)
    if out is None:
        return None
    return int(round(out))


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _safe_sum(values: list[int | float | None]) -> int:
    clean = [float(value) for value in values if value is not None]
    return int(round(sum(clean))) if clean else 0


def _safe_max(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(max(clean))


def _resolve_optional_path(raw_value: str | None, base_dir: Path) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (Path.cwd() / path).resolve()


def _truthy(value: Any) -> bool | None:
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


def _pick_text(frame: Any, columns: list[str]) -> str:
    if frame is None or len(frame) == 0:
        return ""
    for column in columns:
        if column not in frame.columns:
            continue
        values = [str(value).strip() for value in frame[column].tolist() if str(value).strip()]
        if values:
            return values[0]
    return ""


def _pick_bool(frame: Any, columns: list[str]) -> bool | None:
    if frame is None or len(frame) == 0:
        return None
    for column in columns:
        if column not in frame.columns:
            continue
        vals = [_truthy(value) for value in frame[column].tolist()]
        vals = [value for value in vals if value is not None]
        if vals:
            return bool(any(vals))
    return None


def _pick_number(frame: Any, columns: list[str], *, reducer: str = "mean") -> float | None:
    lib = _require_pandas()
    if frame is None or len(frame) == 0:
        return None
    for column in columns:
        if column not in frame.columns:
            continue
        values = lib.to_numeric(frame[column], errors="coerce").dropna().tolist()
        if not values:
            continue
        if reducer == "sum":
            return float(sum(values))
        if reducer == "max":
            return float(max(values))
        return float(sum(values) / len(values))
    return None


def _candidate_variant_keys(label: str, variant_code: str | None) -> list[str]:
    keys = [label]
    if variant_code:
        keys.append(variant_code)
        keys.append(variant_code.lower())
        keys.append(variant_code.upper())
    return keys


def _variant_manifest_payload(manifest_payload: dict[str, Any], variant_label: str, variant_code: str | None) -> dict[str, Any]:
    telemetry_payload = manifest_payload.get("telemetry", {})
    if not isinstance(telemetry_payload, dict):
        telemetry_payload = {}
    variants_payload = telemetry_payload.get("variants", {})
    if not isinstance(variants_payload, dict):
        variants_payload = {}
    for key in _candidate_variant_keys(variant_label, variant_code):
        payload = variants_payload.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _variant_labels(manifest_payload: dict[str, Any], results_df: Any) -> list[tuple[str | None, str]]:
    labels_payload = manifest_payload.get("selection", {}).get("labels", {})
    labels: list[tuple[str | None, str]] = []
    if isinstance(labels_payload, dict):
        for variant_code, label in labels_payload.items():
            text = str(label).strip()
            if text and all(existing_label != text for _, existing_label in labels):
                labels.append((str(variant_code).strip() or None, text))
    if results_df is not None and len(results_df) > 0 and "variant_label" in results_df.columns:
        for label in sorted({str(value).strip() for value in results_df["variant_label"].tolist() if str(value).strip()}):
            if all(existing_label != label for _, existing_label in labels):
                labels.append((None, label))
    return labels


def _real_provider_requested(provider: str, model: str) -> bool:
    joined = f"{provider} {model}".lower()
    return "fake" not in joined


def _availability_for_row(
    *,
    provider: str,
    model: str,
    calls_total: int,
    usage_present: bool,
    cost_known: bool,
    latency_p95_ms: float | None,
    require_usage: bool,
    require_latency: bool,
    provider_available: bool,
) -> tuple[str, str]:
    if not provider and not model and calls_total <= 0:
        return "unavailable", "no_provider_metadata"
    if not provider_available:
        return "provider_unavailable", "provider_env_missing_or_unavailable"
    if calls_total <= 0 and _real_provider_requested(provider, model):
        return "no_real_call", "no_real_call_observed"
    if require_latency and latency_p95_ms is None:
        return "partial", "latency_missing"
    if require_usage and not usage_present:
        return "partial", "usage_missing"
    if usage_present and not cost_known:
        return "partial", "usage_present_cost_missing"
    return "ok", "telemetry_available"


def _overall_availability(rows: list[dict[str, Any]]) -> str:
    statuses = [str(row.get("availability", "")).strip() for row in rows if str(row.get("availability", "")).strip()]
    if not statuses:
        return "unavailable"
    if all(status == "ok" for status in statuses):
        return "ok"
    if any(status == "provider_unavailable" for status in statuses):
        return "provider_unavailable"
    if any(status == "no_real_call" for status in statuses):
        return "no_real_call"
    if any(status == "partial" for status in statuses):
        return "partial"
    return statuses[0]


def _overall_real_call_status(rows: list[dict[str, Any]]) -> str:
    real_rows = []
    for row in rows:
        provider = str(row.get("provider", "")).strip().lower()
        if provider.startswith("fake") or provider == "fake":
            continue
        real_rows.append(row)
    if not real_rows:
        return "simulated"
    if any(int(_to_int(row.get("calls_total")) or 0) > 0 for row in real_rows):
        return "observed"
    return "missing_or_unavailable"


def _collect_source_paths(
    *,
    manifest_path: Path,
    results_long_path: Path,
    compare_summary_path: Path,
    source_compare_dir: Path | None,
) -> list[str]:
    out: list[str] = [str(manifest_path), str(results_long_path), str(compare_summary_path)]
    if source_compare_dir is not None:
        out.append(str(source_compare_dir))
        for name in ("table_model_cost_compare.csv", "table_model_stack_compare.csv"):
            found = sorted(source_compare_dir.rglob(name))
            out.extend(str(path) for path in found[:4])
    return out


def _repeat_summary_paths(suite_root: Path) -> list[Path]:
    repeats_root = suite_root / "repeats"
    if not repeats_root.exists():
        return []
    paths: list[Path] = []
    for repeat_dir in sorted(path for path in repeats_root.iterdir() if path.is_dir() and path.name.startswith("repeat_")):
        summary_path = repeat_dir / "provider_telemetry" / "summary.json"
        if summary_path.exists():
            paths.append(summary_path)
    return paths


def _repeat_summary_stats(suite_root: Path) -> dict[str, Any]:
    paths = _repeat_summary_paths(suite_root)
    payloads = [_read_json(path) for path in paths]
    latencies = [_to_float(payload.get("model_latency_p95_ms_mean")) for payload in payloads]
    latency_values = [float(value) for value in latencies if value is not None]
    latency_mean = _safe_mean(latency_values) if latency_values else None
    latency_std = 0.0
    if len(latency_values) > 1:
        mean = float(latency_mean or 0.0)
        latency_std = float((sum((value - mean) ** 2 for value in latency_values) / float(len(latency_values))) ** 0.5)
    return {
        "repeated_runs_total": int(len(paths)),
        "repeated_provider_unavailable_count": int(
            sum(1 for payload in payloads if str(payload.get("availability", "")).strip() == "provider_unavailable")
        ),
        "repeated_no_real_call_count": int(
            sum(1 for payload in payloads if str(payload.get("availability", "")).strip() == "no_real_call")
        ),
        "repeated_latency_p95_ms_mean": latency_mean,
        "repeated_latency_p95_ms_std": latency_std if latency_values else None,
        "repeat_summary_paths": [str(path) for path in paths],
    }


def _merge_reachability_summary(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    source_paths: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not summary:
        return rows, {}
    merged_rows = [dict(row) for row in rows]
    real_idx = -1
    for idx, row in enumerate(merged_rows):
        provider = str(row.get("provider", "")).strip().lower()
        if provider and not provider.startswith("fake"):
            real_idx = idx
            break
    if real_idx < 0:
        merged_rows.append(
            {
                "variant_label": "real",
                "variant_code": "b",
                "provider": str(summary.get("provider", "")).strip(),
                "model": str(summary.get("model", "")).strip(),
                "api_mode_used": str(summary.get("api_mode_tested", "")).strip(),
                "usage_present": bool(summary.get("usage_present", False)),
                "prompt_tokens": int(summary.get("prompt_tokens", 0) or 0) if summary.get("usage_present", False) else None,
                "completion_tokens": int(summary.get("completion_tokens", 0) or 0) if summary.get("usage_present", False) else None,
                "total_tokens": int(summary.get("total_tokens", 0) or 0) if summary.get("usage_present", False) else None,
                "cost_known": bool(summary.get("model_cost_usd_total") is not None),
                "model_cost_usd_total": _to_float(summary.get("model_cost_usd_total")),
                "model_cost_usd_mean_per_query": _to_float(summary.get("model_cost_usd_total")),
                "latency_p50_ms": _to_float(summary.get("latency_ms")),
                "latency_p95_ms": _to_float(summary.get("latency_ms")),
                "structured_parse_fail_rate": 0.0 if summary.get("structured_output_supported") is True else None,
                "planner_fallback_rate": 0.0,
                "calls_total": int(summary.get("calls_total", 0) or 0),
                "calls_with_usage": int(summary.get("calls_with_usage", 0) or 0),
                "calls_with_cost": int(summary.get("calls_with_cost", 0) or 0),
                "availability": "ok" if str(summary.get("proof_status", "")).strip() == "ok" else "partial",
                "availability_reason": str(summary.get("real_call_status", "")).strip(),
                "provider_available": bool(summary.get("reachable", False)),
                "telemetry_source_paths": json.dumps(source_paths, ensure_ascii=False),
            }
        )
        real_idx = len(merged_rows) - 1
    row = dict(merged_rows[real_idx])
    row["provider"] = str(row.get("provider", "")).strip() or str(summary.get("provider", "")).strip()
    row["model"] = str(row.get("model", "")).strip() or str(summary.get("model", "")).strip()
    row["api_mode_used"] = str(summary.get("api_mode_tested", row.get("api_mode_used", ""))).strip()
    if summary.get("usage_present") is True:
        row["usage_present"] = True
        row["prompt_tokens"] = int(summary.get("prompt_tokens", row.get("prompt_tokens", 0)) or 0)
        row["completion_tokens"] = int(summary.get("completion_tokens", row.get("completion_tokens", 0)) or 0)
        row["total_tokens"] = int(summary.get("total_tokens", row.get("total_tokens", 0)) or 0)
    if _to_float(summary.get("model_cost_usd_total")) is not None:
        row["cost_known"] = True
        row["model_cost_usd_total"] = _to_float(summary.get("model_cost_usd_total"))
        calls_total = int(summary.get("calls_total", 0) or 0)
        if calls_total > 0:
            row["model_cost_usd_mean_per_query"] = float(float(summary.get("model_cost_usd_total")) / float(calls_total))
    latency_ms = _to_float(summary.get("latency_ms"))
    if latency_ms is not None:
        row_latency_p50 = _to_float(row.get("latency_p50_ms"))
        row_latency_p95 = _to_float(row.get("latency_p95_ms"))
        row["latency_p50_ms"] = latency_ms if row_latency_p50 is None else row_latency_p50
        row["latency_p95_ms"] = latency_ms if row_latency_p95 is None else max(row_latency_p95, latency_ms)
    if summary.get("structured_output_supported") is True:
        row["structured_parse_fail_rate"] = 0.0
    elif summary.get("structured_output_supported") is False:
        row["structured_parse_fail_rate"] = 1.0
    row["planner_fallback_rate"] = _to_float(row.get("planner_fallback_rate")) if _to_float(row.get("planner_fallback_rate")) is not None else 0.0
    row["calls_total"] = max(int(_to_int(row.get("calls_total")) or 0), int(summary.get("calls_total", 0) or 0))
    row["calls_with_usage"] = max(int(_to_int(row.get("calls_with_usage")) or 0), int(summary.get("calls_with_usage", 0) or 0))
    row["calls_with_cost"] = max(int(_to_int(row.get("calls_with_cost")) or 0), int(summary.get("calls_with_cost", 0) or 0))
    proof_status = str(summary.get("proof_status", "")).strip()
    if proof_status == "ok":
        row["availability"] = "ok"
        row["availability_reason"] = "provider_reachability_ok"
    elif proof_status == "partial":
        row["availability"] = "partial"
        row["availability_reason"] = str(summary.get("real_call_status", "provider_reachability_partial")).strip()
    row["provider_available"] = bool(summary.get("reachable", False))
    row["proof_status"] = proof_status
    row["structured_output_supported"] = summary.get("structured_output_supported", "unknown")
    row["real_call_status"] = "observed" if proof_status in {"ok", "partial"} and bool(summary.get("reachable", False)) else str(summary.get("real_call_status", "missing_or_unavailable"))
    telemetry_paths = list(source_paths)
    telemetry_paths.append(str(Path(summary.get("provider_health_config", "")).resolve())) if str(summary.get("provider_health_config", "")).strip() else None
    row["telemetry_source_paths"] = json.dumps(sorted(dict.fromkeys(telemetry_paths)), ensure_ascii=False)
    merged_rows[real_idx] = row
    return merged_rows, summary


def build_provider_telemetry_table(
    *,
    suite_dir: str | Path,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    manifest_path = suite_root / "manifest" / "experiment_manifest.yaml"
    compare_summary_path = suite_root / "compare" / "compare_summary.json"
    results_long_path = suite_root / "ledger" / "results_long.csv"
    manifest_payload = _load_yaml(manifest_path)
    compare_summary = _read_json(compare_summary_path)
    results_df = _read_csv(results_long_path)
    if len(results_df) > 0 and "source_kind" in results_df.columns:
        aggregate_df = results_df.loc[results_df["source_kind"].astype(str) == "aggregate"].copy()
    else:
        aggregate_df = results_df.copy()

    require_usage = bool(
        manifest_payload.get("telemetry_require_usage", manifest_payload.get("telemetry", {}).get("require_usage", False))
    )
    require_latency = bool(
        manifest_payload.get("telemetry_require_latency", manifest_payload.get("telemetry", {}).get("require_latency", False))
    )
    telemetry_enabled = bool(
        manifest_payload.get("telemetry_enabled", manifest_payload.get("telemetry", {}).get("enabled", False))
    )

    source_compare_dir = _resolve_optional_path(
        str(compare_summary.get("compare_dir", manifest_payload.get("selection", {}).get("compare_dir", ""))).strip(),
        manifest_path.parent,
    )
    reachability_summary = _read_json(suite_root / "provider_reachability" / "summary.json")
    source_paths = _collect_source_paths(
        manifest_path=manifest_path,
        results_long_path=results_long_path,
        compare_summary_path=compare_summary_path,
        source_compare_dir=source_compare_dir,
    )

    rows: list[dict[str, Any]] = []
    for variant_code, variant_label in _variant_labels(manifest_payload, aggregate_df):
        manifest_variant = _variant_manifest_payload(manifest_payload, variant_label, variant_code)
        variant_df = aggregate_df.copy()
        if len(variant_df) > 0 and "variant_label" in variant_df.columns:
            variant_df = variant_df.loc[variant_df["variant_label"].astype(str) == variant_label].copy()

        provider = _pick_text(variant_df, ["provider", "provider_name"]) or str(manifest_variant.get("provider", "")).strip()
        model = _pick_text(variant_df, ["model", "model_name"]) or str(manifest_variant.get("model", "")).strip()
        api_mode_used = _pick_text(variant_df, ["api_mode_used", "strategy_used"]) or str(
            manifest_variant.get("api_mode_used", "")
        ).strip()
        prompt_tokens = _pick_number(
            variant_df,
            ["prompt_tokens", "input_tokens", "usage_prompt_tokens"],
            reducer="sum",
        )
        if prompt_tokens is None:
            prompt_tokens = _to_float(manifest_variant.get("prompt_tokens"))
        completion_tokens = _pick_number(
            variant_df,
            ["completion_tokens", "output_tokens", "usage_completion_tokens"],
            reducer="sum",
        )
        if completion_tokens is None:
            completion_tokens = _to_float(manifest_variant.get("completion_tokens"))
        total_tokens = _pick_number(variant_df, ["total_tokens", "usage_total_tokens"], reducer="sum")
        if total_tokens is None:
            total_tokens = _to_float(manifest_variant.get("total_tokens"))

        usage_present = _pick_bool(variant_df, ["usage_present"])
        if usage_present is None:
            usage_present = bool(
                manifest_variant.get("usage_present", False)
                or (_pick_number(variant_df, ["calls_with_usage"], reducer="sum") or 0.0) > 0.0
            )

        cost_known = _pick_bool(variant_df, ["cost_known"])
        if cost_known is None:
            cost_known = bool(
                manifest_variant.get("cost_known", False)
                or _pick_number(variant_df, ["model_cost_usd_total", "model_cost_usd_mean_per_query"]) is not None
            )

        model_cost_total = _pick_number(variant_df, ["model_cost_usd_total"], reducer="mean")
        if model_cost_total is None:
            model_cost_total = _to_float(manifest_variant.get("model_cost_usd_total"))
        model_cost_mean = _pick_number(variant_df, ["model_cost_usd_mean_per_query"], reducer="mean")
        if model_cost_mean is None:
            model_cost_mean = _to_float(manifest_variant.get("model_cost_usd_mean_per_query"))
        latency_p50_ms = _pick_number(variant_df, ["latency_p50_ms", "model_latency_p50_ms", "e2e_ms_p50"], reducer="mean")
        if latency_p50_ms is None:
            latency_p50_ms = _to_float(manifest_variant.get("latency_p50_ms"))
        latency_p95_ms = _pick_number(
            variant_df,
            ["latency_p95_ms", "model_latency_p95_ms", "e2e_ms_p95"],
            reducer="mean",
        )
        if latency_p95_ms is None:
            latency_p95_ms = _to_float(manifest_variant.get("latency_p95_ms"))
        structured_parse_fail_rate = _pick_number(
            variant_df,
            ["structured_parse_fail_rate", "parse_fail_rate"],
            reducer="mean",
        )
        if structured_parse_fail_rate is None:
            structured_parse_fail_rate = _to_float(manifest_variant.get("structured_parse_fail_rate"))
        planner_fallback_rate = _pick_number(variant_df, ["planner_fallback_rate"], reducer="mean")
        if planner_fallback_rate is None:
            planner_fallback_rate = _to_float(manifest_variant.get("planner_fallback_rate"))

        calls_total = _to_int(_pick_number(variant_df, ["calls_total", "model_requests_total"], reducer="sum"))
        if calls_total is None:
            calls_total = _to_int(manifest_variant.get("calls_total")) or 0
        calls_with_usage = _to_int(_pick_number(variant_df, ["calls_with_usage"], reducer="sum"))
        if calls_with_usage is None:
            calls_with_usage = _to_int(manifest_variant.get("calls_with_usage")) or (calls_total if usage_present else 0)
        calls_with_cost = _to_int(_pick_number(variant_df, ["calls_with_cost"], reducer="sum"))
        if calls_with_cost is None:
            calls_with_cost = _to_int(manifest_variant.get("calls_with_cost")) or (calls_total if cost_known else 0)

        api_key_env = str(manifest_variant.get("api_key_env", "")).strip()
        provider_available = True
        if api_key_env:
            provider_available = bool(os.environ.get(api_key_env))

        availability, availability_reason = _availability_for_row(
            provider=provider,
            model=model,
            calls_total=int(calls_total),
            usage_present=bool(usage_present),
            cost_known=bool(cost_known),
            latency_p95_ms=latency_p95_ms,
            require_usage=require_usage,
            require_latency=require_latency,
            provider_available=provider_available,
        )
        rows.append(
            {
                "variant_label": variant_label,
                "variant_code": variant_code or "",
                "provider": provider,
                "model": model,
                "api_mode_used": api_mode_used,
                "usage_present": bool(usage_present),
                "prompt_tokens": _to_int(prompt_tokens),
                "completion_tokens": _to_int(completion_tokens),
                "total_tokens": _to_int(total_tokens),
                "cost_known": bool(cost_known),
                "model_cost_usd_total": model_cost_total,
                "model_cost_usd_mean_per_query": model_cost_mean,
                "latency_p50_ms": latency_p50_ms,
                "latency_p95_ms": latency_p95_ms,
                "structured_parse_fail_rate": structured_parse_fail_rate,
                "planner_fallback_rate": planner_fallback_rate,
                "calls_total": int(calls_total),
                "calls_with_usage": int(calls_with_usage),
                "calls_with_cost": int(calls_with_cost),
                "availability": availability,
                "availability_reason": availability_reason,
                "provider_available": provider_available,
                "telemetry_source_paths": json.dumps(source_paths, ensure_ascii=False),
            }
        )

    rows, reachability_summary = _merge_reachability_summary(
        rows=rows,
        summary=reachability_summary,
        source_paths=source_paths,
    )
    out_df = lib.DataFrame(rows)
    summary_source_paths = list(source_paths)
    if reachability_summary:
        summary_source_paths.append(str((suite_root / "provider_reachability" / "summary.json").resolve()))
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "telemetry_enabled": telemetry_enabled,
        "telemetry_require_usage": require_usage,
        "telemetry_require_latency": require_latency,
        "availability": _overall_availability(rows),
        "usage_present_rate": _safe_mean([1.0 if bool(row.get("usage_present")) else 0.0 for row in rows]),
        "model_cost_known_rate": _safe_mean([1.0 if bool(row.get("cost_known")) else 0.0 for row in rows]),
        "model_latency_p95_ms_mean": _safe_mean([_to_float(row.get("latency_p95_ms")) for row in rows]),
        "structured_parse_fail_rate_mean": _safe_mean(
            [_to_float(row.get("structured_parse_fail_rate")) for row in rows]
        ),
        "planner_fallback_rate": _safe_mean([_to_float(row.get("planner_fallback_rate")) for row in rows]),
        "calls_total": _safe_sum([_to_int(row.get("calls_total")) for row in rows]),
        "calls_with_usage": _safe_sum([_to_int(row.get("calls_with_usage")) for row in rows]),
        "calls_with_cost": _safe_sum([_to_int(row.get("calls_with_cost")) for row in rows]),
        "provider_unavailable_count": sum(1 for row in rows if row.get("availability") == "provider_unavailable"),
        "no_real_call_count": sum(1 for row in rows if row.get("availability") == "no_real_call"),
        "real_call_status": _overall_real_call_status(rows),
        "telemetry_source_paths": sorted(dict.fromkeys(summary_source_paths)),
        "variants_total": len(rows),
        "proof_status": str(reachability_summary.get("proof_status", "")) if reachability_summary else "",
        "structured_output_supported": reachability_summary.get("structured_output_supported", "unknown")
        if reachability_summary
        else "unknown",
        "provider_reachability_dir": str((suite_root / "provider_reachability").resolve())
        if (suite_root / "provider_reachability").exists()
        else None,
        "reachability_base_url": str(reachability_summary.get("base_url", "")) if reachability_summary else "",
    }
    summary.update(_repeat_summary_stats(suite_root))
    return out_df, summary


def write_provider_telemetry_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    telemetry_df, summary = build_provider_telemetry_table(suite_dir=suite_dir)
    by_variant_csv = out_root / "by_variant.csv"
    summary_json = out_root / "summary.json"
    telemetry_df.to_csv(by_variant_csv, index=False)
    _write_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2))
    return {
        "summary_json": summary_json,
        "by_variant_csv": by_variant_csv,
        "summary": summary,
        "rows_total": int(len(telemetry_df)),
    }


def load_provider_telemetry_outputs(telemetry_dir: str | Path | None) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    if telemetry_dir is None:
        return lib.DataFrame(), {}
    root = Path(telemetry_dir).resolve()
    if not root.exists():
        return lib.DataFrame(), {}
    df = _read_csv(root / "by_variant.csv")
    summary = _read_json(root / "summary.json")
    return df, summary
