from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ADMISSION_PROFILES: dict[str, dict[str, float | int | bool]] = {
    "main_real": {
        "min_selected_uids": 3,
        "min_coverage_score_mean": 2.5,
        "max_no_data_rate": 0.25,
        "min_significance_available_rate": 0.50,
        "min_effect_size_nonzero_rate": 0.34,
        "max_provider_noise_parse_fail_rate": 0.10,
        "max_provider_noise_fallback_rate": 0.10,
        "min_usage_present_rate": 0.25,
        "allow_partial": True,
    }
}


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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "ok"}


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


def _resolve_profile(manifest_payload: dict[str, Any], requested_profile: str | None) -> tuple[str, dict[str, Any]]:
    profile_name = str(
        requested_profile
        or manifest_payload.get("admission_profile")
        or manifest_payload.get("admission", {}).get("profile")
        or ""
    ).strip()
    if not profile_name:
        return "", {}
    base = ADMISSION_PROFILES.get(profile_name)
    if base is None:
        raise ValueError(f"Unknown admission profile: {profile_name}")
    thresholds = dict(base)
    admission_payload = manifest_payload.get("admission", {})
    if not isinstance(admission_payload, dict):
        admission_payload = {}
    field_names = (
        "min_selected_uids",
        "min_coverage_score_mean",
        "max_no_data_rate",
        "min_significance_available_rate",
        "min_effect_size_nonzero_rate",
        "max_provider_noise_parse_fail_rate",
        "max_provider_noise_fallback_rate",
        "min_usage_present_rate",
        "allow_partial",
    )
    for key in field_names:
        if key in admission_payload:
            thresholds[key] = admission_payload[key]
        elif key in manifest_payload:
            thresholds[key] = manifest_payload[key]
    return profile_name, thresholds


def _coverage_mean(payload: dict[str, Any]) -> float:
    coverage_stats = payload.get("coverage_score_stats", {})
    if not isinstance(coverage_stats, dict):
        coverage_stats = _parse_json_dict(coverage_stats)
    return float(_to_float(coverage_stats.get("mean")) or 0.0)


def _no_data_rate_from_health(health_snapshot: dict[str, Any]) -> float:
    gate_metrics = health_snapshot.get("gate", {}).get("metrics", {})
    if isinstance(gate_metrics, dict):
        gate_rate = _to_float(gate_metrics.get("no_data_rate"))
        if gate_rate is not None:
            return float(gate_rate)
    counts = health_snapshot.get("overall_no_data_reason_counts", {})
    if not isinstance(counts, dict):
        return 1.0
    total = 0
    ok_count = 0
    for key, raw in counts.items():
        value = int(_to_float(raw) or 0)
        total += value
        if str(key) == "ok":
            ok_count += value
    if total <= 0:
        return 1.0
    return float(max(total - ok_count, 0) / total)


def evaluate_admission(
    *,
    manifest_payload: dict[str, Any],
    health_snapshot: dict[str, Any],
    diagnosis_snapshot: dict[str, Any],
    provider_summary: dict[str, Any],
    delta_audit_snapshot: dict[str, Any],
    admission_profile: str | None = None,
) -> dict[str, Any]:
    profile_name, thresholds = _resolve_profile(manifest_payload, admission_profile)
    if not profile_name:
        return {
            "profile": "",
            "thresholds": {},
            "admission_status": "skipped",
            "admission_fail_reasons": [],
            "admission_metrics": {},
        }

    gate_payload = health_snapshot.get("gate", {})
    gate_metrics = gate_payload.get("metrics", {}) if isinstance(gate_payload, dict) else {}
    if not isinstance(gate_metrics, dict):
        gate_metrics = {}

    selected_uids_count = int(
        _to_float(gate_metrics.get("selected_uids_count")) or diagnosis_snapshot.get("selected_uids_count") or 0
    )
    coverage_score_mean = float(_to_float(gate_metrics.get("coverage_score_mean")) or 0.0)
    if coverage_score_mean <= 0.0:
        coverage_score_mean = _coverage_mean(diagnosis_snapshot) or _coverage_mean(health_snapshot.get("selection", {}))
    no_data_rate = _no_data_rate_from_health(health_snapshot)
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
    parse_fail_rate = _to_float(provider_summary.get("structured_parse_fail_rate_mean"))
    fallback_rate = _to_float(provider_summary.get("planner_fallback_rate"))
    usage_present_rate = _to_float(provider_summary.get("usage_present_rate"))
    provider_availability = str(provider_summary.get("availability", "unavailable")).strip() or "unavailable"
    main_recommendation = str(delta_audit_snapshot.get("main_recommendation", "")).strip()

    metrics = {
        "selected_uids_count": selected_uids_count,
        "coverage_score_mean": float(coverage_score_mean),
        "no_data_rate": float(no_data_rate),
        "significance_available_rate": float(significance_available_rate),
        "effect_size_nonzero_rate": float(effect_size_nonzero_rate),
        "provider_noise_structured_parse_fail_rate_mean": parse_fail_rate,
        "provider_noise_planner_fallback_rate": fallback_rate,
        "provider_noise_usage_present_rate": usage_present_rate,
        "provider_availability": provider_availability,
        "gate_status": str(gate_payload.get("gate_status", "skipped")),
        "main_recommendation": main_recommendation,
    }

    fail_reasons: list[str] = []
    if selected_uids_count < int(thresholds["min_selected_uids"]):
        fail_reasons.append(f"selected_uids_count<{int(thresholds['min_selected_uids'])} ({selected_uids_count})")
    if coverage_score_mean < float(thresholds["min_coverage_score_mean"]):
        fail_reasons.append(
            f"coverage_score_mean<{float(thresholds['min_coverage_score_mean']):.2f} ({coverage_score_mean:.3f})"
        )
    if no_data_rate > float(thresholds["max_no_data_rate"]):
        fail_reasons.append(f"no_data_rate>{float(thresholds['max_no_data_rate']):.2f} ({no_data_rate:.3f})")
    if significance_available_rate < float(thresholds["min_significance_available_rate"]):
        fail_reasons.append(
            "significance_available_rate<"
            f"{float(thresholds['min_significance_available_rate']):.2f} ({significance_available_rate:.3f})"
        )
    if effect_size_nonzero_rate < float(thresholds["min_effect_size_nonzero_rate"]):
        fail_reasons.append(
            "effect_size_nonzero_rate<"
            f"{float(thresholds['min_effect_size_nonzero_rate']):.2f} ({effect_size_nonzero_rate:.3f})"
        )
    if parse_fail_rate is not None and parse_fail_rate > float(thresholds["max_provider_noise_parse_fail_rate"]):
        fail_reasons.append(
            "structured_parse_fail_rate_mean>"
            f"{float(thresholds['max_provider_noise_parse_fail_rate']):.2f} ({parse_fail_rate:.3f})"
        )
    if fallback_rate is not None and fallback_rate > float(thresholds["max_provider_noise_fallback_rate"]):
        fail_reasons.append(
            "planner_fallback_rate>"
            f"{float(thresholds['max_provider_noise_fallback_rate']):.2f} ({fallback_rate:.3f})"
        )
    if usage_present_rate is not None and usage_present_rate < float(thresholds["min_usage_present_rate"]):
        fail_reasons.append(
            "usage_present_rate<"
            f"{float(thresholds['min_usage_present_rate']):.2f} ({usage_present_rate:.3f})"
        )
    if provider_availability in {"provider_unavailable", "no_real_call", "unavailable"}:
        fail_reasons.append(f"provider_availability={provider_availability}")

    allow_partial = _to_bool(thresholds.get("allow_partial"))
    hard_fail = selected_uids_count <= 0 or no_data_rate >= 1.0 or provider_availability == "provider_unavailable"
    if not fail_reasons:
        status = "ok"
    elif hard_fail or not allow_partial:
        status = "fail"
    else:
        status = "partial"

    return {
        "profile": profile_name,
        "thresholds": thresholds,
        "admission_status": status,
        "admission_fail_reasons": fail_reasons,
        "admission_metrics": metrics,
    }


def write_admission_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
    admission_profile: str | None = None,
    health_snapshot: dict[str, Any] | None = None,
    diagnosis_snapshot: dict[str, Any] | None = None,
    provider_summary: dict[str, Any] | None = None,
    delta_audit_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suite_root = Path(suite_dir).resolve()
    out_root = Path(out_dir).resolve()
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    health_payload = health_snapshot or _read_json(suite_root / "result_health" / "snapshot.json")
    diagnosis_payload = diagnosis_snapshot or _read_json(suite_root / "result_diagnosis" / "snapshot.json")
    provider_payload = provider_summary or _read_json(suite_root / "provider_telemetry" / "summary.json")
    delta_payload = delta_audit_snapshot or _read_json(suite_root / "delta_audit" / "snapshot.json")

    decision = evaluate_admission(
        manifest_payload=manifest_payload,
        health_snapshot=health_payload,
        diagnosis_snapshot=diagnosis_payload,
        provider_summary=provider_payload,
        delta_audit_snapshot=delta_payload,
        admission_profile=admission_profile,
    )
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "profile": decision.get("profile", ""),
        "thresholds": decision.get("thresholds", {}),
        "admission_status": decision.get("admission_status", "skipped"),
        "admission_fail_reasons": decision.get("admission_fail_reasons", []),
        "admission_metrics": decision.get("admission_metrics", {}),
        "health_gate": health_payload.get("gate", {}),
        "provider_noise_summary": provider_payload,
        "delta_audit_summary": {
            "main_recommendation": delta_payload.get("main_recommendation"),
            "recommended_action_counts": delta_payload.get("recommended_action_counts", {}),
        },
    }
    snapshot_path = out_root / "snapshot.json"
    report_path = out_root / "report.md"
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
    report_lines = [
        "# Admission Control",
        "",
        f"- suite_dir: `{suite_root}`",
        f"- profile: `{snapshot.get('profile')}`",
        f"- admission_status: `{snapshot.get('admission_status')}`",
        f"- admission_fail_reasons: `{snapshot.get('admission_fail_reasons')}`",
        f"- admission_metrics: `{json.dumps(snapshot.get('admission_metrics', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- delta_audit_summary: `{json.dumps(snapshot.get('delta_audit_summary', {}), ensure_ascii=False, sort_keys=True)}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    return {
        "snapshot_json": snapshot_path,
        "report_md": report_path,
        "admission_status": snapshot.get("admission_status", "skipped"),
        "admission_fail_reasons": snapshot.get("admission_fail_reasons", []),
        "admission_metrics": snapshot.get("admission_metrics", {}),
    }
