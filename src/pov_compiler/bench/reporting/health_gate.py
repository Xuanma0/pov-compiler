from __future__ import annotations

import json
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None


HEALTH_GATE_PROFILES: dict[str, dict[str, float | int]] = {
    "main_real": {
        "selected_uids_count_min": 2,
        "significance_available_rate_min": 0.50,
        "effect_size_nonzero_rate_min": 0.34,
        "max_missing_metric_rate": 0.20,
        "max_no_data_rate": 0.25,
    }
}


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for health-gate evaluation.")
    return pd


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _parse_breakdown(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        out: dict[str, int] = {}
        for key, raw in value.items():
            try:
                out[str(key)] = int(raw)
            except Exception:
                out[str(key)] = 0
        return out
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in payload.items():
        try:
            out[str(key)] = int(raw)
        except Exception:
            out[str(key)] = 0
    return out


def _pick_overall_row(table_df: Any) -> dict[str, Any]:
    lib = _require_pandas()
    if table_df is None or len(table_df) == 0:
        return {}
    overall = table_df.loc[table_df["task"].astype(str) == "overall"].copy()
    if len(overall) > 0:
        return overall.iloc[0].to_dict()
    return lib.DataFrame(table_df).iloc[0].to_dict()


def evaluate_health_gate(table_df: Any, snapshot: dict[str, Any], gate_profile: str | None) -> dict[str, Any]:
    profile_name = str(gate_profile or "").strip()
    if not profile_name:
        return {
            "profile": "",
            "thresholds": {},
            "metrics": {},
            "gate_status": "skipped",
            "gate_fail_reasons": [],
        }

    thresholds = HEALTH_GATE_PROFILES.get(profile_name)
    if thresholds is None:
        raise ValueError(f"Unknown health gate profile: {profile_name}")

    overall = _pick_overall_row(table_df)
    selected_uids_count = int(_to_float(overall.get("selected_uids_count")) or 0)
    significance_available_rate = float(_to_float(overall.get("significance_available_rate")) or 0.0)
    effect_size_nonzero_rate = float(_to_float(overall.get("effect_size_nonzero_rate")) or 0.0)
    missing_metric_rate = float(_to_float(overall.get("missing_metric_rate")) or 0.0)

    counts = snapshot.get("overall_no_data_reason_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    parsed_counts = _parse_breakdown(counts)
    total_count = int(sum(parsed_counts.values()))
    ok_count = int(parsed_counts.get("ok", 0))
    no_data_count = max(total_count - ok_count, 0)
    no_data_rate = float(no_data_count / total_count) if total_count > 0 else 1.0

    metrics = {
        "selected_uids_count": selected_uids_count,
        "significance_available_rate": significance_available_rate,
        "effect_size_nonzero_rate": effect_size_nonzero_rate,
        "missing_metric_rate": missing_metric_rate,
        "no_data_rate": no_data_rate,
        "overall_no_data_reason_counts": parsed_counts,
    }

    fail_reasons: list[str] = []
    if selected_uids_count < int(thresholds["selected_uids_count_min"]):
        fail_reasons.append(
            f"selected_uids_count<{int(thresholds['selected_uids_count_min'])} ({selected_uids_count})"
        )
    if significance_available_rate < float(thresholds["significance_available_rate_min"]):
        fail_reasons.append(
            "significance_available_rate<"
            f"{float(thresholds['significance_available_rate_min']):.2f} ({significance_available_rate:.3f})"
        )
    if effect_size_nonzero_rate < float(thresholds["effect_size_nonzero_rate_min"]):
        fail_reasons.append(
            "effect_size_nonzero_rate<"
            f"{float(thresholds['effect_size_nonzero_rate_min']):.2f} ({effect_size_nonzero_rate:.3f})"
        )
    if missing_metric_rate > float(thresholds["max_missing_metric_rate"]):
        fail_reasons.append(
            "missing_metric_rate>"
            f"{float(thresholds['max_missing_metric_rate']):.2f} ({missing_metric_rate:.3f})"
        )
    if no_data_rate > float(thresholds["max_no_data_rate"]):
        fail_reasons.append(
            f"no_data_rate>{float(thresholds['max_no_data_rate']):.2f} ({no_data_rate:.3f})"
        )

    return {
        "profile": profile_name,
        "thresholds": {key: value for key, value in thresholds.items()},
        "metrics": metrics,
        "gate_status": "fail" if fail_reasons else "ok",
        "gate_fail_reasons": fail_reasons,
    }
