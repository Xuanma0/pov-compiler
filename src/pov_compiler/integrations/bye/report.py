from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float], depth: int = 0, max_depth: int = 3) -> None:
    if depth > max_depth:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_numeric(key, v, out, depth + 1, max_depth=max_depth)
        return
    if isinstance(value, list):
        for i, item in enumerate(value[:32]):
            key = f"{prefix}[{i}]"
            _flatten_numeric(key, item, out, depth + 1, max_depth=max_depth)


def _norm_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_")


def _safe_float(value: Any) -> float | None:
    try:
        num = float(value)
    except Exception:
        return None
    if num != num:
        return None
    return float(num)


def load_report_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass
class ByeReportMetrics:
    bye_status: str = "missing_report"
    bye_primary_score: float | None = None
    bye_critical_fn: float | None = None
    bye_latency_p50_ms: float | None = None
    bye_latency_p95_ms: float | None = None
    bye_warnings: list[str] = field(default_factory=list)
    report_path: str = ""
    summary_keys: list[str] = field(default_factory=list)
    numeric_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def parse_bye_report(report: dict[str, Any]) -> ByeReportMetrics:
    if not isinstance(report, dict) or not report:
        return ByeReportMetrics(bye_status="missing_report", bye_warnings=["report payload missing or invalid"])

    flat_numeric: dict[str, float] = {}
    _flatten_numeric("", report, flat_numeric, depth=0, max_depth=3)
    flat_norm = {_norm_key(k): v for k, v in flat_numeric.items()}

    warnings: list[str] = []

    def _pick(candidates: list[str]) -> float | None:
        for cand in candidates:
            v = flat_norm.get(_norm_key(cand))
            if v is not None:
                return float(v)
        return None

    primary_candidates = [
        "qualityScore",
        "quality_score",
        "primary_score",
        "overall_score",
        "score",
        "summary.score",
        "metrics.score",
        "acc",
        "accuracy",
    ]
    critical_candidates = [
        "critical_fn",
        "criticalFN",
        "critical_failures",
        "critical_failure_count",
        "critical_fn_count",
        "critical_fn_rate",
        "safety.critical_fn",
        "safety.critical_fn_rate",
        "summary.critical_fn",
    ]
    latency_p50_candidates = [
        "latency_p50_ms",
        "latency.p50_ms",
        "latency.p50",
        "latency_ms_p50",
        "p50_latency_ms",
        "latency.percentile_50",
        "latency.p50ms",
    ]
    latency_p95_candidates = [
        "latency_p95_ms",
        "latency.p95_ms",
        "latency.p95",
        "latency_ms_p95",
        "p95_latency_ms",
        "latency.percentile_95",
        "latency.p95ms",
    ]

    primary = _pick(primary_candidates)
    critical = _pick(critical_candidates)
    p50 = _pick(latency_p50_candidates)
    p95 = _pick(latency_p95_candidates)

    # Fallback for single latency metric.
    if p50 is None:
        one_latency = _pick(["latency_ms", "latency", "mean_latency_ms"])
        if one_latency is not None:
            p50 = one_latency
    if p95 is None:
        one_latency = _pick(["latency_ms", "latency", "mean_latency_ms"])
        if one_latency is not None:
            p95 = one_latency

    if primary is None:
        warnings.append("primary score key not found")
    if critical is None:
        warnings.append("critical_fn key not found")
    if p50 is None:
        warnings.append("latency p50 key not found")
    if p95 is None:
        warnings.append("latency p95 key not found")

    status = "ok"
    if not flat_numeric:
        status = "parse_error"
        warnings.append("no numeric fields found in report")

    return ByeReportMetrics(
        bye_status=status,
        bye_primary_score=_safe_float(primary),
        bye_critical_fn=_safe_float(critical),
        bye_latency_p50_ms=_safe_float(p50),
        bye_latency_p95_ms=_safe_float(p95),
        bye_warnings=warnings,
        summary_keys=sorted([str(k) for k in report.keys()]),
        numeric_metrics=dict(sorted(flat_numeric.items())),
    )

