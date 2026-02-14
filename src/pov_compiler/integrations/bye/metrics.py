from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pov_compiler.integrations.bye.report import load_report_json, parse_bye_report as parse_bye_report_payload

def parse_bye_report(report_dir: Path) -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_path = report_dir / "report.json"
    if not report_path.exists():
        return {
            "status": "missing_report",
            "report_path": str(report_path),
            "summary_keys": [],
            "numeric_metrics": {},
        }

    payload = load_report_json(report_path)
    if not payload:
        return {
            "status": "invalid_report",
            "report_path": str(report_path),
            "summary_keys": [],
            "numeric_metrics": {},
            "error": "invalid_or_empty_report_json",
        }
    parsed = parse_bye_report_payload(payload)
    summary_keys = sorted([str(k) for k in payload.keys()])
    return {
        "status": str(parsed.bye_status),
        "report_path": str(report_path),
        "summary_keys": summary_keys,
        "numeric_metrics": dict(sorted(parsed.numeric_metrics.items())),
        "bye_primary_score": parsed.bye_primary_score,
        "bye_critical_fn": parsed.bye_critical_fn,
        "bye_latency_p50_ms": parsed.bye_latency_p50_ms,
        "bye_latency_p95_ms": parsed.bye_latency_p95_ms,
        "bye_warnings": list(parsed.bye_warnings),
    }


def save_bye_metrics(metrics: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bye_metrics.json"
    csv_path = out_dir / "bye_metrics.csv"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    row: dict[str, Any] = {
        "status": str(metrics.get("status", "")),
        "report_path": str(metrics.get("report_path", "")),
        "summary_keys": ";".join([str(x) for x in metrics.get("summary_keys", [])]),
    }
    for key in (
        "bye_primary_score",
        "bye_critical_fn",
        "bye_latency_p50_ms",
        "bye_latency_p95_ms",
    ):
        val = metrics.get(key)
        try:
            row[key] = float(val) if val is not None else ""
        except Exception:
            row[key] = ""
    warnings = metrics.get("bye_warnings", [])
    if isinstance(warnings, list):
        row["bye_warnings"] = ";".join([str(x) for x in warnings])

    numeric = metrics.get("numeric_metrics", {})
    if isinstance(numeric, dict):
        for k, v in numeric.items():
            try:
                row[str(k)] = float(v)
            except Exception:
                continue

    fieldnames = list(row.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return json_path, csv_path
