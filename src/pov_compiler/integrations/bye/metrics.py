from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float], depth: int = 0, max_depth: int = 2) -> None:
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
        for idx, v in enumerate(value[:20]):
            key = f"{prefix}[{idx}]"
            _flatten_numeric(key, v, out, depth + 1, max_depth=max_depth)
        return


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

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid_report",
            "report_path": str(report_path),
            "summary_keys": [],
            "numeric_metrics": {},
            "error": str(exc),
        }

    if not isinstance(payload, dict):
        return {
            "status": "invalid_report",
            "report_path": str(report_path),
            "summary_keys": [],
            "numeric_metrics": {},
            "error": "report.json is not an object",
        }

    numeric: dict[str, float] = {}
    _flatten_numeric("", payload, numeric, depth=0, max_depth=2)
    summary_keys = sorted([str(k) for k in payload.keys()])
    return {
        "status": "ok",
        "report_path": str(report_path),
        "summary_keys": summary_keys,
        "numeric_metrics": dict(sorted(numeric.items())),
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

