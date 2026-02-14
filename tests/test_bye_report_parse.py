from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.report import load_report_json, parse_bye_report


def test_parse_bye_report_candidate_keys(tmp_path: Path) -> None:
    report = {
        "qualityScore": 0.81,
        "criticalFN": 2,
        "latency": {"p50_ms": 12.5, "p95_ms": 28.1},
        "extra": {"value": 7.0},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    payload = load_report_json(path)
    metrics = parse_bye_report(payload)
    data = metrics.to_dict()
    assert data["bye_status"] == "ok"
    assert data["bye_primary_score"] == 0.81
    assert data["bye_critical_fn"] == 2.0
    assert data["bye_latency_p50_ms"] == 12.5
    assert data["bye_latency_p95_ms"] == 28.1
    assert isinstance(data["numeric_metrics"], dict) and data["numeric_metrics"]


def test_parse_bye_report_missing_fields() -> None:
    metrics = parse_bye_report({"unknown": "text"})
    data = metrics.to_dict()
    assert data["bye_status"] in {"ok", "parse_error"}
    assert data["bye_primary_score"] is None
    assert isinstance(data["bye_warnings"], list)
