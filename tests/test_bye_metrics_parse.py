from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.metrics import parse_bye_report, save_bye_metrics


def test_parse_bye_report_and_save_csv(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {"score": 0.8, "count": 12, "name": "demo"},
        "latency_ms": 123.4,
        "nested": {"x": {"y": 2.5}, "z": "text"},
    }
    (report_dir / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = parse_bye_report(report_dir)
    assert metrics["status"] == "ok"
    numeric = metrics.get("numeric_metrics", {})
    assert "latency_ms" in numeric
    assert "summary.score" in numeric
    assert "summary.count" in numeric

    out_dir = tmp_path / "out"
    json_path, csv_path = save_bye_metrics(metrics, out_dir)
    assert json_path.exists()
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "latency_ms" in text
    assert "summary.score" in text


def test_parse_bye_report_missing() -> None:
    metrics = parse_bye_report(Path("not_exists_report_dir"))
    assert metrics["status"] == "missing_report"
    assert isinstance(metrics.get("numeric_metrics"), dict)

