from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_signal_uplift_report_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    baseline_dir = tmp_path / "baseline"
    _write_summary(
        suite_dir / "compare" / "compare_summary.json",
        {
            "suite_id": "v152_signal_uplift_real",
            "signal_uplift_status": "improved",
            "next_action_recommendation": "keep_yolo26n_and_scale_real",
            "baseline": {
                "object_detections_total": 24,
                "object_vocab_size": 2,
                "object_memory_items_total": 1,
                "lost_object_query_support_rate": 0.25,
                "query_strength_coverage_rate": 0.40,
                "weak_query_groups_count": 3,
            },
            "uplift": {
                "object_detections_total": 68,
                "object_vocab_size": 7,
                "object_memory_items_total": 4,
                "lost_object_query_support_rate": 0.75,
                "query_strength_coverage_rate": 0.78,
                "weak_query_groups_count": 1,
            },
        },
    )
    _write_summary(
        baseline_dir / "compare" / "compare_summary.json",
        {
            "suite_id": "v152_signal_uplift_fake",
            "signal_uplift_status": "no_change",
            "baseline": {"query_strength_coverage_rate": 0.32},
            "uplift": {"query_strength_coverage_rate": 0.51},
        },
    )

    out_dir = tmp_path / "signal_uplift"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_signal_uplift.py"),
            "--suite_dir",
            str(suite_dir),
            "--baseline_dir",
            str(baseline_dir),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "signal_uplift_recommendation=keep_yolo26n_and_scale_real" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("signal_uplift_status") == "improved"
    assert snapshot.get("object_memory_improved") is True
    assert snapshot.get("next_action_recommendation") == "keep_yolo26n_and_scale_real"
    assert snapshot.get("should_try_sam3_next") is False
    table_text = (out_dir / "tables" / "table_signal_uplift_summary.csv").read_text(encoding="utf-8")
    assert "object_coverage_improved" in table_text
    assert "baseline_vs_fake_query_strength_gap" in table_text
