from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_object_persistence_report_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    baseline_dir = tmp_path / "baseline"
    _write_summary(
        suite_dir / "compare" / "compare_summary.json",
        {
            "suite_id": "v155_signal_uplift_real",
            "query_bank_id": "object_persistence_core_v1",
            "object_persistence_status": "improved",
            "next_action_recommendation": "promote_sam3_to_next_stage",
            "baseline": {
                "object_memory_items_total": 3,
                "object_persistence_support_rate": 0.20,
                "lost_object_query_support_rate": 0.25,
                "chain_object_grounding_support_rate": 0.20,
                "query_strength_coverage_rate": 0.30,
                "weak_query_groups_count": 2,
            },
            "uplift": {
                "object_memory_items_total": 6,
                "object_persistence_support_rate": 0.80,
                "lost_object_query_support_rate": 0.75,
                "chain_object_grounding_support_rate": 0.67,
                "query_strength_coverage_rate": 0.72,
                "weak_query_groups_count": 0,
            },
        },
    )
    _write_summary(
        baseline_dir / "compare" / "compare_summary.json",
        {
            "suite_id": "v155_signal_uplift_fake",
            "query_bank_id": "object_persistence_core_v1",
            "object_persistence_status": "improved",
            "baseline": {
                "object_persistence_support_rate": 0.20,
                "query_strength_coverage_rate": 0.42,
            },
            "uplift": {
                "object_persistence_support_rate": 0.83,
                "query_strength_coverage_rate": 0.83,
            },
        },
    )

    out_dir = tmp_path / "object_persistence_uplift"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_object_persistence_uplift.py"),
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
    assert "object_persistence_recommendation=promote_sam3_to_next_stage" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("object_persistence_status") == "improved"
    assert snapshot.get("object_memory_improved") is True
    assert snapshot.get("lost_object_support_improved") is True
    assert snapshot.get("chain_object_grounding_improved") is True
    assert snapshot.get("query_strength_improved") is True
    assert snapshot.get("should_formalize_sam3_next") is True
    table_text = (out_dir / "tables" / "table_object_persistence_uplift_summary.csv").read_text(encoding="utf-8")
    assert "object_persistence_improved" in table_text
    assert "should_formalize_sam3_next" in table_text
