from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_persistent_object_memory_report_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    baseline_dir = tmp_path / "baseline"
    _write_summary(
        suite_dir / "compare" / "compare_summary.json",
        {
            "suite_id": "v157_object_memory_real",
            "query_bank_id": "persistent_object_memory_core_v1",
            "persistent_object_memory_status": "improved",
            "next_action_recommendation": "promote_persistent_object_memory",
            "baseline": {
                "object_memory_items_total": 7,
                "object_memory_short_term_items_total": 5,
                "object_memory_long_term_items_total": 1,
                "reappearance_support_rate": 0.25,
                "lost_object_query_support_rate": 0.50,
                "chain_object_grounding_support_rate": 0.50,
                "query_strength_coverage_rate": 0.50,
                "weak_query_groups_count": 2,
            },
            "uplift": {
                "object_memory_items_total": 8,
                "object_memory_short_term_items_total": 4,
                "object_memory_long_term_items_total": 3,
                "reappearance_support_rate": 0.75,
                "lost_object_query_support_rate": 0.75,
                "chain_object_grounding_support_rate": 0.83,
                "query_strength_coverage_rate": 0.88,
                "weak_query_groups_count": 0,
            },
        },
    )
    _write_summary(
        baseline_dir / "compare" / "compare_summary.json",
        {
            "suite_id": "v157_object_memory_fake",
            "query_bank_id": "persistent_object_memory_core_v1",
            "persistent_object_memory_status": "improved",
            "baseline": {"object_memory_long_term_items_total": 1},
            "uplift": {"object_memory_long_term_items_total": 2},
        },
    )

    out_dir = tmp_path / "persistent_object_memory_uplift"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_persistent_object_memory_uplift.py"),
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
    assert "persistent_object_memory_recommendation=promote_persistent_object_memory" in proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("persistent_object_memory_status") == "improved"
    assert snapshot.get("long_term_memory_improved") is True
    assert snapshot.get("reappearance_support_improved") is True
    assert snapshot.get("lost_object_support_improved") is True
    assert snapshot.get("chain_object_grounding_improved") is True
    assert snapshot.get("query_strength_improved") is True
    table_text = (out_dir / "tables" / "table_persistent_object_memory_uplift_summary.csv").read_text(encoding="utf-8")
    assert "long_term_memory_improved" in table_text
    assert "should_formalize_persistent_object_memory" in table_text
