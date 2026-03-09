from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_persistent_memory_main_smoke(tmp_path: Path) -> None:
    paper_ready_dir = tmp_path / "paper_ready"
    (paper_ready_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "paper_map_resolved.json").write_text(
        json.dumps({"rows": [{"canonical_id": "Table 1", "canonical_relpath": "canonical/tables/Table_1.csv"}]}),
        encoding="utf-8",
    )
    (paper_ready_dir / "tables" / "table_budget_panel.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "figures" / "fig_budget_primary_vs_seconds_panel.png").write_bytes(b"PNG")
    (paper_ready_dir / "canonical" / "tables" / "Table_1.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "persistent_memory_main_compare").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "persistent_memory_main_decision").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "persistent_memory_main_compare" / "README.md").write_text(
        "# persistent memory main compare\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "persistent_memory_main_compare" / "compare_summary.json").write_text(
        json.dumps({"persistent_memory_main_status": "improved"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (paper_ready_dir / "persistent_memory_main_decision" / "report.md").write_text(
        "# persistent memory main decision\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "persistent_memory_main_decision" / "snapshot.json").write_text(
        json.dumps(
            {"promotion_decision_summary": {"promotion_decision": "promote_persistent_memory_to_mainline"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (paper_ready_dir / "report.md").write_text(
        "# report\n\n## Persistent Memory Main Compare\n\n- aligned compare\n\n## Persistent Memory Main Decision\n\n- promote to mainline\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    persistent_compare_dir = tmp_path / "persistent_memory_main_compare"
    (persistent_compare_dir / "tables").mkdir(parents=True, exist_ok=True)
    (persistent_compare_dir / "figures").mkdir(parents=True, exist_ok=True)
    (persistent_compare_dir / "tables" / "table_persistent_memory_main_compare.csv").write_text(
        "persistent_memory_main_status\nimproved\n",
        encoding="utf-8",
    )
    (persistent_compare_dir / "compare_summary.json").write_text(
        json.dumps({"persistent_memory_main_status": "improved"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (persistent_compare_dir / "README.md").write_text("# compare\n", encoding="utf-8")

    persistent_decision_dir = tmp_path / "persistent_memory_main_decision"
    (persistent_decision_dir / "tables").mkdir(parents=True, exist_ok=True)
    (persistent_decision_dir / "tables" / "table_persistent_memory_main_decision.csv").write_text(
        "promotion_decision\npromote_persistent_memory_to_mainline\n",
        encoding="utf-8",
    )
    (persistent_decision_dir / "report.md").write_text("# decision\n", encoding="utf-8")
    (persistent_decision_dir / "snapshot.json").write_text(
        json.dumps(
            {"promotion_decision_summary": {"promotion_decision": "promote_persistent_memory_to_mainline"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "submission_pack"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_submission_pack.py"),
            "--paper-ready-dir",
            str(paper_ready_dir),
            "--out-dir",
            str(out_dir),
            "--persistent-memory-main-compare-dir",
            str(persistent_compare_dir),
            "--persistent-memory-main-decision-dir",
            str(persistent_decision_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "persistent_memory_main_compare" / "compare_summary.json").exists()
    assert (out_dir / "persistent_memory_main_decision" / "report.md").exists()
    assert (out_dir / "paper_ready" / "persistent_memory_main_compare" / "compare_summary.json").exists()
    assert (out_dir / "paper_ready" / "persistent_memory_main_decision" / "report.md").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "persistent_memory_main_compare/" in readme_text or "persistent memory main compare" in readme_text.lower()
    assert "persistent_memory_main_decision/" in readme_text or "persistent memory main decision" in readme_text.lower()
