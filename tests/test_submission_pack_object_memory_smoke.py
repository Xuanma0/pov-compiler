from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_object_memory_smoke(tmp_path: Path) -> None:
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
    (paper_ready_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    object_memory_dir = tmp_path / "object_memory_uplift"
    (object_memory_dir / "tables").mkdir(parents=True, exist_ok=True)
    (object_memory_dir / "figures").mkdir(parents=True, exist_ok=True)
    (object_memory_dir / "tables" / "table_object_memory_uplift_summary.csv").write_text(
        "object_memory_logic_status\nimproved\n",
        encoding="utf-8",
    )
    (object_memory_dir / "report.md").write_text("# Object Memory Uplift\n", encoding="utf-8")
    (object_memory_dir / "snapshot.json").write_text(
        json.dumps(
            {
                "object_memory_logic_status": "improved",
                "next_action_recommendation": "promote_object_memory_logic",
            }
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
            "--signal-uplift-dir",
            str(object_memory_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "signal_uplift" / "snapshot.json").exists()
    assert "Object Memory" in (out_dir / "signal_uplift" / "report.md").read_text(encoding="utf-8")
