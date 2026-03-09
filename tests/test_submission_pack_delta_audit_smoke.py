from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_delta_audit_smoke(tmp_path: Path) -> None:
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
    (paper_ready_dir / "admission_control").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "admission_control" / "snapshot.json").write_text(
        json.dumps({"admission_status": "partial"}),
        encoding="utf-8",
    )
    (paper_ready_dir / "delta_audit").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "delta_audit" / "report.md").write_text("# delta audit\n", encoding="utf-8")
    (paper_ready_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    suite_dir = tmp_path / "suite"
    (suite_dir / "admission_control").mkdir(parents=True, exist_ok=True)
    (suite_dir / "admission_control" / "report.md").write_text("# admission\n", encoding="utf-8")
    (suite_dir / "admission_control" / "snapshot.json").write_text(
        json.dumps({"admission_status": "partial"}),
        encoding="utf-8",
    )

    delta_audit_dir = tmp_path / "delta_audit"
    (delta_audit_dir / "tables").mkdir(parents=True, exist_ok=True)
    (delta_audit_dir / "figures").mkdir(parents=True, exist_ok=True)
    (delta_audit_dir / "tables" / "table_delta_audit.csv").write_text("task\nnlq\n", encoding="utf-8")
    (delta_audit_dir / "report.md").write_text("# delta audit\n", encoding="utf-8")
    (delta_audit_dir / "snapshot.json").write_text(
        json.dumps({"main_recommendation": "strengthen_query_bank"}),
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
            "--suite-dir",
            str(suite_dir),
            "--delta-audit-dir",
            str(delta_audit_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "admission_control" / "snapshot.json").exists()
    assert (out_dir / "delta_audit" / "report.md").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Admission First" in readme_text
    assert "Delta Audit" in readme_text
    assert "Read `admission_control/` first." in readme_text
