from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_query_audit_smoke(tmp_path: Path) -> None:
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
    (paper_ready_dir / "admission_calibration").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "admission_calibration" / "snapshot.json").write_text(
        json.dumps({"calibration_status": "partial"}),
        encoding="utf-8",
    )
    (paper_ready_dir / "query_strength_audit").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "query_strength_audit" / "report.md").write_text("# query strength\n", encoding="utf-8")
    (paper_ready_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    suite_dir = tmp_path / "suite"
    (suite_dir / "admission_control").mkdir(parents=True, exist_ok=True)
    (suite_dir / "admission_control" / "report.md").write_text("# admission\n", encoding="utf-8")
    (suite_dir / "admission_control" / "snapshot.json").write_text(
        json.dumps({"admission_status": "partial"}),
        encoding="utf-8",
    )

    admission_calibration_dir = tmp_path / "admission_calibration"
    (admission_calibration_dir / "tables").mkdir(parents=True, exist_ok=True)
    (admission_calibration_dir / "tables" / "table_admission_calibration.csv").write_text("status\npartial\n", encoding="utf-8")
    (admission_calibration_dir / "report.md").write_text("# calibration\n", encoding="utf-8")
    (admission_calibration_dir / "snapshot.json").write_text(
        json.dumps({"calibration_status": "partial"}),
        encoding="utf-8",
    )

    query_strength_dir = tmp_path / "query_strength_audit"
    (query_strength_dir / "tables").mkdir(parents=True, exist_ok=True)
    (query_strength_dir / "figures").mkdir(parents=True, exist_ok=True)
    (query_strength_dir / "tables" / "table_query_strength_audit.csv").write_text("query_group\nchain\n", encoding="utf-8")
    (query_strength_dir / "report.md").write_text("# query strength\n", encoding="utf-8")
    (query_strength_dir / "snapshot.json").write_text(
        json.dumps({"main_recommendation": "promote_to_core_query_bank"}),
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
            "--admission-calibration-dir",
            str(admission_calibration_dir),
            "--query-strength-audit-dir",
            str(query_strength_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "admission_calibration" / "snapshot.json").exists()
    assert (out_dir / "query_strength_audit" / "report.md").exists()
    assert (out_dir / "paper_ready" / "admission_calibration" / "snapshot.json").exists()
    assert (out_dir / "paper_ready" / "query_strength_audit" / "report.md").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Calibration First" in readme_text
    assert "Query Strength Audit" in readme_text
