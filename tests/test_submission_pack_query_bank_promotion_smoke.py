from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_query_bank_promotion_smoke(tmp_path: Path) -> None:
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
    (paper_ready_dir / "query_bank_compare").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "query_bank_promotion_decision").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "query_bank_compare" / "README.md").write_text("# compare\n", encoding="utf-8")
    (paper_ready_dir / "query_bank_promotion_decision" / "report.md").write_text("# promotion\n", encoding="utf-8")
    (paper_ready_dir / "report.md").write_text(
        "# report\n\n## Query Bank Compare\n\n- aligned\n\n## Query Bank Promotion Decision\n\n- expand sample first\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    out_dir = tmp_path / "submission_pack"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_submission_pack.py"),
            "--paper-ready-dir",
            str(paper_ready_dir),
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    report_text = (out_dir / "paper_ready" / "report.md").read_text(encoding="utf-8")
    assert "## Query Bank Compare" in report_text
    assert "## Query Bank Promotion Decision" in report_text
