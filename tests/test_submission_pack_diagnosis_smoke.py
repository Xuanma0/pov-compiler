from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_diagnosis_smoke(tmp_path: Path) -> None:
    paper_ready_dir = tmp_path / "paper_ready"
    (paper_ready_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (paper_ready_dir / "tables" / "table_budget_panel.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "figures" / "fig_budget_primary_vs_seconds_panel.png").write_bytes(b"PNG")
    (paper_ready_dir / "canonical" / "tables" / "Table_1.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "canonical" / "figures" / "Figure_2.png").write_bytes(b"PNG")
    (paper_ready_dir / "canonical" / "paper_map_resolved.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "canonical_id": "Table 1",
                        "canonical_relpath": "canonical/tables/Table_1.csv",
                    },
                    {
                        "canonical_id": "Figure 2",
                        "canonical_relpath": "canonical/figures/Figure_2.png",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    suite_dir = tmp_path / "suite"
    (suite_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (suite_dir / "compare").mkdir(parents=True, exist_ok=True)
    (suite_dir / "ledger").mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest" / "experiment_manifest.yaml").write_text("suite_id: suite_pack\n", encoding="utf-8")
    (suite_dir / "compare" / "compare_summary.json").write_text(json.dumps({"suite_id": "suite_pack"}), encoding="utf-8")
    (suite_dir / "compare" / "snapshot.json").write_text(json.dumps({"suite_id": "suite_pack"}), encoding="utf-8")
    (suite_dir / "ledger" / "results_long.csv").write_text("task\nnlq\n", encoding="utf-8")
    (suite_dir / "ledger" / "runs.jsonl").write_text(json.dumps({"task": "nlq"}) + "\n", encoding="utf-8")

    result_diagnosis_dir = tmp_path / "result_diagnosis"
    (result_diagnosis_dir / "tables").mkdir(parents=True, exist_ok=True)
    (result_diagnosis_dir / "figures").mkdir(parents=True, exist_ok=True)
    (result_diagnosis_dir / "tables" / "table_result_diagnosis.csv").write_text("task\nnlq\n", encoding="utf-8")
    (result_diagnosis_dir / "tables" / "table_result_diagnosis.md").write_text("# diagnosis\n", encoding="utf-8")
    (result_diagnosis_dir / "figures" / "fig_result_diagnosis_breakdown.png").write_bytes(b"PNG")
    (result_diagnosis_dir / "report.md").write_text("# diagnosis report\n", encoding="utf-8")
    (result_diagnosis_dir / "snapshot.json").write_text(json.dumps({"diagnosis_recommendations": ["inspect diagnosis"]}), encoding="utf-8")

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
            "--result-diagnosis-dir",
            str(result_diagnosis_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "result_diagnosis" / "tables" / "table_result_diagnosis.csv").exists()
    assert (out_dir / "result_diagnosis" / "report.md").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Diagnosis First" in readme_text
    assert "result_diagnosis/report.md" in readme_text
