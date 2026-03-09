from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_mainline_panels_smoke(tmp_path: Path) -> None:
    parent = tmp_path / "export_root"
    paper_ready_dir = parent / "paper_ready"
    (paper_ready_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "persistent_memory_main_compare").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "persistent_memory_main_decision").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "persistent_memory_main_compare" / "compare_summary.json").write_text(
        json.dumps({"persistent_memory_main_status": "improved"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (paper_ready_dir / "persistent_memory_main_decision" / "report.md").write_text(
        "# decision\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "report.md").write_text(
        "# report\n\n## Persistent Memory Main Compare\n\n- compare\n\n## Persistent Memory Main Decision\n\n- decision\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    compare_dir = tmp_path / "compare"
    (compare_dir / "tables").mkdir(parents=True, exist_ok=True)
    (compare_dir / "tables" / "table_persistent_memory_main_compare.csv").write_text("x\n1\n", encoding="utf-8")
    (compare_dir / "compare_summary.json").write_text(
        json.dumps({"persistent_memory_main_status": "improved"}, ensure_ascii=False),
        encoding="utf-8",
    )

    decision_dir = tmp_path / "decision"
    (decision_dir / "tables").mkdir(parents=True, exist_ok=True)
    (decision_dir / "tables" / "table_persistent_memory_main_decision.csv").write_text(
        "promotion_decision\npromote_persistent_memory_to_mainline\n",
        encoding="utf-8",
    )
    (decision_dir / "report.md").write_text("# decision\n", encoding="utf-8")
    (decision_dir / "snapshot.json").write_text(
        json.dumps({"promotion_decision_summary": {"promotion_decision": "promote_persistent_memory_to_mainline"}}),
        encoding="utf-8",
    )

    cleanup_dir = tmp_path / "cleanup"
    (cleanup_dir / "tables").mkdir(parents=True, exist_ok=True)
    (cleanup_dir / "tables" / "table_mainline_admission_cleanup.csv").write_text("status\nexplained\n", encoding="utf-8")
    (cleanup_dir / "report.md").write_text("# cleanup\n", encoding="utf-8")
    (cleanup_dir / "snapshot.json").write_text(
        json.dumps({"mainline_admission_cleanup_summary": {"mainline_admission_cleanup_status": "explained"}}),
        encoding="utf-8",
    )

    sample_dir = tmp_path / "sample"
    (sample_dir / "tables").mkdir(parents=True, exist_ok=True)
    (sample_dir / "tables" / "table_sample_contract_summary.csv").write_text("status\nborderline\n", encoding="utf-8")
    (sample_dir / "report.md").write_text("# sample\n", encoding="utf-8")
    (sample_dir / "snapshot.json").write_text(
        json.dumps({"sample_contract_summary": {"sample_contract_status": "borderline"}}),
        encoding="utf-8",
    )

    out_dir = parent / "submission_pack"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_submission_pack.py"),
            "--out-dir",
            str(out_dir),
            "--compare-dir",
            str(compare_dir),
            "--persistent-memory-main-decision-dir",
            str(decision_dir),
            "--mainline-admission-cleanup-dir",
            str(cleanup_dir),
            "--sample-contract-dir",
            str(sample_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "paper_ready" / "mainline_admission_cleanup" / "report.md").exists()
    assert (out_dir / "paper_ready" / "sample_contract" / "report.md").exists()
    assert (out_dir / "mainline_admission_cleanup" / "snapshot.json").exists()
    assert (out_dir / "sample_contract" / "snapshot.json").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "persistent_memory_main_compare/" in readme_text or "persistent memory main compare" in readme_text.lower()
    assert "persistent_memory_main_decision/" in readme_text or "persistent memory main decision" in readme_text.lower()
    assert "mainline_admission_cleanup/" in readme_text or "mainline admission cleanup" in readme_text.lower()
    assert "sample_contract/" in readme_text or "sample contract" in readme_text.lower()
