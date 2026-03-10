from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_mainline_closure_smoke(tmp_path: Path) -> None:
    parent = tmp_path / "export_root"
    paper_ready_dir = parent / "paper_ready"
    (paper_ready_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "figures").mkdir(parents=True, exist_ok=True)
    for panel in (
        "persistent_memory_main_compare",
        "persistent_memory_main_decision",
        "mainline_admission_cleanup",
        "harder_sample_contract",
        "mainline_admission_closure",
    ):
        (paper_ready_dir / panel).mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "persistent_memory_main_compare" / "compare_summary.json").write_text(
        json.dumps({"persistent_memory_main_status": "improved"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (paper_ready_dir / "persistent_memory_main_decision" / "report.md").write_text(
        "# decision\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "mainline_admission_cleanup" / "report.md").write_text(
        "# cleanup\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "harder_sample_contract" / "report.md").write_text(
        "# harder sample\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "mainline_admission_closure" / "report.md").write_text(
        "# closure\n",
        encoding="utf-8",
    )
    (paper_ready_dir / "report.md").write_text(
        "\n".join(
            [
                "# report",
                "",
                "## Persistent Memory Main Compare",
                "",
                "## Persistent Memory Main Decision",
                "",
                "## Mainline Admission Cleanup",
                "",
                "## Harder Sample Contract",
                "",
                "## Mainline Admission Closure",
                "",
                "## Mainline Reading Order",
            ]
        ),
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

    harder_dir = tmp_path / "harder"
    (harder_dir / "tables").mkdir(parents=True, exist_ok=True)
    (harder_dir / "tables" / "table_harder_sample_contract_summary.csv").write_text("status\nadequate\n", encoding="utf-8")
    (harder_dir / "report.md").write_text("# harder sample\n", encoding="utf-8")
    (harder_dir / "snapshot.json").write_text(
        json.dumps({"harder_sample_contract_summary": {"sample_contract_status": "adequate"}}),
        encoding="utf-8",
    )

    closure_dir = tmp_path / "closure"
    (closure_dir / "tables").mkdir(parents=True, exist_ok=True)
    (closure_dir / "tables" / "table_mainline_admission_closure.csv").write_text("status\nclosed\n", encoding="utf-8")
    (closure_dir / "report.md").write_text("# closure\n", encoding="utf-8")
    (closure_dir / "snapshot.json").write_text(
        json.dumps({"mainline_admission_closure_summary": {"mainline_admission_closure_status": "closed"}}),
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
            "--harder-sample-contract-dir",
            str(harder_dir),
            "--mainline-admission-closure-dir",
            str(closure_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "paper_ready" / "mainline_admission_cleanup" / "report.md").exists()
    assert (out_dir / "paper_ready" / "harder_sample_contract" / "report.md").exists()
    assert (out_dir / "paper_ready" / "mainline_admission_closure" / "report.md").exists()
    assert (out_dir / "mainline_admission_cleanup" / "snapshot.json").exists()
    assert (out_dir / "harder_sample_contract" / "snapshot.json").exists()
    assert (out_dir / "mainline_admission_closure" / "snapshot.json").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "persistent_memory_main_compare/" in readme_text or "persistent memory main compare" in readme_text.lower()
    assert "persistent_memory_main_decision/" in readme_text or "persistent memory main decision" in readme_text.lower()
    assert "mainline_admission_cleanup/" in readme_text or "mainline admission cleanup" in readme_text.lower()
    assert "harder_sample_contract/" in readme_text or "harder sample contract" in readme_text.lower()
    assert "mainline_admission_closure/" in readme_text or "mainline admission closure" in readme_text.lower()
    assert "None" not in readme_text
    assert "not_provided" in readme_text or "unavailable" in readme_text
