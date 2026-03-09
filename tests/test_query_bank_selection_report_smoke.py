from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_fake_suite(out_dir: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_signal_uplift_pilot.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.53_query_bank_fake.yaml"),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_query_bank_selection_report_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    _run_fake_suite(suite_dir)
    out_dir = tmp_path / "selection"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_query_bank_selection.py"),
            "--suite-dir",
            str(suite_dir),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    rows = list(csv.DictReader((out_dir / "tables" / "table_query_bank_selection.csv").open("r", encoding="utf-8", newline="")))
    assert rows
    decisions = {row["decision"] for row in rows}
    assert "candidate" in decisions
    assert "analysis_only" in decisions
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("selection_summary", {}).get("candidate_count") == 6
    assert snapshot.get("selection_summary", {}).get("analysis_only_count") == 2
