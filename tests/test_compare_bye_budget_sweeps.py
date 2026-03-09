from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_compare_bye_budget_sweeps_outputs(tmp_path: Path) -> None:
    a_csv = tmp_path / "a" / "aggregate" / "metrics_by_budget.csv"
    b_csv = tmp_path / "b" / "aggregate" / "metrics_by_budget.csv"
    rows_a = [
        {"budget_max_total_s": 20, "budget_max_tokens": 50, "budget_max_decisions": 4, "qualityScore": 0.60, "criticalFN": 3},
        {"budget_max_total_s": 40, "budget_max_tokens": 100, "budget_max_decisions": 8, "qualityScore": 0.70, "criticalFN": 2},
        {"budget_max_total_s": 60, "budget_max_tokens": 200, "budget_max_decisions": 12, "qualityScore": 0.75, "criticalFN": 2},
    ]
    rows_b = [
        {"budget_max_total_s": 20, "budget_max_tokens": 50, "budget_max_decisions": 4, "qualityScore": 0.63, "criticalFN": 2},
        {"budget_max_total_s": 40, "budget_max_tokens": 100, "budget_max_decisions": 8, "qualityScore": 0.73, "criticalFN": 1},
        {"budget_max_total_s": 60, "budget_max_tokens": 200, "budget_max_decisions": 12, "qualityScore": 0.79, "criticalFN": 1},
    ]
    _write_csv(a_csv, rows_a)
    _write_csv(b_csv, rows_b)

    out_dir = tmp_path / "compare"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "compare_bye_budget_sweeps.py"),
        "--a_csv",
        str(a_csv),
        "--b_csv",
        str(b_csv),
        "--a_label",
        "stub",
        "--b_label",
        "real",
        "--primary-metric",
        "qualityScore",
        "--out_dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    table_csv = out_dir / "tables" / "table_budget_compare.csv"
    table_md = out_dir / "tables" / "table_budget_compare.md"
    summary_json = out_dir / "compare_summary.json"
    fig1 = out_dir / "figures" / "fig_bye_primary_vs_budget_seconds_compare.png"
    fig2 = out_dir / "figures" / "fig_bye_primary_delta_vs_budget_seconds.png"
    assert table_csv.exists()
    assert table_md.exists()
    assert summary_json.exists()
    assert fig1.exists()
    assert fig2.exists()

    lines = table_csv.read_text(encoding="utf-8").splitlines()
    assert lines and "delta_qualityScore" in lines[0]

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert int(summary["budgets_matched"]) == 3
    assert str(summary["primary_metric"]) == "qualityScore"

