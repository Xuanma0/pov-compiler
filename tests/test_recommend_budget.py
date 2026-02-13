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


def test_recommend_budget_outputs_and_top1(tmp_path: Path) -> None:
    bye_csv = tmp_path / "bye" / "aggregate" / "metrics_by_budget.csv"
    nlq_csv = tmp_path / "nlq" / "aggregate" / "metrics_by_budget.csv"
    _write_csv(
        bye_csv,
        [
            {"budget_max_total_s": 20, "budget_max_tokens": 50, "budget_max_decisions": 4, "qualityScore": 0.60},
            {"budget_max_total_s": 40, "budget_max_tokens": 100, "budget_max_decisions": 8, "qualityScore": 0.75},
            {"budget_max_total_s": 60, "budget_max_tokens": 200, "budget_max_decisions": 12, "qualityScore": 0.80},
        ],
    )
    _write_csv(
        nlq_csv,
        [
            {"budget_max_total_s": 20, "budget_max_tokens": 50, "budget_max_decisions": 4, "nlq_full_hit_at_k_strict": 0.55, "nlq_full_fp_rate": 0.20},
            {"budget_max_total_s": 40, "budget_max_tokens": 100, "budget_max_decisions": 8, "nlq_full_hit_at_k_strict": 0.70, "nlq_full_fp_rate": 0.15},
            {"budget_max_total_s": 60, "budget_max_tokens": 200, "budget_max_decisions": 12, "nlq_full_hit_at_k_strict": 0.72, "nlq_full_fp_rate": 0.25},
        ],
    )
    out_dir = tmp_path / "reco"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "recommend_budget.py"),
        "--bye_csv",
        str(bye_csv),
        "--nlq_csv",
        str(nlq_csv),
        "--out_dir",
        str(out_dir),
        "--weights-json",
        json.dumps(
            {
                "bye_qualityScore": 1.0,
                "nlq_full_hit_at_k_strict": 1.0,
                "nlq_full_fp_rate": -0.5,
                "budget_seconds": -0.01,
            }
        ),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    assert (out_dir / "tables" / "table_budget_recommend.csv").exists()
    assert (out_dir / "tables" / "table_budget_recommend.md").exists()
    assert (out_dir / "recommend_summary.json").exists()
    assert (out_dir / "figures" / "fig_objective_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_pareto_frontier.png").exists()

    summary = json.loads((out_dir / "recommend_summary.json").read_text(encoding="utf-8"))
    assert int(summary["budgets_joined"]) == 3
    assert str(summary["top1_budget_key"]) == "40/100/8"

