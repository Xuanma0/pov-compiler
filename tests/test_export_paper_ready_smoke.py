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
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_export_paper_ready_smoke(tmp_path: Path) -> None:
    compare_dir = tmp_path / "compare"
    # NLQ stub/real
    _write_csv(
        compare_dir / "nlq_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "nlq_full_hit_at_k_strict": 0.4},
            {"budget_key": "40/100/8", "budget_seconds": 40, "nlq_full_hit_at_k_strict": 0.6},
        ],
    )
    _write_csv(
        compare_dir / "nlq_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "nlq_full_hit_at_k_strict": 0.45},
            {"budget_key": "40/100/8", "budget_seconds": 40, "nlq_full_hit_at_k_strict": 0.67},
        ],
    )
    # Streaming stub/real
    _write_csv(
        compare_dir / "streaming_budget" / "stub" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.5, "e2e_ms_p50": 8.0, "e2e_ms_p95": 12.0},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.62, "e2e_ms_p50": 9.0, "e2e_ms_p95": 13.0},
        ],
    )
    _write_csv(
        compare_dir / "streaming_budget" / "real" / "aggregate" / "metrics_by_budget.csv",
        [
            {"budget_key": "20/50/4", "budget_seconds": 20, "hit@k_strict": 0.55, "e2e_ms_p50": 8.5, "e2e_ms_p95": 12.5},
            {"budget_key": "40/100/8", "budget_seconds": 40, "hit@k_strict": 0.7, "e2e_ms_p50": 9.5, "e2e_ms_p95": 13.5},
        ],
    )
    # BYE intentionally missing: should be handled gracefully.
    (compare_dir / "budget_recommend" / "stub").mkdir(parents=True, exist_ok=True)
    (compare_dir / "budget_recommend" / "real").mkdir(parents=True, exist_ok=True)
    (compare_dir / "budget_recommend" / "stub" / "recommend_summary.json").write_text(
        json.dumps({"top1_budget_key": "40/100/8"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (compare_dir / "budget_recommend" / "real" / "recommend_summary.json").write_text(
        json.dumps({"top1_budget_key": "40/100/8"}, ensure_ascii=False),
        encoding="utf-8",
    )

    out_dir = tmp_path / "paper_ready"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_paper_ready.py"),
        "--compare_dir",
        str(compare_dir),
        "--out_dir",
        str(out_dir),
        "--format",
        "md+csv",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_table_panel=" in proc.stdout
    assert "saved_table_delta=" in proc.stdout

    panel_csv = out_dir / "tables" / "table_budget_panel.csv"
    panel_md = out_dir / "tables" / "table_budget_panel.md"
    delta_csv = out_dir / "tables" / "table_budget_panel_delta.csv"
    delta_md = out_dir / "tables" / "table_budget_panel_delta.md"
    report_md = out_dir / "report.md"
    snapshot_json = out_dir / "snapshot.json"
    assert panel_csv.exists()
    assert panel_md.exists()
    assert delta_csv.exists()
    assert delta_md.exists()
    assert report_md.exists()
    assert snapshot_json.exists()
    assert (out_dir / "figures" / "fig_budget_primary_vs_seconds_panel.png").exists()
    assert (out_dir / "figures" / "fig_budget_primary_delta_vs_seconds_panel.png").exists()
    assert (out_dir / "figures" / "fig_budget_latency_vs_seconds_streaming.png").exists()

    header = panel_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "task" in header
    assert "budget_seconds" in header
    assert "primary_a" in header
    assert "primary_b" in header
    assert "delta_primary" in header
