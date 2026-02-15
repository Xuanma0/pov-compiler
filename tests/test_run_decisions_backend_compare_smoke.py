from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for k in row:
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_run_decisions_backend_compare_smoke(tmp_path: Path) -> None:
    run_a = tmp_path / "run_A"
    run_b = tmp_path / "run_B"
    uid = "u_demo"

    _write_csv(
        run_a / "summary.csv",
        [
            {
                "video_uid": uid,
                "status": "ok",
                "decisions_backend": "heuristic",
                "decision_pool_kind": "decisions_v1",
                "critical_fn_rate": 0.30,
                "latency_p95_ms": 10.0,
            }
        ],
    )
    _write_csv(
        run_b / "summary.csv",
        [
            {
                "video_uid": uid,
                "status": "ok",
                "decisions_backend": "model",
                "decision_pool_kind": "decisions_model_v1",
                "critical_fn_rate": 0.22,
                "latency_p95_ms": 11.2,
            }
        ],
    )

    _write_csv(
        run_a / "nlq_summary_all.csv",
        [
            {
                "video_uid": uid,
                "variant": "full",
                "query_type": "hard_pseudo_nlq",
                "mrr": 0.35,
                "top1_in_distractor_rate": 0.40,
            }
        ],
    )
    _write_csv(
        run_b / "nlq_summary_all.csv",
        [
            {
                "video_uid": uid,
                "variant": "full",
                "query_type": "hard_pseudo_nlq",
                "mrr": 0.55,
                "top1_in_distractor_rate": 0.28,
            }
        ],
    )

    out_dir = tmp_path / "decisions_cmp"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_decisions_backend_compare.py"),
        "--run-a-dir",
        str(run_a),
        "--run-b-dir",
        str(run_b),
        "--out_dir",
        str(out_dir),
        "--a-label",
        "stub",
        "--b-label",
        "real",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_compare=" in proc.stdout

    compare_dir = out_dir
    assert (compare_dir / "tables" / "table_decisions_backend_compare.csv").exists()
    assert (compare_dir / "tables" / "table_decisions_backend_compare.md").exists()
    assert (compare_dir / "figures" / "fig_decisions_backend_delta.png").exists()
    assert (compare_dir / "figures" / "fig_decisions_backend_tradeoff.png").exists()
    assert (compare_dir / "compare_summary.json").exists()
    assert (compare_dir / "snapshot.json").exists()

    with (compare_dir / "tables" / "table_decisions_backend_compare.csv").open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    assert "delta_mrr_strict" in header
    assert "delta_distractor_rate" in header
    assert "delta_critical_fn_rate" in header
