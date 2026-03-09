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
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_compare_bye_report_metrics_smoke(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    uid = "u001"

    _write_csv(
        run_a / "summary.csv",
        [{"video_uid": uid, "bye_status": "ok", "bye_report_metrics_path": f"bye/{uid}/bye_report_metrics.json"}],
    )
    _write_csv(
        run_b / "summary.csv",
        [{"video_uid": uid, "bye_status": "ok", "bye_report_metrics_path": f"bye/{uid}/bye_report_metrics.json"}],
    )

    (run_a / "bye" / uid).mkdir(parents=True, exist_ok=True)
    (run_b / "bye" / uid).mkdir(parents=True, exist_ok=True)
    (run_a / "bye" / uid / "bye_report_metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "bye_primary_score": 0.7,
                "bye_critical_fn": 0.2,
                "bye_latency_p50_ms": 12.0,
                "bye_latency_p95_ms": 24.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_b / "bye" / uid / "bye_report_metrics.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "bye_primary_score": 0.8,
                "bye_critical_fn": 0.1,
                "bye_latency_p50_ms": 13.0,
                "bye_latency_p95_ms": 23.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "compare_bye_report_metrics.py"),
        "--a-dir",
        str(run_a),
        "--b-dir",
        str(run_b),
        "--a-label",
        "stub",
        "--b-label",
        "real",
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "tables" / "table_bye_report_compare.csv").exists()
    assert (out_dir / "tables" / "table_bye_report_compare.md").exists()
    assert (out_dir / "figures" / "fig_bye_critical_fn_delta.png").exists()
    assert (out_dir / "figures" / "fig_bye_critical_fn_delta.pdf").exists()
    assert (out_dir / "figures" / "fig_bye_latency_delta.png").exists()
    assert (out_dir / "figures" / "fig_bye_latency_delta.pdf").exists()
    summary = json.loads((out_dir / "compare_summary.json").read_text(encoding="utf-8"))
    assert summary.get("uids_with_both") == 1
