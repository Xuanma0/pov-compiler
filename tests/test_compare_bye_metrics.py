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
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_compare_bye_metrics_outputs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    uid = "u001"
    metrics_rel = f"bye/{uid}/bye_metrics.csv"

    _write_csv(
        run_a / "summary.csv",
        [{"video_uid": uid, "bye_status": "ok", "bye_metrics_path": metrics_rel}],
    )
    _write_csv(
        run_b / "summary.csv",
        [{"video_uid": uid, "bye_status": "ok", "bye_metrics_path": metrics_rel}],
    )
    _write_csv(
        run_a / metrics_rel,
        [{"status": "ok", "report_path": "r.json", "summary_keys": "a", "acc": 0.6, "latency_ms": 10.0}],
    )
    _write_csv(
        run_b / metrics_rel,
        [{"status": "ok", "report_path": "r.json", "summary_keys": "a", "acc": 0.8, "latency_ms": 13.0}],
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "compare_bye_metrics.py"),
        "--run_a",
        str(run_a),
        "--run_b",
        str(run_b),
        "--out_dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert (out_dir / "table_bye_compare.csv").exists()
    assert (out_dir / "table_bye_compare.md").exists()

    text = (out_dir / "table_bye_compare.md").read_text(encoding="utf-8")
    assert "delta_acc" in text
    assert "0.200000" in text

