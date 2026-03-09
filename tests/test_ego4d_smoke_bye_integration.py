from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_fake_tool(path: Path, *, writes_report: bool = False) -> None:
    lines = [
        "import argparse",
        "import json",
        "from pathlib import Path",
        "p=argparse.ArgumentParser()",
        "p.add_argument('--run-package','--run_package','--package',dest='run_package',default='')",
        "p.add_argument('--out-dir','--out_dir','--out',dest='out_dir',default='')",
        "p.add_argument('pos', nargs='*')",
        "a=p.parse_args()",
        "rp=a.run_package or (a.pos[0] if a.pos else '')",
        "od=a.out_dir or (a.pos[1] if len(a.pos)>1 else rp)",
        "print('OK', Path(__file__).name, rp, od)",
    ]
    if writes_report:
        lines.extend(
            [
                "out=Path(od)",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out/'report.json').write_text(json.dumps({'acc':0.7,'latency_ms':12.5}), encoding='utf-8')",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ego4d_smoke_with_bye_hook(tmp_path: Path) -> None:
    root = tmp_path / "ego_root"
    root.mkdir(parents=True, exist_ok=True)
    video = root / "demo.mp4"
    video.write_bytes(b"\x00\x00\x00\x00")

    uid = hashlib.md5("demo.mp4".encode("utf-8")).hexdigest()
    out_dir = tmp_path / "smoke_out"
    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    pipeline_json = {
        "video_id": uid,
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "highlights": [{"id": "hl_0001", "t0": 0.1, "t1": 0.8, "source_event": "event_0001", "anchor_type": "turn_head", "anchor_t": 0.2, "conf": 0.8}],
        "decision_points": [{"id": "dp_0001", "t": 0.2, "t0": 0.1, "t1": 0.8, "source_event": "event_0001", "trigger": {}, "state": {}, "action": {}, "constraints": [], "outcome": {}, "alternatives": [], "conf": 0.7}],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    (out_dir / "json" / f"{uid}_v03_decisions.json").write_text(json.dumps(pipeline_json), encoding="utf-8")
    (out_dir / "cache" / f"{uid}.index.npz").write_bytes(b"NPZ")
    (out_dir / "cache" / f"{uid}.index_meta.json").write_text("{}", encoding="utf-8")

    fake_bye = tmp_path / "fake_bye"
    _write_fake_tool(fake_bye / "Gateway" / "scripts" / "lint_run_package.py")
    _write_fake_tool(fake_bye / "report_run.py", writes_report=True)
    _write_fake_tool(fake_bye / "run_regression_suite.py")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ego4d_smoke.py"),
        "--root",
        str(root),
        "--out_dir",
        str(out_dir),
        "--n",
        "1",
        "--seed",
        "0",
        "--min-size-bytes",
        "0",
        "--probe-candidates",
        "0",
        "--no-proxy",
        "--no-run-eval",
        "--no-run-nlq",
        "--no-run-perception",
        "--run-bye",
        "--bye-root",
        str(fake_bye),
        "--bye-skip-regression",
        "--jobs",
        "1",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    bye_uid_dir = out_dir / "bye" / uid
    assert bye_uid_dir.exists()
    assert (bye_uid_dir / "snapshot.json").exists()
    assert (bye_uid_dir / "bye_metrics.csv").exists()
    assert (bye_uid_dir / "bye_report_metrics.json").exists()

    summary_csv = out_dir / "summary.csv"
    assert summary_csv.exists()
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    row = rows[0]
    assert "bye_status" in row
    assert "bye_report_rc" in row
    assert "bye_metrics_path" in row
    assert "bye_primary_score" in row
    assert "bye_critical_fn" in row
    assert "bye_latency_p50_ms" in row
    assert "bye_latency_p95_ms" in row
    bye_numeric_cols = [k for k in row.keys() if k.startswith("bye_numeric_")]
    assert bye_numeric_cols
