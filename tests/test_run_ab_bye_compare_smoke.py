from __future__ import annotations

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
                "(out/'report.json').write_text(json.dumps({'acc':0.75,'latency_ms':11.0}), encoding='utf-8')",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pipeline_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_perception_complete(run_dir: Path, uid: str) -> None:
    pdir = run_dir / "perception" / uid
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "perception.json").write_text(json.dumps({"video_id": uid}), encoding="utf-8")
    (pdir / "events_v0.json").write_text(json.dumps({"video_id": uid, "events_v0": []}), encoding="utf-8")
    (pdir / "report.md").write_text("# ok", encoding="utf-8")


def test_run_ab_bye_compare_minimal(tmp_path: Path) -> None:
    root = tmp_path / "ego_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "demo.mp4").write_bytes(b"0")
    uid = hashlib.md5("demo.mp4".encode("utf-8")).hexdigest()
    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(uid + "\n", encoding="utf-8")

    out_dir = tmp_path / "ab_out"
    for run_name in ("run_stub", "run_real"):
        run_dir = out_dir / run_name
        _write_pipeline_json(run_dir / "json" / f"{uid}_v03_decisions.json", uid)
        (run_dir / "cache" / f"{uid}.index.npz").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "cache" / f"{uid}.index.npz").write_bytes(b"NPZ")
        (run_dir / "cache" / f"{uid}.index_meta.json").write_text("{}", encoding="utf-8")
        _write_perception_complete(run_dir, uid)

    fake_bye = tmp_path / "fake_bye"
    _write_fake_tool(fake_bye / "Gateway" / "scripts" / "lint_run_package.py")
    _write_fake_tool(fake_bye / "report_run.py", writes_report=True)
    _write_fake_tool(fake_bye / "run_regression_suite.py")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ab_bye_compare.py"),
        "--root",
        str(root),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--jobs",
        "1",
        "--with-bye",
        "--with-bye-budget-sweep",
        "--bye-budgets",
        "20/50/4,40/100/8",
        "--bye-root",
        str(fake_bye),
        "--bye-skip-regression",
        "--with-perception",
        "--stub-perception-backend",
        "stub",
        "--real-perception-backend",
        "stub",
        "--min-size-bytes",
        "0",
        "--probe-candidates",
        "0",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    compare_dir = out_dir / "compare"
    assert (compare_dir / "bye" / "table_bye_compare.csv").exists()
    assert (compare_dir / "bye" / "table_bye_compare.md").exists()
    assert (compare_dir / "bye" / "compare_summary.json").exists()
    assert (compare_dir / "bye_budget" / "stub" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "bye_budget" / "real" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "commands.sh").exists()
    assert (compare_dir / "README.md").exists()
