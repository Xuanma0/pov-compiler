from __future__ import annotations

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
                "(out/'report.json').write_text(json.dumps({'qualityScore':0.82,'criticalFN':1.0}), encoding='utf-8')",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pov_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events_v1": [{"id": "ev1", "t0": 0.0, "t1": 5.0, "label": "idle", "scores": {"boundary_conf": 0.7}}],
        "highlights": [{"id": "hl1", "t0": 0.2, "t1": 1.8, "conf": 0.8, "source_event": "ev1", "anchor_type": "turn_head"}],
        "decision_points": [{"id": "dp1", "t0": 0.3, "t1": 0.9, "t": 0.6, "conf": 0.7, "source_event": "ev1"}],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": [{"id": "tok1", "t0": 0.4, "t1": 0.8, "type": "ATTENTION_TURN_HEAD", "conf": 0.7, "source_event": "ev1"}]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_sweep_bye_budgets_smoke(tmp_path: Path) -> None:
    pov_dir = tmp_path / "json"
    uid = "demo_uid"
    _write_pov_json(pov_dir / f"{uid}_v03_decisions.json", uid)
    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(uid + "\n", encoding="utf-8")

    fake_bye = tmp_path / "fake_bye"
    _write_fake_tool(fake_bye / "Gateway" / "scripts" / "lint_run_package.py")
    _write_fake_tool(fake_bye / "report_run.py", writes_report=True)
    _write_fake_tool(fake_bye / "run_regression_suite.py")

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_bye_budgets.py"),
        "--pov-json-dir",
        str(pov_dir),
        "--uids-file",
        str(uids_file),
        "--out-dir",
        str(out_dir),
        "--bye-root",
        str(fake_bye),
        "--budgets",
        "20/50/4,40/100/8",
        "--primary-metric",
        "qualityScore",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "selection_mode=uids_file" in result.stdout
    assert "uids_missing_count=0" in result.stdout
    assert "fallback" not in result.stdout.lower()

    assert (out_dir / "aggregate" / "metrics_by_budget.csv").exists()
    assert (out_dir / "aggregate" / "metrics_by_budget.md").exists()
    assert (out_dir / "snapshot.json").exists()
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    selection = snapshot.get("selection", {})
    assert "selection_mode" in selection
    assert "uids_file_path" in selection
    assert "uids_requested" in selection
    assert "uids_found" in selection
    assert "uids_missing_count" in selection
    assert "uids_missing_sample" in selection
    assert "dir_uids_sample" in selection
    figures = list((out_dir / "figures").glob("fig_bye_quality_vs_budget_seconds.*"))
    assert figures
    critical_figs = list((out_dir / "figures").glob("fig_bye_critical_fn_vs_budget_seconds.*"))
    assert critical_figs
