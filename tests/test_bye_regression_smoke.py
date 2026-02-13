from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.run_package import resolve_bye_root


def _sample_output(video_id: str = "demo_uid") -> dict[str, object]:
    return {
        "video_id": video_id,
        "events_v0": [
            {"id": "v0_event_0001", "t0": 0.0, "t1": 2.0, "scores": {"boundary_conf": 0.7}},
            {"id": "v0_event_0002", "t0": 2.0, "t1": 4.0, "scores": {"boundary_conf": 0.8}},
        ],
        "highlights": [
            {
                "id": "hl_0001",
                "t0": 0.2,
                "t1": 0.9,
                "source_event": "v0_event_0001",
                "anchor_type": "turn_head",
                "anchor_t": 0.4,
                "conf": 0.88,
            }
        ],
        "token_codec": {
            "tokens": [
                {
                    "id": "tok_0001",
                    "t0": 0.2,
                    "t1": 0.8,
                    "type": "ATTENTION_TURN_HEAD",
                    "conf": 0.8,
                    "source_event": "v0_event_0001",
                }
            ]
        },
        "decision_points": [
            {
                "id": "dp_0001",
                "t": 0.4,
                "t0": 0.2,
                "t1": 0.9,
                "source_event": "v0_event_0001",
                "source_highlight": "hl_0001",
                "conf": 0.84,
                "trigger": {"anchor_type": "turn_head"},
                "state": {"motion_state_before": "MOVING"},
                "action": {"type": "ATTENTION_TURN_HEAD"},
                "constraints": [{"type": "STABILITY_CONSTRAINT", "score": 0.2}],
                "outcome": {"type": "MOTION_INCREASE"},
            }
        ],
    }


def _write_fake_tool(path: Path, *, writes_report: bool = False) -> None:
    content = [
        "import argparse",
        "import json",
        "from pathlib import Path",
        "p = argparse.ArgumentParser()",
        "p.add_argument('--run-package', '--run_package', '--package', dest='run_package', default='')",
        "p.add_argument('--out-dir', '--out_dir', '--out', dest='out_dir', default='')",
        "p.add_argument('pos', nargs='*')",
        "a = p.parse_args()",
        "rp = a.run_package or (a.pos[0] if a.pos else '')",
        "od = a.out_dir or (a.pos[1] if len(a.pos) > 1 else '')",
        "print('FAKE_TOOL_OK', Path(__file__).name, 'rp=', rp, 'od=', od)",
    ]
    if writes_report:
        content.extend(
            [
                "out_base = Path(od) if od else Path(rp)",
                "out_base.mkdir(parents=True, exist_ok=True)",
                "(out_base / 'report.json').write_text(json.dumps({'ok': True, 'source': Path(__file__).name}), encoding='utf-8')",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def test_resolve_bye_root_and_smoke_with_fake_repo(tmp_path: Path) -> None:
    fake_bye = tmp_path / "fake_bye_repo"
    _write_fake_tool(fake_bye / "Gateway" / "scripts" / "lint_run_package.py")
    _write_fake_tool(fake_bye / "report_run.py", writes_report=True)
    _write_fake_tool(fake_bye / "run_regression_suite.py")

    old_env = os.environ.get("BYE_ROOT")
    try:
        os.environ["BYE_ROOT"] = str(fake_bye)
        resolved = resolve_bye_root(None)
        assert resolved is not None
        assert resolved.resolve() == fake_bye.resolve()

        pov_json = tmp_path / "demo_v03_decisions.json"
        pov_json.write_text(json.dumps(_sample_output(), ensure_ascii=False, indent=2), encoding="utf-8")
        out_dir = tmp_path / "bye_smoke_out"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "bye_regression_smoke.py"),
            "--pov_json",
            str(pov_json),
            "--out_dir",
            str(out_dir),
            "--bye_root",
            str(fake_bye),
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr or result.stdout

        assert (out_dir / "events" / "events_v1.jsonl").exists()
        assert (out_dir / "run_package" / "events" / "events_v1.jsonl").exists()
        assert any(p.name.startswith("lint.") for p in (out_dir / "logs").glob("*.stdout.log"))
        assert any(p.name.startswith("report.") for p in (out_dir / "logs").glob("*.stdout.log"))

        snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
        assert snapshot["video_id"] == "demo_uid"
        assert "bye" in snapshot and "steps" in snapshot["bye"]
        assert snapshot["bye"]["root"] == str(fake_bye.resolve())
        assert isinstance(snapshot["bye"]["found_entrypoints"], dict)
        assert len(snapshot["bye"]["steps"]) >= 2
        assert "outputs" in snapshot and "events_jsonl" in snapshot["outputs"]
        assert isinstance(snapshot["outputs"].get("report_files", []), list)
    finally:
        if old_env is None:
            os.environ.pop("BYE_ROOT", None)
        else:
            os.environ["BYE_ROOT"] = old_env


def test_smoke_without_bye_root_warns_and_exports(tmp_path: Path) -> None:
    pov_json = tmp_path / "demo_v03_decisions.json"
    pov_json.write_text(json.dumps(_sample_output("uid_no_bye"), ensure_ascii=False, indent=2), encoding="utf-8")
    out_dir = tmp_path / "bye_smoke_out_no_bye"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bye_regression_smoke.py"),
        "--pov_json",
        str(pov_json),
        "--out_dir",
        str(out_dir),
        "--bye_root",
        str(tmp_path / "does_not_exist"),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "WARN:" in result.stdout
    assert (out_dir / "events" / "events_v1.jsonl").exists()
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["bye"]["status"] == "missing_bye_root"

    strict_cmd = cmd + ["--strict"]
    strict_result = subprocess.run(strict_cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert strict_result.returncode != 0
