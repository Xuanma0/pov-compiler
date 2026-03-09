from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repeated_real_pilot_plan(tmp_path: Path) -> None:
    out_dir = tmp_path / "v151_main_real_repeat"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_main_real_benchmark.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.51_main_real_repeat.yaml"),
            "--mode",
            "pilot",
            "--dry-collect",
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_repeatability_audit=skipped" in proc.stdout
    assert "saved_sample_size_recommendation=skipped" in proc.stdout
    assert "saved_query_uplift_candidates=skipped" in proc.stdout
    snapshot = json.loads((out_dir / "manifest" / "dry_collect_snapshot.json").read_text(encoding="utf-8"))
    repeat_payload = snapshot.get("repeat", {})
    assert repeat_payload.get("enabled") is True
    assert repeat_payload.get("repeat_count") == 3
    assert str(repeat_payload.get("repeat_profile", "")).endswith("configs/repeatability/repeat_profile_v1.yaml")
