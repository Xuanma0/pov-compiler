from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_small_pilot_live_plan(tmp_path: Path) -> None:
    out_dir = tmp_path / "v149_main_real_dry"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_main_real_benchmark.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.49_main_real_pilot.yaml"),
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
    assert "saved_provider_normalization=skipped" in proc.stdout
    assert "saved_query_promotion_pack=skipped" in proc.stdout
    payload = json.loads((out_dir / "manifest" / "dry_collect_snapshot.json").read_text(encoding="utf-8"))
    assert payload.get("suite_id") == "v1.49_main_real_pilot"
    assert payload.get("provider_normalization_enabled") is True
    assert payload.get("query_promotion_enabled") is True
    assert payload.get("require_real_calls") is True
    telemetry = payload.get("telemetry", {})
    assert telemetry.get("variants", {}).get("real", {}).get("provider") == "openai"

