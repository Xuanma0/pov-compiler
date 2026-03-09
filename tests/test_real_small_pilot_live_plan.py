from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_small_pilot_live_plan(tmp_path: Path) -> None:
    out_dir = tmp_path / "v150_main_real_golden"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_main_real_benchmark.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.50_main_real_golden.yaml"),
            "--mode",
            "pilot",
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_provider_reachability=" in proc.stdout
    assert "saved_golden_real_sample=" in proc.stdout
    assert "proof_status=ok" in proc.stdout
    reachability = json.loads((out_dir / "provider_reachability" / "summary.json").read_text(encoding="utf-8"))
    assert reachability.get("proof_status") == "ok"
    assert reachability.get("real_call_status") == "ok"
    golden = json.loads((out_dir / "golden_real_sample" / "snapshot.json").read_text(encoding="utf-8"))
    assert golden.get("golden_sample_status") in {"ok", "partial"}
