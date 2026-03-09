from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_object_memory_uplift_pilot_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "v156_object_memory_fake"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_object_memory_uplift_pilot.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.56_object_memory_fake.yaml"),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_compare=" in proc.stdout
    assert "object_memory_logic_status=improved" in proc.stdout
    assert (out_dir / "compare" / "tables" / "table_object_memory_uplift.csv").exists()
    assert (out_dir / "compare" / "figures" / "fig_object_memory_uplift_delta.png").exists()
    assert (out_dir / "compare" / "README.md").exists()
    summary = json.loads((out_dir / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    assert summary.get("query_bank_id") == "object_memory_logic_core_v1"
    assert summary.get("query_bank_hash")
    assert summary.get("object_memory_logic_status") == "improved"
    assert summary.get("baseline", {}).get("object_memory_logic_variant") == "current"
    assert summary.get("uplift", {}).get("object_memory_logic_variant") == "persistence_v1"
    assert summary.get("uplift", {}).get("object_memory_persistence_items_total") == 4
