from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_object_persistence_pilot_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "v155_signal_uplift_fake"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_object_persistence_pilot.py"),
            "--manifest",
            str(ROOT / "configs" / "benchmarks" / "v1.55_signal_uplift_fake.yaml"),
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
    assert "object_persistence_status=improved" in proc.stdout
    assert (out_dir / "compare" / "tables" / "table_object_persistence_uplift.csv").exists()
    assert (out_dir / "compare" / "figures" / "fig_object_persistence_uplift_delta.png").exists()
    assert (out_dir / "compare" / "README.md").exists()
    summary = json.loads((out_dir / "compare" / "compare_summary.json").read_text(encoding="utf-8"))
    assert summary.get("query_bank_id") == "object_persistence_core_v1"
    assert summary.get("query_bank_hash")
    assert summary.get("object_persistence_status") == "improved"
    assert summary.get("baseline", {}).get("segmentation_backend_used", "") == ""
    assert summary.get("uplift", {}).get("segmentation_backend_used") == "sam3_local_proxy"
    assert summary.get("uplift", {}).get("persistent_tracks_total") == 5
