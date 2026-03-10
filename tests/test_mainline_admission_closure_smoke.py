from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_mainline_admission_closure_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    compare_root = tmp_path / "compare_root"
    compare_dir = compare_root / "compare"
    decision_dir = compare_root / "promotion_decision"
    cleanup_dir = tmp_path / "cleanup"
    sample_dir = tmp_path / "harder_sample_contract"

    _write_json(
        compare_dir / "compare_summary.json",
        {
            "persistent_memory_main_status": "improved",
            "large_sample_claim_status": "supported",
            "provider_health_status": "ok",
        },
    )
    _write_json(
        decision_dir / "snapshot.json",
        {
            "promotion_decision_summary": {
                "promotion_decision": "promote_persistent_memory_to_mainline",
                "promotion_ready": True,
                "mainline_admission_ready": True,
                "recommended_next_step": "promote_persistent_memory_to_mainline",
            }
        },
    )
    _write_json(
        cleanup_dir / "snapshot.json",
        {
            "mainline_admission_cleanup_summary": {
                "mainline_admission_cleanup_status": "improved",
                "evidence_hardness_status": "supported",
                "provider_cleanliness_status": "clean",
                "provider_semantics_gap": False,
            }
        },
    )
    _write_json(
        sample_dir / "snapshot.json",
        {
            "harder_sample_contract_summary": {
                "sample_contract_status": "adequate",
                "large_sample_claim_status": "supported",
                "wording_recommendation": "large-sample real main experiment",
            }
        },
    )

    out_dir = tmp_path / "closure"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_mainline_admission_closure.py"),
            "--suite-dir",
            str(suite_dir),
            "--compare-dir",
            str(compare_root),
            "--decision-dir",
            str(decision_dir),
            "--cleanup-dir",
            str(cleanup_dir),
            "--sample-contract-dir",
            str(sample_dir),
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    summary = dict(snapshot.get("mainline_admission_closure_summary", {}))
    assert summary.get("promotion_ready") is True
    assert summary.get("mainline_admission_ready") is True
    assert summary.get("mainline_admission_closure_status") == "closed"
    assert summary.get("recommended_next_step") == "mainline_admission_closed"
    assert (out_dir / "tables" / "table_mainline_admission_closure.csv").exists()
