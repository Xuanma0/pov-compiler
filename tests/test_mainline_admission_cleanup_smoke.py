from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_suite_and_compare(tmp_path: Path) -> tuple[Path, Path]:
    suite_dir = tmp_path / "suite"
    compare_root = tmp_path / "compare_root"
    compare_dir = compare_root / "compare"

    _write_json(
        suite_dir / "result_health" / "snapshot.json",
        {
            "admission_status": "partial",
            "calibration_status": "ok",
            "compare_summary": {
                "uid_set_id": "persistent_memory_main_large_sample_v1",
                "sample_signature_hash": "samplehash",
            },
            "selection": {
                "selected_uids_count": 6,
                "coverage_score_stats": {"mean": 2.75, "min": 2.4, "max": 3.1},
            },
        },
    )
    _write_json(
        suite_dir / "admission_control" / "snapshot.json",
        {
            "admission_status": "partial",
            "admission_fail_reasons": ["provider_availability=unavailable"],
        },
    )
    _write_json(
        suite_dir / "provider_normalization" / "snapshot.json",
        {"normalization_status": "ok", "real_call_status": "observed"},
    )
    _write_json(
        suite_dir / "provider_reachability" / "summary.json",
        {"proof_status": "ok", "real_call_status": "ok"},
    )
    _write_json(
        suite_dir / "freeze" / "freeze_manifest.json",
        {
            "artifact_count": 30,
            "freeze_sha256": "abc",
            "query_bank_id": "persistent_object_memory_core_v1",
            "query_bank_hash": "hash123",
        },
    )
    _write_json(
        compare_dir / "compare_summary.json",
        {
            "compare_pair_id": "persistent_memory_main_real",
            "alignment_ok": True,
            "mismatch_reasons": [],
            "selected_uids_count": 6,
            "paired_sample_count": 12,
            "budget_keys": ["20/50/4", "40/100/8"],
            "query_bank_a_id": "persistent_object_memory_core_v1",
            "query_bank_b_id": "persistent_object_memory_core_v1",
            "query_bank_a_hash": "hash123",
            "query_bank_b_hash": "hash123",
            "object_memory_logic_variant_b": "persistent_v2",
            "provider_signature_match": True,
            "perception_signature_match": True,
            "provider_noise_summary_hash_a": "sig",
            "provider_noise_summary_hash_b": "sig",
            "provider_health_status": "ok",
            "admission_status_b": "partial",
            "persistent_memory_main_status": "improved",
            "main_gain_state": "strict_main_gain_improved",
        },
    )
    _write_json(
        compare_root / "promotion_decision" / "snapshot.json",
        {
            "promotion_decision_summary": {
                "promotion_decision": "promote_persistent_memory_to_mainline",
                "promotion_ready": True,
                "decision_confidence": "high",
                "recommended_next_step": "promote_persistent_memory_to_mainline",
            }
        },
    )
    return suite_dir, compare_root


def test_mainline_admission_cleanup_smoke(tmp_path: Path) -> None:
    suite_dir, compare_root = _seed_suite_and_compare(tmp_path)
    out_dir = tmp_path / "cleanup"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_mainline_admission_cleanup.py"),
            "--suite-dir",
            str(suite_dir),
            "--compare-dir",
            str(compare_root),
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
    summary = dict(snapshot.get("mainline_admission_cleanup_summary", {}))
    assert summary.get("mainline_admission_cleanup_status") == "explained"
    assert summary.get("promotion_vs_admission_consistency_status") == "consistent_different_layers"
    assert summary.get("sample_contract_status") == "borderline"
    assert summary.get("large_sample_claim_status") == "supported_with_caveat"
    assert (out_dir / "tables" / "table_mainline_admission_cleanup.csv").exists()
