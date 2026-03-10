from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_suite_and_compare(tmp_path: Path) -> tuple[Path, Path, Path]:
    suite_dir = tmp_path / "suite"
    compare_root = tmp_path / "compare_root"
    compare_dir = compare_root / "compare"
    manifest_path = tmp_path / "v1.60_mainline_harder_real_persistent.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "compare_pair_id: persistent_memory_main_real_harder",
                "uid_set_id: persistent_memory_main_large_sample_v2",
                "source_query_bank_id: persistent_object_memory_core_v1",
                "source_query_bank_hash: hash123",
                "object_memory:",
                "  logic_variant: persistent_v2",
                "thresholds:",
                "  selected_uids_count_borderline_min: 8",
                "  selected_uids_count_adequate_min: 8",
                "  paired_sample_count_borderline_min: 16",
                "  paired_sample_count_adequate_min: 16",
                "  coverage_mean_min: 2.7",
                "  coverage_min_min: 2.2",
                "  freeze_artifact_count_min: 1",
            ]
        ),
        encoding="utf-8",
    )

    _write_json(
        suite_dir / "result_health" / "snapshot.json",
        {
            "selection": {
                "selected_uids_count": 8,
                "coverage_score_stats": {"mean": 2.9, "min": 2.4, "max": 3.2},
            },
            "compare_summary": {
                "uid_set_id": "persistent_memory_main_large_sample_v2",
                "sample_signature_hash": "samplehash_v2",
            },
        },
    )
    _write_json(
        suite_dir / "freeze" / "freeze_manifest.json",
        {
            "artifact_count": 32,
            "freeze_sha256": "abc",
            "query_bank_id": "persistent_object_memory_core_v1",
            "query_bank_hash": "hash123",
        },
    )
    _write_json(
        compare_dir / "compare_summary.json",
        {
            "compare_pair_id": "persistent_memory_main_real_harder",
            "alignment_ok": True,
            "mismatch_reasons": [],
            "selected_uids_count": 8,
            "paired_sample_count": 16,
            "budget_keys": ["20/50/4", "40/100/8"],
            "uid_set_id": "persistent_memory_main_large_sample_v2",
            "sample_signature_hash": "samplehash_v2",
            "query_bank_a_id": "persistent_object_memory_core_v1",
            "query_bank_b_id": "persistent_object_memory_core_v1",
            "query_bank_a_hash": "hash123",
            "query_bank_b_hash": "hash123",
            "object_memory_logic_variant_b": "persistent_v2",
            "provider_signature_match": True,
            "perception_signature_match": True,
            "provider_noise_summary_hash_a": "sig",
            "provider_noise_summary_hash_b": "sig",
        },
    )
    return suite_dir, compare_root, manifest_path


def test_harder_sample_contract_smoke(tmp_path: Path) -> None:
    suite_dir, compare_root, manifest_path = _seed_suite_and_compare(tmp_path)
    out_dir = tmp_path / "harder_sample_contract"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_harder_sample_contract.py"),
            "--suite-dir",
            str(suite_dir),
            "--compare-dir",
            str(compare_root),
            "--out_dir",
            str(out_dir),
            "--manifest",
            str(manifest_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    summary = dict(snapshot.get("harder_sample_contract_summary", {}))
    assert summary.get("sample_contract_status") == "adequate"
    assert summary.get("large_sample_claim_status") == "supported"
    assert summary.get("selected_uids_gap_to_adequate") == 0
    assert summary.get("paired_sample_gap_to_adequate") == 0
    assert (out_dir / "tables" / "table_harder_sample_contract_summary.csv").exists()
