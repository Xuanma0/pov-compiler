from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.query_bank import QueryBank, load_query_banks_from_manifest


def test_query_bank_configs_load_and_hash_match() -> None:
    for rel in (
        "configs/queries/core_real_v1.yaml",
        "configs/queries/core_chain_v1.yaml",
        "configs/queries/core_lost_object_v1.yaml",
    ):
        path = ROOT / rel
        bank = QueryBank.from_path(path)
        assert bank.query_bank_id
        assert bank.query_bank_version
        assert bank.groups
        assert bank.queries
        assert bank.declared_hash_ok(path)


def test_main_fake_manifest_references_primary_and_auxiliary_query_banks() -> None:
    manifest_path = ROOT / "configs" / "benchmarks" / "v1.43_main_fake.yaml"
    bank_info = load_query_banks_from_manifest(manifest_path)
    primary = bank_info.get("primary") or {}
    banks = bank_info.get("banks", [])
    assert primary.get("query_bank_id") == "core_real_v1"
    assert primary.get("query_bank_hash")
    assert len(banks) == 3
