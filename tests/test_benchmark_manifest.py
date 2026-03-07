from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.manifest import BudgetPoint, ExperimentManifest, load_manifest


def test_repo_manifest_loads_and_resolves() -> None:
    manifest_path = ROOT / "configs" / "benchmarks" / "v1.42_main.yaml"
    manifest, resolved = load_manifest(manifest_path)

    assert manifest.suite_id == "v1.42_main"
    assert manifest.suite_version == "1.42"
    assert manifest.selection.labels["a"] == "stub"
    assert manifest.selection.labels["b"] == "real"
    assert [point.key for point in manifest.budgets.points] == ["20/50/4", "40/100/8", "60/200/12"]
    assert resolved["selection"]["compare_dir"].endswith(str(Path("data/outputs/v1_42_compare")))
    assert resolved["prompts"]["registry"].endswith(str(Path("configs/prompts/registry_v1.yaml")))


def test_budget_point_normalizes_key_and_fields() -> None:
    point = BudgetPoint(key="30/120/6")
    assert point.budget_seconds == 30.0
    assert point.max_total_s == 30.0
    assert point.max_tokens == 120
    assert point.max_decisions == 6

    manifest = ExperimentManifest.model_validate(
        {
            "suite_id": "demo",
            "suite_version": "1.0",
            "selection": {"labels": {"a": "base", "b": "new"}},
            "budgets": {"points": [{"budget_seconds": 25, "max_tokens": 80, "max_decisions": 5}]},
            "variants": {"baseline": "a", "treatment": "b"},
        }
    )
    assert manifest.budgets.points[0].key == "25/80/5"
