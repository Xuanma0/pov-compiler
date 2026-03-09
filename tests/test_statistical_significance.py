from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.manifest import ExperimentManifest
from pov_compiler.bench.reporting.significance import build_significance_tables, paired_bootstrap_ci


def _demo_manifest() -> ExperimentManifest:
    return ExperimentManifest.model_validate(
        {
            "suite_id": "sig_demo",
            "suite_version": "1.0",
            "seed": 7,
            "selection": {
                "tasks": ["nlq"],
                "labels": {"a": "stub", "b": "real"},
            },
            "variants": {"baseline": "a", "treatment": "b"},
            "budgets": {"points": [{"key": "20/50/4"}]},
            "metrics": {
                "primary": {"nlq": "score"},
                "binary": {"nlq": "hit_binary"},
            },
        }
    )


def test_build_significance_tables_with_paired_rows() -> None:
    manifest = _demo_manifest()
    rows: list[dict[str, object]] = []
    for idx in range(8):
        rows.append(
            {
                "task": "nlq",
                "label_key": "a",
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "sample_unit": f"u{idx}",
                "source_kind": "pair",
                "score": 0.20 + 0.03 * idx,
                "hit_binary": 0 if idx < 5 else 1,
            }
        )
        rows.append(
            {
                "task": "nlq",
                "label_key": "b",
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "sample_unit": f"u{idx}",
                "source_kind": "pair",
                "score": 0.35 + 0.03 * idx,
                "hit_binary": 1,
            }
        )
    table_df, ci_df = build_significance_tables(pd.DataFrame(rows), manifest)
    row = table_df.iloc[0]

    assert row["status"] == "ok"
    assert int(row["n_pairs"]) == 8
    assert float(row["mean_delta"]) > 0.0
    assert pd.notna(row["wilcoxon_p"])
    assert pd.notna(row["mcnemar_p"])
    assert not ci_df.empty


def test_significance_degrades_to_insufficient_pairs() -> None:
    ci = paired_bootstrap_ci([0.1], [0.2], seed=0)
    assert ci["status"] == "insufficient_pairs"


def test_run_statistical_significance_cli(tmp_path: Path) -> None:
    manifest_path = ROOT / "configs" / "benchmarks" / "v1.42_main.yaml"
    suite_dir = tmp_path / "suite"
    (suite_dir / "ledger").mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest" / "manifest_resolved.json").write_text(
        manifest_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "task": "nlq",
                "label_key": "a",
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "sample_unit": "aggregate",
                "source_kind": "aggregate",
                "nlq_full_hit_at_k_strict": 0.4,
            },
            {
                "task": "nlq",
                "label_key": "b",
                "budget_key": "20/50/4",
                "budget_seconds": 20,
                "sample_unit": "aggregate",
                "source_kind": "aggregate",
                "nlq_full_hit_at_k_strict": 0.5,
            },
        ]
    ).to_csv(suite_dir / "ledger" / "results_long.csv", index=False)

    out_dir = tmp_path / "significance"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_statistical_significance.py"),
        "--suite_dir",
        str(suite_dir),
        "--manifest",
        str(manifest_path),
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "tables" / "table_significance_main.csv").exists()
    assert (out_dir / "tables" / "table_confidence_intervals.csv").exists()
    assert (out_dir / "report.md").exists()
