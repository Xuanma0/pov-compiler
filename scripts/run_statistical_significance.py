from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.manifest import ExperimentManifest, load_manifest
from pov_compiler.bench.reporting.significance import write_significance_outputs


def _load_manifest_any(path: Path) -> ExperimentManifest:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Resolved manifest JSON must be a mapping: {path}")
        payload.pop("manifest_path", None)
        return ExperimentManifest.model_validate(payload)
    manifest, _ = load_manifest(path)
    return manifest


def _load_results(path: Path) -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required to load results_long.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired significance analysis over benchmark suite outputs")
    parser.add_argument("--suite_dir", default=None, help="Benchmark suite output root")
    parser.add_argument("--results_long", default=None, help="Path to ledger/results_long.csv")
    parser.add_argument("--manifest", default=None, help="Path to manifest YAML or resolved JSON")
    parser.add_argument("--out_dir", required=True, help="Output directory for significance artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_dir = Path(args.suite_dir) if args.suite_dir else None

    if args.results_long:
        results_path = Path(args.results_long)
    elif suite_dir is not None:
        results_path = suite_dir / "ledger" / "results_long.csv"
    else:
        raise ValueError("Either --results_long or --suite_dir is required.")

    if args.manifest:
        manifest_path = Path(args.manifest)
    elif suite_dir is not None and (suite_dir / "manifest" / "manifest_resolved.json").exists():
        manifest_path = suite_dir / "manifest" / "manifest_resolved.json"
    elif suite_dir is not None and (suite_dir / "manifest" / "experiment_manifest.yaml").exists():
        manifest_path = suite_dir / "manifest" / "experiment_manifest.yaml"
    else:
        raise ValueError("Could not resolve a manifest path. Use --manifest.")

    manifest = _load_manifest_any(manifest_path)
    results_long = _load_results(results_path)
    outputs = write_significance_outputs(results_long=results_long, manifest=manifest, out_dir=args.out_dir)

    print(f"rows_total={len(outputs['table_df'])}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_ci={outputs['ci_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
