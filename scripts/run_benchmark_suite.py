from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.suite_runner import run_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect, aggregate, and export benchmark-suite artifacts.")
    parser.add_argument("--manifest", required=True, help="Experiment manifest YAML path")
    parser.add_argument("--out_dir", required=True, help="Benchmark suite output directory")
    parser.add_argument("--compare_dir", default=None, help="Optional override for selection.compare_dir")
    parser.add_argument("--mode", choices=["collect-only", "collect", "export"], default="collect-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_suite(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        compare_dir_override=args.compare_dir,
        mode=args.mode,
    )
    print(f"saved_results_long={outputs['results_long_csv']}")
    print(f"saved_runs={outputs['runs_jsonl']}")
    print(f"saved_table_main={outputs['table_main_csv']}")
    print(f"saved_table_significance={outputs['table_significance_csv']}")
    print(f"saved_table_failure={outputs['table_failure_csv']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
