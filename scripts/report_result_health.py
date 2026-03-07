from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.result_health import write_result_health_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a result-health report from a benchmark suite output.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for result-health artifacts")
    parser.add_argument("--epsilon", type=float, default=1e-9, help="Absolute threshold for zero-delta checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_result_health_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
        epsilon=float(args.epsilon),
    )
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"no_data_reason_counts={outputs['no_data_reason_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
