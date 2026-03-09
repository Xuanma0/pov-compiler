from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.query_strength_audit import write_query_strength_audit_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a query-strength audit from a benchmark suite output.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for query-strength audit artifacts")
    parser.add_argument("--effect-size-threshold", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_query_strength_audit_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
        effect_size_threshold=float(args.effect_size_threshold),
    )
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"query_strength_summary={outputs['query_strength_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
