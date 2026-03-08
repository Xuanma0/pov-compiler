from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.sample_size_recommendation import write_sample_size_recommendation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend next-round sample sizes from repeated pilot outputs.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for sample-size recommendation artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_sample_size_recommendation_outputs(suite_dir=args.suite_dir, out_dir=args.out_dir)
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"sample_size_recommendation_status={outputs['summary'].get('sample_size_recommendation_status', 'weak')}")
    print(f"recommendation_status_counts={outputs['summary'].get('recommendation_status_counts', {})}")
    return 0 if str(outputs["summary"].get("sample_size_recommendation_status", "weak")) in {"ok", "range_only", "weak"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
