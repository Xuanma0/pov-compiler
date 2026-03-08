from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.query_uplift_candidates import write_query_uplift_candidates_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export query uplift candidates from query-strength audit outputs.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for query uplift candidates")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_query_uplift_candidates_outputs(suite_dir=args.suite_dir, out_dir=args.out_dir)
    print(f"saved_candidate={outputs['candidate_yaml']}")
    print(f"saved_analysis_only={outputs['analysis_only_yaml']}")
    print(f"saved_drop={outputs['drop_yaml']}")
    print(f"saved_summary={outputs['uplift_summary_json']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"query_uplift_summary={outputs['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
