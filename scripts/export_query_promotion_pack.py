from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.query_promotion_pack import write_query_promotion_pack_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a candidate query-promotion pack from query-strength audit outputs.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for the promotion pack")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_query_promotion_pack_outputs(suite_dir=args.suite_dir, out_dir=args.out_dir)
    summary = outputs["summary"]
    print(f"saved_promoted={outputs['promoted_yaml']}")
    print(f"saved_analysis_only={outputs['analysis_only_yaml']}")
    print(f"saved_summary={outputs['promotion_summary_json']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"query_promotion_summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
