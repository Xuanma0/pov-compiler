from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.signal_uplift import write_signal_uplift_report_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write signal-uplift summary outputs.")
    parser.add_argument("--suite_dir", required=True, help="Signal-uplift suite directory to summarize")
    parser.add_argument("--baseline_dir", required=True, help="Baseline suite directory used for comparison")
    parser.add_argument("--out_dir", required=True, help="Output directory for summary tables/report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_signal_uplift_report_outputs(
        suite_dir=args.suite_dir,
        baseline_dir=args.baseline_dir,
        out_dir=args.out_dir,
    )
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"signal_uplift_recommendation={outputs['next_action_recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
