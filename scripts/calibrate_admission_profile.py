from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.admission_calibration import write_admission_calibration_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build admission calibration recommendations from a pilot suite output.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for admission calibration artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_admission_calibration_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
    )
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"calibration_status={outputs['calibration_status']}")
    print(f"calibration_summary={outputs['calibration_summary']}")
    return 0 if str(outputs["calibration_status"]) != "weak" else 1


if __name__ == "__main__":
    raise SystemExit(main())
