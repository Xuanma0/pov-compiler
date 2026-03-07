from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.result_diagnosis import write_result_diagnosis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a result-diagnosis report from a benchmark suite output.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for diagnosis artifacts")
    parser.add_argument("--near-zero-threshold", type=float, default=1e-6)
    parser.add_argument("--effect-size-threshold", type=float, default=1e-9)
    parser.add_argument("--significance-threshold", type=float, default=0.50)
    parser.add_argument("--provider-telemetry-dir", default=None, help="Optional directory with supplemental provider telemetry CSVs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_result_diagnosis_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
        near_zero_threshold=float(args.near_zero_threshold),
        effect_size_threshold=float(args.effect_size_threshold),
        significance_threshold=float(args.significance_threshold),
        provider_telemetry_dir=args.provider_telemetry_dir,
    )
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"no_data_reason_counts={outputs['overall_no_data_reason_counts']}")
    print(f"provider_noise_summary={outputs['provider_noise_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
