from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.provider_normalization import write_provider_normalization_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize provider telemetry into a unified result-layer schema.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for normalized telemetry artifacts")
    parser.add_argument("--provider-telemetry-dir", default=None, help="Optional raw provider telemetry directory")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional provider normalization YAML (defaults to configs/telemetry/provider_normalization_v1.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_provider_normalization_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
        provider_telemetry_dir=args.provider_telemetry_dir,
        config_path=args.config,
    )
    summary = outputs["summary"]
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"normalization_status={summary.get('normalization_status', 'unavailable')}")
    print(f"real_call_status={summary.get('real_call_status', 'missing_or_unavailable')}")
    print(f"missing_fields_union={summary.get('missing_fields_union', [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
