from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.repeatability_audit import write_repeatability_audit_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a repeatability audit from repeated benchmark outputs.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for repeatability artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_repeatability_audit_outputs(suite_dir=args.suite_dir, out_dir=args.out_dir)
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"repeatability_status={outputs['summary'].get('repeatability_status', 'weak')}")
    print(f"stability_flag_counts={outputs['summary'].get('stability_flag_counts', {})}")
    return 0 if str(outputs["summary"].get("repeatability_status", "weak")) in {"ok", "partial", "weak"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
