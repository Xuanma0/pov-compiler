from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.mainline_admission_cleanup import write_sample_contract_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write sample/coverage/freeze contract evidence for mainline wording.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = write_sample_contract_outputs(
        suite_dir=args.suite_dir,
        compare_dir=args.compare_dir,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
    )
    summary = artifacts["summary"]
    print(f"saved_table={artifacts['csv_path']}")
    print(f"saved_report={artifacts['report_path']}")
    print(f"saved_snapshot={artifacts['snapshot_path']}")
    print(f"sample_contract_status={summary.get('sample_contract_status', '')}")
    print(f"large_sample_claim_status={summary.get('large_sample_claim_status', '')}")
    print(f"sample_contract_summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
