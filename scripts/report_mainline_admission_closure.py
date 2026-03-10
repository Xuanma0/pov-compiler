from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.mainline_admission_closure import (
    write_mainline_admission_closure_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the final admission-closure conclusion for persistent-memory mainline.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--decision-dir", required=True)
    parser.add_argument("--cleanup-dir", required=True)
    parser.add_argument("--sample-contract-dir", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = write_mainline_admission_closure_outputs(
        suite_dir=args.suite_dir,
        compare_dir=args.compare_dir,
        decision_dir=args.decision_dir,
        cleanup_dir=args.cleanup_dir,
        sample_contract_dir=args.sample_contract_dir,
        out_dir=args.out_dir,
    )
    summary = artifacts["summary"]
    print(f"saved_table={artifacts['csv_path']}")
    print(f"saved_report={artifacts['report_path']}")
    print(f"saved_snapshot={artifacts['snapshot_path']}")
    print(f"mainline_admission_closure_status={summary.get('mainline_admission_closure_status', '')}")
    print("mainline_admission_closure_summary=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
