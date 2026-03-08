from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.query_bank_selection import write_query_bank_rewrite_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a stronger signal-aware query bank candidate from existing suite evidence.")
    parser.add_argument("--suite-dir", required=True, help="Existing suite directory containing manifest/ and evidence outputs")
    parser.add_argument("--out_dir", required=True, help="Output directory for the rewritten query-bank artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_query_bank_rewrite_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
    )
    print(f"saved_summary={outputs['rewrite_summary_json']}")
    print(f"saved_candidate={outputs['candidate_yaml']}")
    print(f"saved_analysis_only={outputs['analysis_yaml']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"rewrite_confidence={outputs['rewrite_summary'].get('rewrite_confidence', 'low')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
