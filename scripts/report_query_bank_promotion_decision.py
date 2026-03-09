from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.query_bank_promotion import write_query_bank_promotion_decision_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a promotion decision for v1 vs v2 query-bank compare outputs.")
    parser.add_argument("--compare_dir", required=True, help="Query-bank compare output root")
    parser.add_argument("--out_dir", required=True, help="Output directory for the promotion-decision report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_query_bank_promotion_decision_outputs(
        compare_dir=args.compare_dir,
        out_dir=args.out_dir,
    )
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(
        f"promotion_decision_summary={json.dumps(outputs['promotion_decision_summary'], ensure_ascii=False, sort_keys=True)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
