from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.query_bank_promotion import write_query_bank_compare_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two aligned v1/v2 query-bank benchmark runs.")
    parser.add_argument("--run_a", required=True, help="Run directory for query-bank A (usually v1)")
    parser.add_argument("--run_b", required=True, help="Run directory for query-bank B (usually v2 candidate)")
    parser.add_argument("--out_dir", required=True, help="Output directory for aligned compare artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_query_bank_compare_outputs(
        run_a=args.run_a,
        run_b=args.run_b,
        out_dir=args.out_dir,
    )
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"saved_summary={outputs['compare_summary_json']}")
    print(f"alignment_ok={outputs['compare_summary'].get('alignment_ok', False)}")
    print(f"compare_summary={json.dumps(outputs['compare_summary'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
