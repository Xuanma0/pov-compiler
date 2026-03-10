from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.persistent_memory_main import (
    write_persistent_memory_main_compare_outputs,
)
from pov_compiler.bench.reporting.mainline_admission_closure import (
    refresh_persistent_memory_main_compare_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare large-sample baseline vs persistent-memory main runs.")
    parser.add_argument("--run_a", required=True, help="Run directory for the baseline mainline run")
    parser.add_argument("--run_b", required=True, help="Run directory for the persistent-memory mainline run")
    parser.add_argument("--out_dir", required=True, help="Output directory for the persistent-memory main compare")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_persistent_memory_main_compare_outputs(
        run_a=args.run_a,
        run_b=args.run_b,
        out_dir=args.out_dir,
    )
    compare_summary = refresh_persistent_memory_main_compare_outputs(
        run_a=args.run_a,
        run_b=args.run_b,
        out_dir=args.out_dir,
    )
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"saved_summary={outputs['compare_summary_json']}")
    print(f"alignment_ok={compare_summary.get('alignment_ok', False)}")
    print(
        "persistent_memory_main_compare_summary="
        + json.dumps(compare_summary, ensure_ascii=False, sort_keys=True)
    )
    return 0 if compare_summary.get("alignment_ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
