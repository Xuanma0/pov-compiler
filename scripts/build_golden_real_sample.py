from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.golden_real_sample import write_golden_real_sample_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a golden real-sample contract from a pilot suite output.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for golden real sample artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_golden_real_sample_outputs(suite_dir=args.suite_dir, out_dir=args.out_dir)
    sample_manifest = outputs.get("sample_manifest", {})
    print(f"saved_golden_real_sample={Path(args.out_dir).resolve()}")
    print(f"golden_sample_status={outputs.get('status', 'empty')}")
    print(f"selected_query_groups={sample_manifest.get('selected_query_groups', [])}")
    print(f"selected_query_ids={sample_manifest.get('selected_query_ids', [])}")
    return 0 if str(outputs.get("status", "empty")) in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
