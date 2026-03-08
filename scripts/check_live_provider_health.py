from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.provider_reachability import write_provider_reachability_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check live provider or local OpenAI-compatible reachability.")
    parser.add_argument("--config", required=True, help="Provider health YAML config")
    parser.add_argument("--out_dir", required=True, help="Output directory for reachability artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_provider_reachability_outputs(config_path=args.config, out_dir=args.out_dir)
    summary = outputs.get("summary", {})
    print(f"saved_provider_reachability={Path(args.out_dir).resolve()}")
    print(f"proof_status={summary.get('proof_status', 'fail')}")
    print(f"real_call_status={summary.get('real_call_status', 'unknown')}")
    print(f"reachable={summary.get('reachable', False)}")
    print(f"usage_present={summary.get('usage_present', False)}")
    print(f"latency_ms={summary.get('latency_ms')}")
    print(f"structured_output_supported={summary.get('structured_output_supported', 'unknown')}")
    return 0 if str(summary.get("proof_status", "fail")) in {"ok", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
