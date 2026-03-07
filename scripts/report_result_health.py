from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.result_health import write_result_health_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a result-health report from a benchmark suite output.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for result-health artifacts")
    parser.add_argument("--epsilon", type=float, default=1e-9, help="Absolute threshold for zero-delta checks")
    parser.add_argument("--gate-profile", default=None, help="Optional health-gate profile name")
    parser.add_argument("--diagnosis-dir", default=None, help="Optional diagnosis directory for snapshot linkage")
    return parser.parse_args()


def _annotate_snapshot(snapshot_path: Path, diagnosis_dir: str | None) -> None:
    if not snapshot_path.exists():
        return
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    diagnosis_path = Path(diagnosis_dir).resolve() if diagnosis_dir else None
    payload["diagnosis_available"] = bool(diagnosis_path is not None and diagnosis_path.exists())
    payload["diagnosis_dir"] = str(diagnosis_path) if diagnosis_path is not None else None
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    outputs = write_result_health_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
        epsilon=float(args.epsilon),
        gate_profile=args.gate_profile,
    )
    _annotate_snapshot(Path(outputs["snapshot_json"]), args.diagnosis_dir)
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"no_data_reason_counts={outputs['no_data_reason_counts']}")
    print(f"gate_status={outputs['gate_status']}")
    print(f"gate_fail_reasons={outputs['gate_fail_reasons']}")
    return 0 if str(outputs["gate_status"]) != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
