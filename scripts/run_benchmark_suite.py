from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.query_bank import copy_query_bank_files, load_query_banks_from_manifest, selection_stats, write_query_bank_lock
from pov_compiler.bench.suite_runner import run_suite


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required to load benchmark manifests.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect, aggregate, and export benchmark-suite artifacts.")
    parser.add_argument("--manifest", required=True, help="Experiment manifest YAML path")
    parser.add_argument("--out_dir", required=True, help="Benchmark suite output directory")
    parser.add_argument("--compare_dir", default=None, help="Optional override for selection.compare_dir")
    parser.add_argument("--mode", choices=["collect-only", "collect", "export"], default="collect-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    raw_manifest = _load_yaml(manifest_path)
    outputs = run_suite(
        manifest_path=manifest_path,
        out_dir=args.out_dir,
        compare_dir_override=args.compare_dir,
        mode=args.mode,
    )
    out_dir = Path(args.out_dir).resolve()
    summary_path = Path(outputs["compare_summary_json"]).resolve()
    snapshot_path = Path(outputs["snapshot_json"]).resolve()
    manifest_dir = out_dir / "manifest"
    source_compare_dir = Path(
        str(
            _read_json(summary_path).get(
                "compare_dir",
                str(raw_manifest.get("selection", {}).get("compare_dir", args.compare_dir or "")),
            )
        )
    ).resolve()

    bank_info = load_query_banks_from_manifest(manifest_path)
    selection_info = selection_stats(compare_dir=source_compare_dir, manifest_path=manifest_path)
    if not selection_info.get("selection_mode"):
        selection_info["selection_mode"] = str(raw_manifest.get("selection", {}).get("mode", "")).strip()

    query_lock = {
        "primary": bank_info.get("primary"),
        "banks": bank_info.get("banks", []),
        "groups": bank_info.get("groups", []),
        "top_k": bank_info.get("top_k"),
        "selection_mode": selection_info.get("selection_mode", ""),
        "signal_min_score": raw_manifest.get("selection", {}).get("signal_min_score"),
        "top_k_uids": raw_manifest.get("selection", {}).get("top_k_uids"),
    }
    query_lock_path = write_query_bank_lock(query_lock, manifest_dir / "query_bank_lock.json")
    copied_banks = copy_query_bank_files(query_lock, manifest_dir / "query_banks")

    compare_summary = _read_json(summary_path)
    primary_bank = bank_info.get("primary") or {}
    manifest_hash = _sha256_path(manifest_path)
    health_gate_profile = str(raw_manifest.get("health_gate_profile", "")).strip()
    paper_map = str(raw_manifest.get("paper_map", "")).strip()
    output_root = str(raw_manifest.get("output_root", raw_manifest.get("output", {}).get("root", ""))).strip()
    compare_summary["query_bank_id"] = str(primary_bank.get("query_bank_id", ""))
    compare_summary["query_bank_version"] = str(primary_bank.get("query_bank_version", ""))
    compare_summary["query_bank_hash"] = str(primary_bank.get("query_bank_hash", ""))
    compare_summary["manifest_hash"] = manifest_hash
    compare_summary["health_gate_profile"] = health_gate_profile
    compare_summary["paper_map"] = paper_map
    compare_summary["output_root"] = output_root
    compare_summary["selection_mode"] = str(selection_info.get("selection_mode", ""))
    compare_summary["selected_uids_count"] = int(selection_info.get("selected_uids_count", 0) or 0)
    compare_summary["coverage_score_stats"] = selection_info.get("coverage_score_stats", {})
    compare_summary["missing_signal_breakdown"] = selection_info.get("missing_signal_breakdown", {})
    compare_summary["query_banks"] = bank_info.get("banks", [])
    _write_json(summary_path, compare_summary)

    snapshot = _read_json(snapshot_path)
    snapshot["query_bank"] = {
        "query_bank_id": str(primary_bank.get("query_bank_id", "")),
        "query_bank_version": str(primary_bank.get("query_bank_version", "")),
        "query_bank_hash": str(primary_bank.get("query_bank_hash", "")),
        "lock_json": str(query_lock_path),
        "copied_files": [str(path) for path in copied_banks],
        "banks": bank_info.get("banks", []),
    }
    snapshot["selection"] = selection_info
    snapshot["manifest_hash"] = manifest_hash
    snapshot["health_gate_profile"] = health_gate_profile
    snapshot["paper_map"] = paper_map
    snapshot["output_root"] = output_root
    _write_json(snapshot_path, snapshot)

    print(f"saved_results_long={outputs['results_long_csv']}")
    print(f"saved_runs={outputs['runs_jsonl']}")
    print(f"saved_table_main={outputs['table_main_csv']}")
    print(f"saved_table_significance={outputs['table_significance_csv']}")
    print(f"saved_table_failure={outputs['table_failure_csv']}")
    print(f"saved_query_bank_lock={query_lock_path}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
