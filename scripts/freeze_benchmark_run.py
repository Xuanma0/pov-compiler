from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_artifacts(suite_dir: Path) -> list[tuple[str, Path]]:
    compare_dir = suite_dir / "compare"
    manifest_dir = suite_dir / "manifest"
    significance_dir = suite_dir / "significance"
    health_dir = suite_dir / "result_health"

    items: list[tuple[str, Path]] = []
    for path in sorted((compare_dir / "tables").glob("*")):
        if path.suffix.lower() in {".csv", ".md"}:
            items.append(("compare_tables", path))
    for path in sorted((compare_dir / "figures").glob("*")):
        if path.suffix.lower() in {".png", ".pdf"}:
            items.append(("compare_figures", path))
    for name in ("compare_summary.json", "snapshot.json"):
        path = compare_dir / name
        if path.exists():
            items.append(("compare_meta", path))
    for name in ("experiment_manifest.yaml", "manifest_resolved.json", "prompt_lock.json", "query_bank_lock.json"):
        path = manifest_dir / name
        if path.exists():
            items.append(("manifest", path))
    for path in sorted((manifest_dir / "query_banks").glob("*")):
        if path.is_file():
            items.append(("query_banks", path))
    for path in sorted((significance_dir / "tables").glob("*")):
        if path.suffix.lower() in {".csv", ".md"}:
            items.append(("significance", path))
    for name in ("report.md", "snapshot.json"):
        path = significance_dir / name
        if path.exists():
            items.append(("significance", path))
    for path in sorted((health_dir / "tables").glob("*")):
        if path.suffix.lower() in {".csv", ".md"}:
            items.append(("result_health", path))
    for path in sorted((health_dir / "figures").glob("*")):
        if path.suffix.lower() in {".png", ".pdf"}:
            items.append(("result_health", path))
    health_snapshot = health_dir / "snapshot.json"
    if health_snapshot.exists():
        items.append(("result_health", health_snapshot))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze benchmark artifacts by hashing key outputs.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Freeze output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _collect_artifacts(suite_dir)
    rows: list[dict[str, Any]] = []
    for group, path in artifacts:
        rows.append(
            {
                "artifact_group": group,
                "relpath": str(path.relative_to(suite_dir)),
                "sha256": _sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
        )

    csv_path = out_dir / "artifacts_sha256.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact_group", "relpath", "sha256", "size_bytes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    query_bank_lock = _load_json(suite_dir / "manifest" / "query_bank_lock.json")
    primary_bank = query_bank_lock.get("primary") if isinstance(query_bank_lock, dict) else {}
    compare_summary = _load_json(suite_dir / "compare" / "compare_summary.json")
    manifest_hash = ""
    manifest_path = suite_dir / "manifest" / "experiment_manifest.yaml"
    if manifest_path.exists():
        manifest_hash = _sha256_path(manifest_path)
    canonical_rows = [
        {"relpath": row["relpath"], "sha256": row["sha256"]}
        for row in sorted(rows, key=lambda item: str(item["relpath"]))
    ]
    freeze_hash = hashlib.sha256(json.dumps(canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    freeze_manifest = {
        "suite_id": str(compare_summary.get("suite_id", "")),
        "query_bank_id": str(primary_bank.get("query_bank_id", "")),
        "query_bank_hash": str(primary_bank.get("query_bank_hash", "")),
        "manifest_hash": manifest_hash,
        "health_gate_profile": str(compare_summary.get("health_gate_profile", "")),
        "paper_map": str(compare_summary.get("paper_map", "")),
        "artifact_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_hash,
    }
    manifest_out = out_dir / "freeze_manifest.json"
    manifest_out.write_text(json.dumps(freeze_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"artifact_count={len(rows)}")
    print(f"saved_freeze_manifest={manifest_out}")
    print(f"saved_artifacts_sha256={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
