from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.paper_map import canonical_artifacts_for_freeze


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze canonical paper tables and figures by hashing exported artifacts.")
    parser.add_argument("--paper-ready-dir", required=True, help="paper_ready output directory")
    parser.add_argument("--paper-map", required=True, help="Canonical paper-map YAML")
    parser.add_argument("--out_dir", required=True, help="Paper freeze output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper_ready_dir = Path(args.paper_ready_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = canonical_artifacts_for_freeze(paper_ready_dir=paper_ready_dir, paper_map_path=args.paper_map)
    rows: list[dict[str, object]] = []
    for path in sorted(resolved["files"], key=lambda item: str(item)):
        rows.append(
            {
                "artifact_group": "paper_artifact",
                "relpath": str(path.relative_to(paper_ready_dir)) if path.is_relative_to(paper_ready_dir) else str(path.name),
                "sha256": _sha256_path(path),
                "size_bytes": path.stat().st_size,
            }
        )

    csv_path = out_dir / "paper_artifacts_sha256.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact_group", "relpath", "sha256", "size_bytes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    canonical_rows = [
        {"relpath": row["relpath"], "sha256": row["sha256"]}
        for row in sorted(rows, key=lambda item: str(item["relpath"]))
    ]
    freeze_hash = hashlib.sha256(
        json.dumps(canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "paper_map_id": resolved["paper_map_id"],
        "paper_map_version": resolved["paper_map_version"],
        "paper_map_hash": resolved["paper_map_hash"],
        "artifact_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_freeze_sha256": freeze_hash,
        "paper_ready_dir": str(paper_ready_dir),
    }
    manifest_path = out_dir / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"artifact_count={len(rows)}")
    print(f"saved_freeze_manifest={manifest_path}")
    print(f"saved_paper_artifacts_sha256={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
