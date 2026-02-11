from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.utils.fs import list_mp4_files
from pov_compiler.utils.media import probe_video_metadata


def _duration_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 30:
        return "<30s"
    if value < 60:
        return "30-60s"
    if value < 180:
        return "1-3m"
    if value < 600:
        return "3-10m"
    if value < 1800:
        return "10-30m"
    return ">30m"


def _fmt_time(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _probe_one(path: Path) -> dict[str, Any]:
    meta = probe_video_metadata(path)
    stat = path.stat()
    size_bytes = int(stat.st_size)
    row = {
        "path": str(path.resolve()),
        "size_bytes": size_bytes,
        "size_mb": float(size_bytes / (1024.0**2)),
        "mtime": _fmt_time(float(stat.st_mtime)),
        "duration_s": meta.get("duration_s"),
        "fps": meta.get("fps"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "probe_backend": meta.get("probe_backend"),
        "ok": bool(meta.get("ok", False)),
        "error": meta.get("error"),
    }
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build video catalog with duration/fps/resolution metadata")
    parser.add_argument("--root", default=r"D:\Ego4D_Dataset", help="Dataset root")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=0, help="Probe only first N files (0 means all)")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel probe jobs")
    parser.add_argument("--min-size-bytes", "--min_size_bytes", dest="min_size_bytes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"error=root_not_found path={root}")
        return 1

    mp4_files = list_mp4_files(root=root, min_size_bytes=int(args.min_size_bytes))
    if int(args.limit) > 0:
        mp4_files = mp4_files[: int(args.limit)]

    rows: list[dict[str, Any]] = []
    jobs = max(1, int(args.jobs))
    if jobs <= 1:
        for path in mp4_files:
            rows.append(_probe_one(path))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_probe_one, path) for path in mp4_files]
            for fut in as_completed(futures):
                rows.append(fut.result())

    rows.sort(key=lambda r: str(r.get("path", "")).lower())
    out_csv = Path(args.out)
    out_jsonl = out_csv.with_suffix(".jsonl")
    _write_csv(out_csv, rows)
    _write_jsonl(out_jsonl, rows)

    total = len(rows)
    ok_rows = [r for r in rows if bool(r.get("ok"))]
    fail_rows = [r for r in rows if not bool(r.get("ok"))]
    dist: dict[str, int] = {}
    for row in rows:
        key = _duration_bin(row.get("duration_s"))
        dist[key] = dist.get(key, 0) + 1

    with_duration = [r for r in rows if isinstance(r.get("duration_s"), (int, float))]
    shortest = sorted(with_duration, key=lambda r: float(r.get("duration_s", 0.0)))[:10]
    longest = sorted(with_duration, key=lambda r: float(r.get("duration_s", 0.0)), reverse=True)[:10]

    print(f"root={root}")
    print(f"total_mp4={total}")
    print(f"probed_ok={len(ok_rows)}")
    print(f"probed_failed={len(fail_rows)}")
    print(f"saved_csv={out_csv}")
    print(f"saved_jsonl={out_jsonl}")
    print("duration_distribution:")
    for key in ["<30s", "30-60s", "1-3m", "3-10m", "10-30m", ">30m", "unknown"]:
        if key in dist:
            print(f"  {key}: {dist[key]}")

    print("top10_shortest:")
    for row in shortest:
        print(f"  {float(row['duration_s']):8.2f}s  {row['path']}")
    print("top10_longest:")
    for row in longest:
        print(f"  {float(row['duration_s']):8.2f}s  {row['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
