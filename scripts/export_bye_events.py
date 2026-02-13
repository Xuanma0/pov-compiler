from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.integrations.bye.exporter import export_bye_events_from_output_dict, write_jsonl


def _infer_video_id(output_dict: dict[str, Any], json_path: Path) -> str:
    video_id = output_dict.get("video_id")
    if isinstance(video_id, str) and video_id:
        return video_id
    stem = json_path.stem
    for suffix in ("_v03_decisions", "_v02_token", "_v01_decision", "_v01", "_v0"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _get_git_commit_short() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        value = (proc.stdout or "").strip()
        return value or None
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export POV output json to BYE events_v1.jsonl format")
    parser.add_argument("--json", required=True, help="Path to input POV output json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--video_id", default=None, help="Optional override for output video_id")
    parser.add_argument(
        "--include",
        default="events_v1,highlights,tokens,decisions",
        help="Comma-separated sections: events_v1,highlights,tokens,decisions",
    )
    parser.add_argument("--no-sort", action="store_true", help="Disable deterministic sorting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input json must be an object")

    include = tuple([x.strip() for x in str(args.include).split(",") if x.strip()])
    video_id = str(args.video_id) if args.video_id else _infer_video_id(payload, json_path)

    events = export_bye_events_from_output_dict(
        payload,
        video_id=video_id,
        include=include,
        sort=not bool(args.no_sort),
    )

    events_dir = out_dir / "events"
    jsonl_path = events_dir / "events_v1.jsonl"
    write_jsonl(events, str(jsonl_path))

    counts = Counter([str(x.get("name", "")) for x in events])
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit_short(),
        "input_path": str(json_path),
        "video_id": video_id,
        "include": list(include),
        "counts_by_name": dict(sorted(counts.items())),
        "counts_total": int(len(events)),
        "saved_jsonl": str(jsonl_path),
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"video_id={video_id}")
    print(f"counts_total={len(events)}")
    for name, count in sorted(counts.items()):
        print(f"count_{name}={count}")
    print(f"saved_jsonl={jsonl_path}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

