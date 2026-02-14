from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.schemas import Output


def _as_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _collect_json_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(path.glob("*_v03_decisions.json"))


def _select(paths: list[Path], n: int, seed: int) -> list[Path]:
    if n <= 0 or n >= len(paths):
        return list(paths)
    rng = random.Random(int(seed))
    copy = list(paths)
    rng.shuffle(copy)
    return sorted(copy[:n], key=lambda p: p.name)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _group_place_segments(output: Output) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for ev in output.events_v1:
        pid = str(getattr(ev, "place_segment_id", "") or "").strip()
        if not pid:
            pid = "__unknown__"
        current = groups.get(pid)
        t0 = float(ev.t0)
        t1 = float(ev.t1)
        conf = _safe_float(getattr(ev, "place_segment_conf", 0.0))
        reason = str(getattr(ev, "place_segment_reason", "") or "heuristic_merge")
        if current is None:
            groups[pid] = {
                "place_segment_id": pid,
                "t0": t0,
                "t1": t1,
                "conf": conf,
                "reason": reason,
                "events": [str(ev.id)],
            }
        else:
            current["t0"] = min(float(current["t0"]), t0)
            current["t1"] = max(float(current["t1"]), t1)
            current["conf"] = max(_safe_float(current["conf"]), conf)
            current["events"].append(str(ev.id))
    segments = list(groups.values())
    segments.sort(key=lambda x: (_safe_float(x["t0"]), str(x["place_segment_id"])))
    return segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check for place segments and interaction signatures.")
    parser.add_argument("--json", required=True, help="Path to *_v03_decisions.json or directory containing JSON files.")
    parser.add_argument("--index_dir", default="", help="Reserved for compatibility/debug.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--n", type=int, default=1, help="Number of uid JSONs to sample when --json is a directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for directory sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = _collect_json_paths(src_path)
    selected = _select(all_paths, int(args.n), int(args.seed))
    if not selected:
        raise SystemExit(f"no json selected from {src_path}")

    place_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    report_lines: list[str] = ["# Place + Interaction Smoke", ""]
    report_lines.append(f"- selected_uids: {len(selected)}")
    report_lines.append("")

    for path in selected:
        output = ensure_events_v1(_as_output(path))
        uid = str(output.video_id)
        segments = _group_place_segments(output)
        place_rows.append({"video_id": uid, "segments": segments})

        object_counter: Counter[str] = Counter()
        score_vals: list[float] = []
        rate_vals: list[float] = []
        burst_vals: list[int] = []
        for ev in output.events_v1:
            sig = dict(getattr(ev, "interaction_signature", {}) or {})
            obj = str(sig.get("active_object_top1", getattr(ev, "interaction_primary_object", "")) or "").strip().lower()
            if obj:
                object_counter[obj] += 1
            score_vals.append(_safe_float(getattr(ev, "interaction_score", sig.get("interaction_score", 0.0))))
            rate_vals.append(_safe_float(sig.get("contact_rate", 0.0)))
            burst_vals.append(int(_safe_float(sig.get("contact_burst_count", sig.get("contact_bursts", 0.0)))))

        top_object = object_counter.most_common(1)[0][0] if object_counter else ""
        row = {
            "video_id": uid,
            "events_v1_count": int(len(output.events_v1)),
            "place_segments_count": int(len(segments)),
            "interaction_score_mean": float(sum(score_vals) / max(1, len(score_vals))),
            "interaction_score_max": float(max(score_vals) if score_vals else 0.0),
            "contact_rate_mean": float(sum(rate_vals) / max(1, len(rate_vals))),
            "contact_bursts_total": int(sum(burst_vals)),
            "interaction_primary_object_top1": top_object,
        }
        interaction_rows.append(row)

        report_lines.append(f"## {uid}")
        report_lines.append("")
        report_lines.append(f"- place_segments_count: {row['place_segments_count']}")
        report_lines.append(f"- events_v1_count: {row['events_v1_count']}")
        report_lines.append(f"- interaction_primary_object_top1: `{top_object}`")
        report_lines.append(f"- interaction_score_mean: {row['interaction_score_mean']:.4f}")
        report_lines.append(f"- contact_rate_mean: {row['contact_rate_mean']:.4f}")
        report_lines.append(f"- contact_bursts_total: {row['contact_bursts_total']}")
        report_lines.append("")

    place_path = out_dir / "place_segments.json"
    interaction_csv = out_dir / "interaction_summary.csv"
    report_md = out_dir / "report.md"
    snapshot_path = out_dir / "snapshot.json"

    place_path.write_text(json.dumps(place_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = [
        "video_id",
        "events_v1_count",
        "place_segments_count",
        "interaction_score_mean",
        "interaction_score_max",
        "contact_rate_mean",
        "contact_bursts_total",
        "interaction_primary_object_top1",
    ]
    with interaction_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in interaction_rows:
            writer.writerow(row)

    report_md.write_text("\n".join(report_lines), encoding="utf-8")

    cfg_obj = {"json": str(src_path), "n": int(args.n), "seed": int(args.seed), "index_dir": str(args.index_dir)}
    cfg_hash = hashlib.sha1(json.dumps(cfg_obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg_obj,
        "cfg_hash": cfg_hash,
        "selected_json": [str(p) for p in selected],
        "outputs": {
            "place_segments": str(place_path),
            "interaction_summary_csv": str(interaction_csv),
            "report_md": str(report_md),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected_uids={len(selected)}")
    print(f"saved_place_segments={place_path}")
    print(f"saved_interaction_summary={interaction_csv}")
    print(f"saved_report={report_md}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
