from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select UIDs for experiments from signal coverage table")
    parser.add_argument("--coverage-csv", required=True, help="Path to coverage.csv from audit_signal_coverage.py")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--min-score", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--require-cols",
        default="",
        help="Comma-separated required coverage columns (truthy/positive), e.g. interaction_events_count,lost_object_queries_total",
    )
    return parser.parse_args()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _is_truthy_cell(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan"}:
        return False
    if text in {"true", "yes", "y"}:
        return True
    if text in {"false", "no", "n"}:
        return False
    return _to_float(value, default=0.0) > 0.0


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()
    coverage_csv = Path(args.coverage_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(coverage_csv)
    require_cols = [x.strip() for x in str(args.require_cols).split(",") if x.strip()]

    filtered: list[dict[str, str]] = []
    reason_counts: dict[str, int] = {"below_min_score": 0}
    for col in require_cols:
        reason_counts[f"missing_required_{col}"] = 0

    for row in rows:
        score = _to_float(row.get("coverage_score"), default=0.0)
        if score < float(args.min_score):
            reason_counts["below_min_score"] = reason_counts.get("below_min_score", 0) + 1
            continue
        failed = False
        for col in require_cols:
            if not _is_truthy_cell(row.get(col, "")):
                reason_counts[f"missing_required_{col}"] = reason_counts.get(f"missing_required_{col}", 0) + 1
                failed = True
                break
        if failed:
            continue
        filtered.append(row)

    filtered_sorted = sorted(
        filtered,
        key=lambda r: (-_to_float(r.get("coverage_score"), default=0.0), str(r.get("uid", "")).strip()),
    )
    selected = filtered_sorted[: max(0, int(args.top_k))]
    selected_uids = [str(r.get("uid", "")).strip() for r in selected if str(r.get("uid", "")).strip()]

    selected_uids_path = out_dir / "selected_uids.txt"
    selection_report_path = out_dir / "selection_report.md"
    snapshot_path = out_dir / "snapshot.json"

    selected_uids_path.write_text("\n".join(selected_uids) + ("\n" if selected_uids else ""), encoding="utf-8")

    lines = [
        "# Selection Report",
        "",
        f"- coverage_csv: `{coverage_csv}`",
        f"- min_score: `{float(args.min_score)}`",
        f"- top_k: `{int(args.top_k)}`",
        f"- require_cols: `{require_cols}`",
        f"- rows_total: `{len(rows)}`",
        f"- rows_after_filters: `{len(filtered)}`",
        f"- selected_uids_count: `{len(selected_uids)}`",
        "",
        "## Filtered Reasons",
        "",
    ]
    for k in sorted(reason_counts.keys()):
        lines.append(f"- {k}: {int(reason_counts.get(k, 0))}")
    lines.extend(["", "## Selected UIDs", ""])
    for uid in selected_uids:
        lines.append(f"- {uid}")
    selection_report_path.write_text("\n".join(lines), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "coverage_csv": str(coverage_csv),
            "min_score": float(args.min_score),
            "top_k": int(args.top_k),
            "require_cols": require_cols,
        },
        "counts": {
            "rows_total": len(rows),
            "rows_after_filters": len(filtered),
            "selected_uids_count": len(selected_uids),
            "reason_counts": reason_counts,
        },
        "outputs": {
            "selected_uids": str(selected_uids_path),
            "selection_report": str(selection_report_path),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected_uids_count={len(selected_uids)}")
    print(f"saved_selected_uids={selected_uids_path}")
    print(f"saved_selection_report={selection_report_path}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

