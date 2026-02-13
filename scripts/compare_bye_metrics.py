from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:
        return None
    return x


def _load_bye_metrics(run_dir: Path, uid: str, summary_row: dict[str, str]) -> dict[str, float]:
    path_text = str(summary_row.get("bye_metrics_path", "")).strip()
    if path_text:
        p = Path(path_text)
        metrics_path = p if p.is_absolute() else (run_dir / p)
    else:
        metrics_path = run_dir / "bye" / uid / "bye_metrics.csv"
    rows = _read_csv(metrics_path)
    if not rows:
        return {}
    row = rows[0]
    out: dict[str, float] = {}
    for k, v in row.items():
        if k in {"status", "report_path", "summary_keys"}:
            continue
        num = _to_float(v)
        if num is not None:
            out[str(k)] = float(num)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BYE metrics between two smoke runs")
    parser.add_argument("--run_a", required=True, help="Run A root (e.g. stub)")
    parser.add_argument("--run_b", required=True, help="Run B root (e.g. real)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_a = _read_csv(run_a / "summary.csv")
    summary_b = _read_csv(run_b / "summary.csv")
    by_uid_a = {str(r.get("video_uid", "")): r for r in summary_a if str(r.get("video_uid", ""))}
    by_uid_b = {str(r.get("video_uid", "")): r for r in summary_b if str(r.get("video_uid", ""))}
    uids = sorted(set(by_uid_a.keys()) | set(by_uid_b.keys()))

    label_a = run_a.name or "run_a"
    label_b = run_b.name or "run_b"

    metrics_a_by_uid: dict[str, dict[str, float]] = {}
    metrics_b_by_uid: dict[str, dict[str, float]] = {}
    for uid in uids:
        metrics_a_by_uid[uid] = _load_bye_metrics(run_a, uid, by_uid_a.get(uid, {}))
        metrics_b_by_uid[uid] = _load_bye_metrics(run_b, uid, by_uid_b.get(uid, {}))

    common_metrics = sorted(
        set().union(*[set(m.keys()) for m in metrics_a_by_uid.values()]).intersection(
            set().union(*[set(m.keys()) for m in metrics_b_by_uid.values()])
        )
    )

    rows: list[dict[str, Any]] = []
    delta_values: dict[str, list[float]] = {m: [] for m in common_metrics}
    valid_uid_count = 0
    missing_uid_count = 0
    for uid in uids:
        sa = by_uid_a.get(uid, {})
        sb = by_uid_b.get(uid, {})
        ma = metrics_a_by_uid.get(uid, {})
        mb = metrics_b_by_uid.get(uid, {})
        row: dict[str, Any] = {
            "uid": uid,
            f"bye_status_{label_a}": sa.get("bye_status", ""),
            f"bye_status_{label_b}": sb.get("bye_status", ""),
        }
        has_any = False
        for metric in common_metrics:
            va = ma.get(metric)
            vb = mb.get(metric)
            row[f"{metric}_{label_a}"] = "" if va is None else float(va)
            row[f"{metric}_{label_b}"] = "" if vb is None else float(vb)
            if va is None or vb is None:
                row[f"delta_{metric}"] = ""
            else:
                delta = float(vb - va)
                row[f"delta_{metric}"] = delta
                delta_values[metric].append(delta)
                has_any = True
        if has_any:
            valid_uid_count += 1
        else:
            missing_uid_count += 1
        rows.append(row)

    stats_rows: list[dict[str, Any]] = []
    for metric in common_metrics:
        vals = delta_values.get(metric, [])
        stats_rows.append(
            {
                "metric": metric,
                "valid_count": len(vals),
                "mean_delta": statistics.mean(vals) if vals else None,
                "median_delta": statistics.median(vals) if vals else None,
            }
        )

    out_csv = out_dir / "table_bye_compare.csv"
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    out_md = out_dir / "table_bye_compare.md"
    lines: list[str] = []
    lines.append("# BYE Metrics Compare")
    lines.append("")
    lines.append(f"- run_a: `{run_a}`")
    lines.append(f"- run_b: `{run_b}`")
    lines.append(f"- uid_total: {len(uids)}")
    lines.append(f"- uid_with_common_metrics: {valid_uid_count}")
    lines.append(f"- uid_missing_common_metrics: {missing_uid_count}")
    lines.append("")
    lines.append("## Per UID")
    lines.append("")
    if rows and columns:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for row in rows:
            vals = [str(row.get(c, "")) for c in columns]
            lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("No comparable BYE metrics found.")

    lines.append("")
    lines.append("## Delta Stats")
    lines.append("")
    lines.append("| metric | valid_count | mean_delta | median_delta |")
    lines.append("|---|---:|---:|---:|")
    for row in stats_rows:
        mean_delta = "" if row["mean_delta"] is None else f"{float(row['mean_delta']):.6f}"
        median_delta = "" if row["median_delta"] is None else f"{float(row['median_delta']):.6f}"
        lines.append(f"| {row['metric']} | {row['valid_count']} | {mean_delta} | {median_delta} |")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"uids_total={len(uids)}")
    print(f"metrics_common={len(common_metrics)}")
    print(f"saved_csv={out_csv}")
    print(f"saved_md={out_md}")

    snapshot = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "uids_total": len(uids),
        "uids_with_common_metrics": valid_uid_count,
        "uids_missing_common_metrics": missing_uid_count,
        "common_metrics": common_metrics,
    }
    (out_dir / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

