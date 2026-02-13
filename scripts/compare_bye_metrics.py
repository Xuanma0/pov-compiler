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
    parser.add_argument("--format", choices=["md", "csv", "md+csv"], default="md+csv")
    return parser.parse_args()


def _build_rows(run_a: Path, run_b: Path, max_metrics: int = 30) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    summary_a = _read_csv(run_a / "summary.csv")
    summary_b = _read_csv(run_b / "summary.csv")
    by_uid_a = {str(r.get("video_uid", "")): r for r in summary_a if str(r.get("video_uid", ""))}
    by_uid_b = {str(r.get("video_uid", "")): r for r in summary_b if str(r.get("video_uid", ""))}
    uids = sorted(set(by_uid_a.keys()) | set(by_uid_b.keys()))

    metrics_a_by_uid: dict[str, dict[str, float]] = {}
    metrics_b_by_uid: dict[str, dict[str, float]] = {}
    for uid in uids:
        metrics_a_by_uid[uid] = _load_bye_metrics(run_a, uid, by_uid_a.get(uid, {}))
        metrics_b_by_uid[uid] = _load_bye_metrics(run_b, uid, by_uid_b.get(uid, {}))

    common = sorted(
        set().union(*[set(m.keys()) for m in metrics_a_by_uid.values()]).intersection(
            set().union(*[set(m.keys()) for m in metrics_b_by_uid.values()])
        )
    )
    common_metrics = common[: max(0, int(max_metrics))]

    rows: list[dict[str, Any]] = []
    delta_values: dict[str, list[float]] = {m: [] for m in common_metrics}
    for uid in uids:
        sa = by_uid_a.get(uid, {})
        sb = by_uid_b.get(uid, {})
        ma = metrics_a_by_uid.get(uid, {})
        mb = metrics_b_by_uid.get(uid, {})
        row: dict[str, Any] = {
            "uid": uid,
            "status_a": sa.get("bye_status", ""),
            "status_b": sb.get("bye_status", ""),
        }
        for metric in common_metrics:
            va = ma.get(metric)
            vb = mb.get(metric)
            row[f"{metric}_a"] = "" if va is None else float(va)
            row[f"{metric}_b"] = "" if vb is None else float(vb)
            if va is None or vb is None:
                row[f"delta_{metric}"] = ""
            else:
                delta = float(vb - va)
                row[f"delta_{metric}"] = delta
                delta_values[metric].append(delta)
        rows.append(row)

    summary = {
        "uids_total": len(uids),
        "uids_with_a": sum(1 for uid in uids if metrics_a_by_uid.get(uid)),
        "uids_with_b": sum(1 for uid in uids if metrics_b_by_uid.get(uid)),
        "uids_with_both": sum(1 for uid in uids if metrics_a_by_uid.get(uid) and metrics_b_by_uid.get(uid)),
        "metrics_common": len(common_metrics),
        "per_metric": {},
    }
    for metric in common_metrics:
        vals = delta_values.get(metric, [])
        summary["per_metric"][metric] = {
            "count": len(vals),
            "mean_delta": statistics.mean(vals) if vals else None,
            "median_delta": statistics.median(vals) if vals else None,
        }
    return rows, common_metrics, summary


def _write_csv(out_csv: Path, rows: list[dict[str, Any]], common_metrics: list[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["uid", "status_a", "status_b"]
    for metric in common_metrics:
        columns.append(f"{metric}_a")
        columns.append(f"{metric}_b")
    for metric in common_metrics:
        columns.append(f"delta_{metric}")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(out_md: Path, rows: list[dict[str, Any]], common_metrics: list[str], summary: dict[str, Any], run_a: Path, run_b: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    columns = ["uid", "status_a", "status_b"]
    for metric in common_metrics:
        columns.append(f"{metric}_a")
        columns.append(f"{metric}_b")
    for metric in common_metrics:
        columns.append(f"delta_{metric}")

    lines: list[str] = []
    lines.append("# BYE Metrics Compare")
    lines.append("")
    lines.append(f"- run_a: `{run_a}`")
    lines.append(f"- run_b: `{run_b}`")
    lines.append(f"- uids_total: {summary.get('uids_total', 0)}")
    lines.append(f"- uids_with_a: {summary.get('uids_with_a', 0)}")
    lines.append(f"- uids_with_b: {summary.get('uids_with_b', 0)}")
    lines.append(f"- uids_with_both: {summary.get('uids_with_both', 0)}")
    lines.append(f"- metrics_common: {summary.get('metrics_common', 0)}")
    lines.append("")
    lines.append("## Per UID")
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        vals = [str(row.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Delta Stats")
    lines.append("")
    lines.append("| metric | count | mean_delta | median_delta |")
    lines.append("|---|---:|---:|---:|")
    per_metric = summary.get("per_metric", {})
    for metric in common_metrics:
        item = per_metric.get(metric, {}) if isinstance(per_metric, dict) else {}
        mean_delta = "" if item.get("mean_delta") is None else f"{float(item['mean_delta']):.6f}"
        median_delta = "" if item.get("median_delta") is None else f"{float(item['median_delta']):.6f}"
        lines.append(f"| {metric} | {int(item.get('count', 0))} | {mean_delta} | {median_delta} |")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, common_metrics, summary = _build_rows(run_a, run_b, max_metrics=30)
    out_csv = out_dir / "table_bye_compare.csv"
    out_md = out_dir / "table_bye_compare.md"
    summary_path = out_dir / "compare_summary.json"

    if args.format in {"csv", "md+csv"}:
        _write_csv(out_csv, rows, common_metrics)
    if args.format in {"md", "md+csv"}:
        _write_md(out_md, rows, common_metrics, summary, run_a, run_b)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"uids_total={summary.get('uids_total', 0)}")
    print(f"metrics_common={summary.get('metrics_common', 0)}")
    if args.format in {"csv", "md+csv"}:
        print(f"saved_csv={out_csv}")
    if args.format in {"md", "md+csv"}:
        print(f"saved_md={out_md}")
    print(f"saved_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

