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


def _to_float(value: Any) -> float | None:
    try:
        num = float(value)
    except Exception:
        return None
    if num != num:
        return None
    return float(num)


def _read_summary(run_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(run_dir / "summary.csv")
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        uid = str(row.get("video_uid", "")).strip()
        if uid:
            out[uid] = row
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pick_status(row: dict[str, str]) -> str:
    return str(row.get("bye_status", row.get("status", ""))).strip()


def _load_metrics_from_csv(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    if not rows:
        return {}
    row = rows[0]
    out: dict[str, Any] = {
        "bye_status": str(row.get("status", "")).strip() or "ok",
        "bye_primary_score": _to_float(row.get("bye_primary_score")),
        "bye_critical_fn": _to_float(row.get("bye_critical_fn")),
        "bye_latency_p50_ms": _to_float(row.get("bye_latency_p50_ms")),
        "bye_latency_p95_ms": _to_float(row.get("bye_latency_p95_ms")),
    }
    # Fallback for old metrics files.
    if out["bye_primary_score"] is None:
        for key in ("qualityScore", "score", "summary.score", "acc"):
            val = _to_float(row.get(key))
            if val is not None:
                out["bye_primary_score"] = val
                break
    if out["bye_critical_fn"] is None:
        for key in ("critical_fn", "criticalFN", "critical_failures", "critical_fn_rate"):
            val = _to_float(row.get(key))
            if val is not None:
                out["bye_critical_fn"] = val
                break
    if out["bye_latency_p50_ms"] is None:
        for key in ("latency_p50_ms", "latency_ms", "latency"):
            val = _to_float(row.get(key))
            if val is not None:
                out["bye_latency_p50_ms"] = val
                break
    if out["bye_latency_p95_ms"] is None:
        for key in ("latency_p95_ms", "latency_ms", "latency"):
            val = _to_float(row.get(key))
            if val is not None:
                out["bye_latency_p95_ms"] = val
                break
    return out


def _load_uid_metrics(run_dir: Path, uid: str, summary_row: dict[str, str]) -> dict[str, Any]:
    report_json_path = str(summary_row.get("bye_report_metrics_path", "")).strip()
    report_json = (run_dir / report_json_path) if report_json_path and not Path(report_json_path).is_absolute() else Path(report_json_path)
    if not report_json_path:
        report_json = run_dir / "bye" / uid / "bye_report_metrics.json"
    payload = _load_json(report_json)
    if payload:
        return {
            "bye_status": str(payload.get("status", payload.get("bye_status", _pick_status(summary_row) or "ok"))),
            "bye_primary_score": _to_float(payload.get("bye_primary_score")),
            "bye_critical_fn": _to_float(payload.get("bye_critical_fn")),
            "bye_latency_p50_ms": _to_float(payload.get("bye_latency_p50_ms")),
            "bye_latency_p95_ms": _to_float(payload.get("bye_latency_p95_ms")),
            "bye_report_metrics_path": str(report_json),
        }

    csv_path = str(summary_row.get("bye_metrics_path", "")).strip()
    metrics_csv = (run_dir / csv_path) if csv_path and not Path(csv_path).is_absolute() else Path(csv_path)
    if not csv_path:
        metrics_csv = run_dir / "bye" / uid / "bye_metrics.csv"
    from_csv = _load_metrics_from_csv(metrics_csv)
    if from_csv:
        from_csv["bye_report_metrics_path"] = str(metrics_csv)
        if not str(from_csv.get("bye_status", "")).strip():
            from_csv["bye_status"] = _pick_status(summary_row) or "ok"
        return from_csv
    return {
        "bye_status": _pick_status(summary_row) or "missing_report",
        "bye_primary_score": None,
        "bye_critical_fn": None,
        "bye_latency_p50_ms": None,
        "bye_latency_p95_ms": None,
        "bye_report_metrics_path": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BYE report metrics between two run directories")
    parser.add_argument("--a-dir", required=True, help="Run A root")
    parser.add_argument("--b-dir", required=True, help="Run B root")
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--format", choices=["md", "csv", "md+csv"], default="md+csv")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], header_lines: list[str]) -> None:
    lines = ["# BYE Report Compare", "", *header_lines, "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    uids = [str(r.get("uid", "")) for r in rows]
    crit_delta = [float(_to_float(r.get("delta_bye_critical_fn")) or 0.0) for r in rows]
    lat_delta = [float(_to_float(r.get("delta_bye_latency_p95_ms")) or 0.0) for r in rows]

    fig_paths: list[str] = []
    p1 = out_dir / "fig_bye_critical_fn_delta"
    plt.figure(figsize=(8.0, 4.2))
    xs = list(range(len(uids)))
    plt.bar(xs, crit_delta)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xticks(xs, uids, rotation=30, ha="right")
    plt.ylabel("delta bye_critical_fn")
    plt.title("BYE Critical FN Delta (B - A)")
    plt.tight_layout()
    for ext in formats:
        path = p1.with_suffix(f".{ext}")
        plt.savefig(path)
        fig_paths.append(str(path))
    plt.close()

    p2 = out_dir / "fig_bye_latency_delta"
    plt.figure(figsize=(8.0, 4.2))
    plt.bar(xs, lat_delta)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xticks(xs, uids, rotation=30, ha="right")
    plt.ylabel("delta bye_latency_p95_ms")
    plt.title("BYE Latency P95 Delta (B - A)")
    plt.tight_layout()
    for ext in formats:
        path = p2.with_suffix(f".{ext}")
        plt.savefig(path)
        fig_paths.append(str(path))
    plt.close()
    return fig_paths


def main() -> int:
    args = parse_args()
    a_dir = Path(args.a_dir)
    b_dir = Path(args.b_dir)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_a = _read_summary(a_dir)
    summary_b = _read_summary(b_dir)
    uids = sorted(set(summary_a.keys()) | set(summary_b.keys()))

    rows: list[dict[str, Any]] = []
    primary_deltas: list[float] = []
    critical_deltas: list[float] = []
    latency_deltas: list[float] = []

    missing_counts = {
        "bye_primary_score": 0,
        "bye_critical_fn": 0,
        "bye_latency_p50_ms": 0,
        "bye_latency_p95_ms": 0,
    }

    for uid in uids:
        row_a = summary_a.get(uid, {})
        row_b = summary_b.get(uid, {})
        ma = _load_uid_metrics(a_dir, uid, row_a)
        mb = _load_uid_metrics(b_dir, uid, row_b)
        result: dict[str, Any] = {
            "uid": uid,
            "status_a": str(ma.get("bye_status", "")),
            "status_b": str(mb.get("bye_status", "")),
        }
        for key in ("bye_primary_score", "bye_critical_fn", "bye_latency_p50_ms", "bye_latency_p95_ms"):
            va = _to_float(ma.get(key))
            vb = _to_float(mb.get(key))
            result[f"{key}_a"] = "" if va is None else va
            result[f"{key}_b"] = "" if vb is None else vb
            if va is None or vb is None:
                result[f"delta_{key}"] = ""
                missing_counts[key] += 1
            else:
                delta = float(vb - va)
                result[f"delta_{key}"] = delta
                if key == "bye_primary_score":
                    primary_deltas.append(delta)
                elif key == "bye_critical_fn":
                    critical_deltas.append(delta)
                elif key == "bye_latency_p95_ms":
                    latency_deltas.append(delta)
        rows.append(result)

    columns = [
        "uid",
        "status_a",
        "status_b",
        "bye_primary_score_a",
        "bye_primary_score_b",
        "delta_bye_primary_score",
        "bye_critical_fn_a",
        "bye_critical_fn_b",
        "delta_bye_critical_fn",
        "bye_latency_p50_ms_a",
        "bye_latency_p50_ms_b",
        "delta_bye_latency_p50_ms",
        "bye_latency_p95_ms_a",
        "bye_latency_p95_ms_b",
        "delta_bye_latency_p95_ms",
    ]

    table_csv = tables_dir / "table_bye_report_compare.csv"
    table_md = tables_dir / "table_bye_report_compare.md"
    summary_json = out_dir / "compare_summary.json"
    figure_paths = _make_figures(rows, figures_dir, formats=["png", "pdf"])

    if args.format in {"csv", "md+csv"}:
        _write_csv(table_csv, rows, columns)
    if args.format in {"md", "md+csv"}:
        _write_md(
            table_md,
            rows,
            columns,
            header_lines=[
                f"- a_dir: `{a_dir}` ({args.a_label})",
                f"- b_dir: `{b_dir}` ({args.b_label})",
                f"- uids_total: {len(uids)}",
            ],
        )

    summary = {
        "labels": {"a": str(args.a_label), "b": str(args.b_label)},
        "uids_total": len(uids),
        "uids_with_a": len(summary_a),
        "uids_with_b": len(summary_b),
        "uids_with_both": sum(1 for uid in uids if uid in summary_a and uid in summary_b),
        "missing_rates": {k: (float(v) / float(len(uids)) if uids else 0.0) for k, v in missing_counts.items()},
        "delta_stats": {
            "bye_primary_score": {
                "count": len(primary_deltas),
                "mean": statistics.mean(primary_deltas) if primary_deltas else None,
                "median": statistics.median(primary_deltas) if primary_deltas else None,
            },
            "bye_critical_fn": {
                "count": len(critical_deltas),
                "mean": statistics.mean(critical_deltas) if critical_deltas else None,
                "median": statistics.median(critical_deltas) if critical_deltas else None,
            },
            "bye_latency_p95_ms": {
                "count": len(latency_deltas),
                "mean": statistics.mean(latency_deltas) if latency_deltas else None,
                "median": statistics.median(latency_deltas) if latency_deltas else None,
            },
        },
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"a_label={args.a_label}")
    print(f"b_label={args.b_label}")
    print(f"uids_total={len(uids)}")
    if args.format in {"csv", "md+csv"}:
        print(f"saved_table_csv={table_csv}")
    if args.format in {"md", "md+csv"}:
        print(f"saved_table_md={table_md}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

