from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BYE budget sweep outputs (A/B)")
    parser.add_argument("--a_dir", default=None, help="Sweep A output directory (contains snapshot.json)")
    parser.add_argument("--b_dir", default=None, help="Sweep B output directory (contains snapshot.json)")
    parser.add_argument("--a_csv", default=None, help="Sweep A aggregate metrics_by_budget.csv")
    parser.add_argument("--b_csv", default=None, help="Sweep B aggregate metrics_by_budget.csv")
    parser.add_argument("--a_label", default="A")
    parser.add_argument("--b_label", default="B")
    parser.add_argument("--primary-metric", default="qualityScore")
    parser.add_argument("--x-axis", choices=["budget_seconds"], default="budget_seconds")
    parser.add_argument("--format", choices=["md", "csv", "md+csv"], default="md+csv")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if x != x:
        return None
    return x


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_csv_from_dir(sweep_dir: Path) -> Path:
    snapshot = sweep_dir / "snapshot.json"
    if snapshot.exists():
        try:
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            outputs = payload.get("outputs", {}) if isinstance(payload, dict) else {}
            csv_path_raw = outputs.get("metrics_by_budget_csv") if isinstance(outputs, dict) else None
            if isinstance(csv_path_raw, str) and csv_path_raw.strip():
                candidate = Path(csv_path_raw)
                if not candidate.is_absolute():
                    candidate = (sweep_dir / candidate).resolve()
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    fallback = sweep_dir / "aggregate" / "metrics_by_budget.csv"
    return fallback


def _value_by_alias(row: dict[str, Any], aliases: list[str]) -> Any:
    for key in aliases:
        if key in row:
            return row.get(key)
    return None


def _budget_fields(row: dict[str, Any]) -> tuple[float | None, int | None, int | None]:
    total = _to_float(_value_by_alias(row, ["budget_max_total_s", "max_total_s", "max_total_seconds"]))
    tokens_f = _to_float(_value_by_alias(row, ["budget_max_tokens", "max_tokens"]))
    decisions_f = _to_float(_value_by_alias(row, ["budget_max_decisions", "max_decisions"]))
    tokens = int(tokens_f) if tokens_f is not None else None
    decisions = int(decisions_f) if decisions_f is not None else None
    return total, tokens, decisions


def _budget_key(row: dict[str, Any]) -> str:
    total, tokens, decisions = _budget_fields(row)
    if total is not None and tokens is not None and decisions is not None:
        total_tag = int(round(total))
        return f"{total_tag}/{tokens}/{decisions}"
    tag = str(_value_by_alias(row, ["budget_tag"]) or "").strip()
    if tag:
        return tag
    return "unknown_budget"


def _budget_seconds(row: dict[str, Any]) -> float:
    total, _, _ = _budget_fields(row)
    return float(total if total is not None else 0.0)


def _status_from_row(row: dict[str, Any]) -> str:
    val = str(row.get("status", "")).strip()
    if val:
        return val
    ok = _to_float(row.get("runs_ok"))
    total = _to_float(row.get("runs_total"))
    if ok is not None and total is not None:
        return "ok" if int(ok) == int(total) else "partial"
    return ""


def _numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    reserved = {
        "budget_tag",
        "budget_max_total_s",
        "budget_max_tokens",
        "budget_max_decisions",
        "max_total_s",
        "max_tokens",
        "max_decisions",
        "num_uids",
        "runs_ok",
        "runs_total",
        "selection_mode",
        "uids_requested",
        "uids_found",
        "uids_missing_count",
    }
    out: list[str] = []
    for key in rows[0].keys():
        if key in reserved:
            continue
        vals = [_to_float(r.get(key)) for r in rows]
        if any(v is not None for v in vals):
            out.append(key)
    return out


def _to_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _budget_key(row)
        if key not in out:
            out[key] = row
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BYE Budget Compare",
        "",
        f"- labels: {summary.get('labels', {})}",
        f"- primary_metric: {summary.get('primary_metric')}",
        f"- budgets_total: {summary.get('budgets_total')}",
        f"- budgets_matched: {summary.get('budgets_matched')}",
        f"- budgets_only_a: {summary.get('budgets_only_a')}",
        f"- budgets_only_b: {summary.get('budgets_only_b')}",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    a_label: str,
    b_label: str,
    primary_metric: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: float(r.get("budget_seconds", 0.0)))
    xs = [float(r.get("budget_seconds", 0.0)) for r in sorted_rows]
    ya = [float(r.get(f"{primary_metric}_{a_label}", 0.0)) for r in sorted_rows]
    yb = [float(r.get(f"{primary_metric}_{b_label}", 0.0)) for r in sorted_rows]
    yd = [float(r.get(f"delta_{primary_metric}", 0.0)) for r in sorted_rows]

    fig_paths: list[Path] = []

    fig1 = out_dir / "fig_bye_primary_vs_budget_seconds_compare"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, ya, marker="o", label=a_label)
    plt.plot(xs, yb, marker="o", label=b_label)
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel(primary_metric)
    plt.title(f"{primary_metric} vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = fig1.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(p)
    plt.close()

    fig2 = out_dir / "fig_bye_primary_delta_vs_budget_seconds"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, yd, marker="o")
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel(f"delta_{primary_metric} ({b_label}-{a_label})")
    plt.title(f"Delta {primary_metric} vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = fig2.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(p)
    plt.close()

    return fig_paths


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.a_csv and args.a_dir:
        print("error=use either --a_csv or --a_dir")
        return 2
    if args.b_csv and args.b_dir:
        print("error=use either --b_csv or --b_dir")
        return 2
    if not (args.a_csv or args.a_dir) or not (args.b_csv or args.b_dir):
        print("error=provide A and B inputs via --*_csv or --*_dir")
        return 2

    a_csv = Path(args.a_csv) if args.a_csv else _resolve_csv_from_dir(Path(args.a_dir))
    b_csv = Path(args.b_csv) if args.b_csv else _resolve_csv_from_dir(Path(args.b_dir))
    if not a_csv.exists() or not b_csv.exists():
        print(f"error=metrics csv missing: a={a_csv} b={b_csv}")
        return 2

    a_rows = _read_csv(a_csv)
    b_rows = _read_csv(b_csv)
    if not a_rows or not b_rows:
        print("error=empty csv input")
        return 2

    a_map = _to_map(a_rows)
    b_map = _to_map(b_rows)
    keys_a = set(a_map.keys())
    keys_b = set(b_map.keys())
    matched_keys = sorted(keys_a & keys_b, key=lambda k: float(_budget_seconds(a_map[k])))
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    a_numeric = set(_numeric_columns(a_rows))
    b_numeric = set(_numeric_columns(b_rows))
    common_numeric = sorted(a_numeric & b_numeric)
    if len(common_numeric) > 30:
        common_numeric = common_numeric[:30]

    primary_metric = str(args.primary_metric)
    if primary_metric not in common_numeric:
        if common_numeric:
            primary_metric = common_numeric[0]
        else:
            primary_metric = "runs_ok"
    other_numeric = [c for c in common_numeric if c != primary_metric]

    table_rows: list[dict[str, Any]] = []
    for key in matched_keys:
        ra = a_map[key]
        rb = b_map[key]
        r: dict[str, Any] = {
            "budget_key": key,
            "budget_seconds": _budget_seconds(ra),
            f"{primary_metric}_{args.a_label}": _to_float(ra.get(primary_metric)),
            f"{primary_metric}_{args.b_label}": _to_float(rb.get(primary_metric)),
            f"delta_{primary_metric}": (
                (_to_float(rb.get(primary_metric)) or 0.0) - (_to_float(ra.get(primary_metric)) or 0.0)
            ),
            "status_a": _status_from_row(ra),
            "status_b": _status_from_row(rb),
        }
        for c in other_numeric:
            av = _to_float(ra.get(c))
            bv = _to_float(rb.get(c))
            r[f"{c}_{args.a_label}"] = av
            r[f"{c}_{args.b_label}"] = bv
            r[f"delta_{c}"] = None if av is None or bv is None else (bv - av)
        table_rows.append(r)

    columns = [
        "budget_key",
        "budget_seconds",
        f"{primary_metric}_{args.a_label}",
        f"{primary_metric}_{args.b_label}",
        f"delta_{primary_metric}",
        "status_a",
        "status_b",
    ]
    for c in other_numeric:
        columns.extend([f"{c}_{args.a_label}", f"{c}_{args.b_label}", f"delta_{c}"])

    summary = {
        "budgets_total": len(keys_a | keys_b),
        "budgets_matched": len(matched_keys),
        "budgets_only_a": only_a,
        "budgets_only_b": only_b,
        "metrics_numeric_common": common_numeric,
        "primary_metric": primary_metric,
        "labels": {"a_label": args.a_label, "b_label": args.b_label},
        "inputs": {"a_csv": str(a_csv), "b_csv": str(b_csv)},
    }

    table_csv = tables_dir / "table_budget_compare.csv"
    table_md = tables_dir / "table_budget_compare.md"
    summary_json = out_dir / "compare_summary.json"
    figure_paths = _make_figures(
        rows=table_rows,
        out_dir=figures_dir,
        a_label=args.a_label,
        b_label=args.b_label,
        primary_metric=primary_metric,
    )

    if args.format in {"csv", "md+csv"}:
        _write_csv(table_csv, table_rows, columns)
    if args.format in {"md", "md+csv"}:
        _write_md(table_md, table_rows, columns, summary)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"a_label={args.a_label}")
    print(f"b_label={args.b_label}")
    print(f"primary_metric={primary_metric}")
    print(f"budgets_matched={len(matched_keys)}")
    if args.format in {"csv", "md+csv"}:
        print(f"saved_table_csv={table_csv}")
    if args.format in {"md", "md+csv"}:
        print(f"saved_table_md={table_md}")
    print(f"saved_figures={[str(p) for p in figure_paths]}")
    print(f"saved_summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

