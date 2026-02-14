from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-objective budget recommender using BYE + NLQ sweep metrics")
    parser.add_argument("--bye_csv", default=None)
    parser.add_argument("--bye_dir", default=None)
    parser.add_argument("--nlq_csv", default=None)
    parser.add_argument("--nlq_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--weights-json", default=None)
    parser.add_argument("--gates-json", default=None)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--primary-bye-metric", default="qualityScore")
    parser.add_argument("--primary-nlq-metric", default="nlq_full_hit_at_k_strict")
    return parser.parse_args()


def _to_float(v: Any) -> float | None:
    try:
        x = float(v)
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


def _resolve_csv(csv_arg: str | None, dir_arg: str | None) -> Path | None:
    if csv_arg:
        return Path(csv_arg)
    if not dir_arg:
        return None
    d = Path(dir_arg)
    snap = d / "snapshot.json"
    if snap.exists():
        try:
            payload = json.loads(snap.read_text(encoding="utf-8"))
            outputs = payload.get("outputs", {}) if isinstance(payload, dict) else {}
            path_raw = outputs.get("metrics_by_budget_csv") if isinstance(outputs, dict) else None
            if isinstance(path_raw, str) and path_raw.strip():
                p = Path(path_raw)
                if not p.is_absolute():
                    p = (d / p).resolve()
                if p.exists():
                    return p
        except Exception:
            pass
    fallback = d / "aggregate" / "metrics_by_budget.csv"
    return fallback


def _budget_key(row: dict[str, Any]) -> str:
    s = _to_float(row.get("budget_max_total_s", row.get("budget_seconds", row.get("max_total_s"))))
    t = _to_float(row.get("budget_max_tokens", row.get("max_tokens")))
    d = _to_float(row.get("budget_max_decisions", row.get("max_decisions")))
    if s is not None and t is not None and d is not None:
        return f"{int(round(s))}/{int(t)}/{int(d)}"
    return str(row.get("budget_key", row.get("budget_tag", "unknown_budget")))


def _budget_seconds(row: dict[str, Any]) -> float:
    v = _to_float(row.get("budget_max_total_s", row.get("budget_seconds", row.get("max_total_s"))))
    return float(v if v is not None else 0.0)


def _numeric_map(row: dict[str, Any], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    reserved = {
        "budget_key",
        "budget_tag",
        "budget_seconds",
        "budget_max_total_s",
        "budget_max_tokens",
        "budget_max_decisions",
        "max_total_s",
        "max_tokens",
        "max_decisions",
    }
    for key, value in row.items():
        if key in reserved:
            continue
        num = _to_float(value)
        if num is None:
            continue
        name = str(key)
        if prefix == "bye":
            out_key = name if name.startswith("bye_") else f"bye_{name}"
        else:
            out_key = name if name.startswith("nlq_") else f"nlq_{name}"
        out[out_key] = float(num)
    return out


def _normalize_metric_name(name: str, source: str) -> str:
    key = str(name).strip()
    if source == "bye":
        return key if key.startswith("bye_") else f"bye_{key}"
    return key if key.startswith("nlq_") else f"nlq_{key}"


def _parse_json_obj(raw: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if raw is None:
        return dict(default)
    text = str(raw).strip()
    if not text:
        return dict(default)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else dict(default)
    except Exception:
        path = Path(text)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else dict(default)
            except Exception:
                return dict(default)
    return dict(default)


def _eval_gate(metric_value: float | None, op: str, threshold: float) -> bool:
    if metric_value is None:
        return False
    if op == "<=":
        return float(metric_value) <= float(threshold)
    if op == "<":
        return float(metric_value) < float(threshold)
    if op == ">=":
        return float(metric_value) >= float(threshold)
    if op == ">":
        return float(metric_value) > float(threshold)
    if op == "==":
        return float(metric_value) == float(threshold)
    return False


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Budget Recommendation",
        "",
        f"- label: {summary.get('label', '')}",
        f"- primary_bye_metric: {summary.get('primary_bye_metric', '')}",
        f"- primary_nlq_metric: {summary.get('primary_nlq_metric', '')}",
        f"- budgets_joined: {summary.get('budgets_joined', 0)}",
        f"- budgets_accepted: {summary.get('budgets_accepted', 0)}",
        f"- top1_budget_key: {summary.get('top1_budget_key', '')}",
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
    primary_nlq_metric: str,
) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: float(r.get("budget_seconds", 0.0)))
    xs = [float(_to_float(r.get("budget_seconds")) or 0.0) for r in sorted_rows]
    ys_obj = [float(_to_float(r.get("objective")) or 0.0) for r in sorted_rows]
    ys_nlq = [float(_to_float(r.get(primary_nlq_metric)) or 0.0) for r in sorted_rows]

    out: list[str] = []
    p1 = out_dir / "fig_objective_vs_budget_seconds"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, ys_obj, marker="o")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("objective")
    plt.title("Objective vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = p1.with_suffix(f".{ext}")
        plt.savefig(path)
        out.append(str(path))
    plt.close()

    p2 = out_dir / "fig_pareto_frontier"
    plt.figure(figsize=(7.0, 4.2))
    plt.scatter(xs, ys_nlq)
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel(primary_nlq_metric)
    plt.title("Pareto Frontier (budget_seconds vs primary_nlq)")
    plt.grid(True, alpha=0.35)
    # naive frontier: maximize y, minimize x
    points = sorted([(x, y) for x, y in zip(xs, ys_nlq)], key=lambda t: (t[0], -t[1]))
    frontier: list[tuple[float, float]] = []
    best_y = float("-inf")
    for x, y in points:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    if frontier:
        plt.plot([x for x, _ in frontier], [y for _, y in frontier], linestyle="--")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = p2.with_suffix(f".{ext}")
        plt.savefig(path)
        out.append(str(path))
    plt.close()
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    bye_csv = _resolve_csv(args.bye_csv, args.bye_dir)
    nlq_csv = _resolve_csv(args.nlq_csv, args.nlq_dir)
    if bye_csv is None or nlq_csv is None or not bye_csv.exists() or not nlq_csv.exists():
        print(f"error=missing input csv bye={bye_csv} nlq={nlq_csv}")
        return 2

    bye_rows = _read_csv(bye_csv)
    nlq_rows = _read_csv(nlq_csv)
    if not bye_rows or not nlq_rows:
        print("error=empty input csv")
        return 2

    bye_map = {_budget_key(r): r for r in bye_rows}
    nlq_map = {_budget_key(r): r for r in nlq_rows}
    joined_keys = sorted(set(bye_map.keys()) & set(nlq_map.keys()), key=lambda k: _budget_seconds(bye_map[k]))
    if not joined_keys:
        print("error=no joined budgets between bye/nlq")
        return 2

    primary_bye_metric = _normalize_metric_name(str(args.primary_bye_metric), "bye")
    primary_nlq_metric = _normalize_metric_name(str(args.primary_nlq_metric), "nlq")
    default_weights = {
        "bye_qualityScore": 1.0,
        "nlq_full_hit_at_k_strict": 1.0,
        "nlq_full_fp_rate": -0.5,
        "nlq_full_top1_in_distractor_rate": -0.5,
        "budget_seconds": -0.1,
    }
    weights = _parse_json_obj(args.weights_json, default_weights)
    gates = _parse_json_obj(args.gates_json, {"safety_critical_fn_rate_full": {"op": "<=", "value": 0.2}})

    rows: list[dict[str, Any]] = []
    for key in joined_keys:
        br = bye_map[key]
        nr = nlq_map[key]
        merged: dict[str, Any] = {
            "budget_key": key,
            "budget_seconds": _budget_seconds(br),
            **_numeric_map(br, "bye"),
            **_numeric_map(nr, "nlq"),
        }

        reason = ""
        accepted = True
        for gate_key, gate_val in gates.items():
            if not isinstance(gate_val, dict):
                continue
            op = str(gate_val.get("op", "<="))
            threshold = _to_float(gate_val.get("value"))
            if threshold is None:
                continue
            candidates = [str(gate_key)]
            gk = str(gate_key)
            if not gk.startswith("bye_"):
                candidates.append(f"bye_{gk}")
            if not gk.startswith("nlq_"):
                candidates.append(f"nlq_{gk}")
            metric_val = None
            for cand in candidates:
                metric_val = _to_float(merged.get(cand))
                if metric_val is not None:
                    break
            if metric_val is None:
                # Missing gate metric: skip this gate instead of hard reject.
                continue
            if not _eval_gate(metric_val, op, threshold):
                accepted = False
                reason = f"gate_failed:{gate_key}{op}{threshold}"
                break

        objective = 0.0
        for wk, wv in weights.items():
            weight = _to_float(wv)
            if weight is None:
                continue
            metric = _to_float(merged.get(str(wk)))
            if metric is None:
                continue
            objective += float(weight) * float(metric)

        row = {
            "budget_key": key,
            "budget_seconds": float(merged.get("budget_seconds", 0.0)),
            "status": "accepted" if accepted else "rejected",
            "objective": float(objective),
            "rank": "",
            "reason": reason,
            primary_bye_metric: _to_float(merged.get(primary_bye_metric)),
            primary_nlq_metric: _to_float(merged.get(primary_nlq_metric)),
            **merged,
        }
        rows.append(row)

    accepted_rows = sorted([r for r in rows if str(r["status"]) == "accepted"], key=lambda r: float(r["objective"]), reverse=True)
    for i, row in enumerate(accepted_rows, start=1):
        row["rank"] = int(i)

    topn = max(1, int(args.topn))
    topn_keys = [str(r["budget_key"]) for r in accepted_rows[:topn]]
    top1 = topn_keys[0] if topn_keys else ""

    fixed_cols = [
        "budget_key",
        "budget_seconds",
        "status",
        "objective",
        "rank",
        "reason",
        primary_bye_metric,
        primary_nlq_metric,
    ]
    numeric_extra = sorted(
        [
            k
            for k in rows[0].keys()
            if k not in set(fixed_cols) and isinstance(_to_float(rows[0].get(k)), float)
        ]
    )
    if len(numeric_extra) > 30:
        numeric_extra = numeric_extra[:30]
    columns = fixed_cols + [c for c in numeric_extra if c not in fixed_cols]

    table_csv = tables_dir / "table_budget_recommend.csv"
    table_md = tables_dir / "table_budget_recommend.md"
    summary_json = out_dir / "recommend_summary.json"
    _write_csv(table_csv, rows, columns)
    _write_md(
        table_md,
        rows,
        columns,
        {
            "label": str(args.label),
            "primary_bye_metric": primary_bye_metric,
            "primary_nlq_metric": primary_nlq_metric,
            "budgets_joined": len(joined_keys),
            "budgets_accepted": len(accepted_rows),
            "top1_budget_key": top1,
        },
    )
    figure_paths = _make_figures(rows=rows, out_dir=figs_dir, primary_nlq_metric=primary_nlq_metric)

    summary = {
        "label": str(args.label),
        "budgets_total": len(set(bye_map.keys()) | set(nlq_map.keys())),
        "budgets_joined": len(joined_keys),
        "budgets_accepted": len(accepted_rows),
        "top1_budget_key": top1,
        "topn": topn_keys,
        "weights": weights,
        "gates": gates,
        "primary_bye_metric": primary_bye_metric,
        "primary_nlq_metric": primary_nlq_metric,
        "inputs": {"bye_csv": str(bye_csv), "nlq_csv": str(nlq_csv)},
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"budgets_joined={len(joined_keys)}")
    print(f"budgets_accepted={len(accepted_rows)}")
    print(f"top1={top1}")
    print(f"saved_table_csv={table_csv}")
    print(f"saved_table_md={table_md}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
