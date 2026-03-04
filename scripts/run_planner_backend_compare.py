from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        return f"{int(round(self.max_total_s))}/{int(self.max_tokens)}/{int(self.max_decisions)}"

    @property
    def tag(self) -> str:
        return f"s{int(round(self.max_total_s))}_t{int(self.max_tokens)}_d{int(self.max_decisions)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heuristic vs model planner backend on same UIDs/budgets")
    parser.add_argument("--pov-json-dir", required=True)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="20/50/4,60/200/12")
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--queries-total", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--planner-a", default="heuristic", choices=["heuristic", "model", "auto"])
    parser.add_argument("--planner-b", default="model", choices=["heuristic", "model", "auto"])
    parser.add_argument("--planner-b-provider", default="fake")
    parser.add_argument("--planner-b-model", default="fake-planner-v1")
    parser.add_argument("--planner-b-base-url", default=None)
    parser.add_argument("--planner-b-api-key-env", default=None)
    parser.add_argument("--planner-b-api-mode", choices=["auto", "responses", "chat"], default="auto")
    parser.add_argument("--retrieval-plan", default="baseline", choices=["baseline", "summary_then_token", "summary_then_decision", "summary_then_event"])
    parser.add_argument("--summary-topk", type=int, default=3)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def _to_float(v: Any) -> float | None:
    try:
        out = float(v)
    except Exception:
        return None
    if out != out:
        return None
    return float(out)


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        pieces = [x.strip() for x in token.split("/") if x.strip()]
        if len(pieces) != 3:
            raise ValueError(f"invalid budget: {token}")
        out.append(BudgetPoint(float(pieces[0]), int(pieces[1]), int(pieces[2])))
    if not out:
        raise ValueError("no budgets parsed")
    return out


def _uid_from_json_path(path: Path) -> str:
    stem = str(path.stem)
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_v03_decisions$", "", cleaned)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    m = re.search(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", cleaned)
    if m:
        return str(m.group(1)).lower()
    return cleaned.lower()


def _read_uids(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "")
        if "#" in text:
            text = text.split("#", 1)[0]
        text = text.strip()
        if not text:
            continue
        for token in re.split(r"[,\s]+", text):
            uid = str(token).strip().replace(".mp4", "").lower()
            if uid:
                out.append(uid)
    return out


def _select_jsons(json_dir: Path, uids_file: str | None) -> tuple[list[Path], dict[str, Any]]:
    all_json = sorted(json_dir.glob("*_decisions.json"))
    if not all_json:
        raise FileNotFoundError(f"no *_decisions.json under {json_dir}")
    by_uid = {_uid_from_json_path(p): p for p in all_json}
    dir_sample = sorted(by_uid.keys())[:5]
    if not uids_file:
        return list(by_uid.values()), {
            "selection_mode": "all_json",
            "uids_requested": len(by_uid),
            "uids_found": len(by_uid),
            "uids_missing_count": 0,
            "dir_uids_sample": dir_sample,
        }
    requested = _read_uids(Path(uids_file))
    selected = [by_uid[uid] for uid in requested if uid in by_uid]
    missing = [uid for uid in requested if uid not in by_uid]
    if not selected:
        raise RuntimeError("uids-file provided but no uid matched pov-json-dir")
    return selected, {
        "selection_mode": "uids_file",
        "uids_file_path": str(uids_file),
        "uids_requested": len(requested),
        "uids_found": len(selected),
        "uids_missing_count": len(missing),
        "uids_missing_sample": missing[:10],
        "dir_uids_sample": dir_sample,
    }


def _run(cmd: list[str], *, cwd: Path, commands_file: Path, log_prefix: Path) -> int:
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    with commands_file.open("a", encoding="utf-8") as f:
        f.write(f"# {datetime.now(timezone.utc).isoformat()}\n")
        f.write(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd) + "\n\n")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    (log_prefix.with_suffix(".stdout.log")).write_text(proc.stdout or "", encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
    return int(proc.returncode)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _variant_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    out = [r for r in rows if str(r.get("variant", "")).strip().lower() in {"", "full"}]
    return out or rows


def _mean_metric(rows: list[dict[str, str]], keys: tuple[str, ...]) -> float:
    vals: list[float] = []
    for row in rows:
        for key in keys:
            v = _to_float(row.get(key))
            if v is not None:
                vals.append(v)
                break
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _load_safety_rate(path: Path) -> float:
    if not path.exists():
        return float("nan")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            v = _to_float(payload.get("critical_fn_rate"))
            if v is not None:
                return float(v)
    except Exception:
        return float("nan")
    return float("nan")


def _collect_metrics(run_dir: Path, budgets: list[BudgetPoint], uid_paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for budget in budgets:
        for json_path in uid_paths:
            uid = _uid_from_json_path(json_path)
            uid_dir = run_dir / budget.tag / uid
            summary_rows = _variant_rows(uid_dir / "nlq_summary.csv")
            per_query_rows = _read_csv(uid_dir / "nlq_results.csv")
            planner_backend_vals = [str(r.get("planner_backend_used", "")) for r in per_query_rows]
            planner_fallback_vals = [1.0 if str(r.get("planner_fallback_reason", "")) else 0.0 for r in per_query_rows]
            planner_backend_model_rate = (
                float(sum(1.0 for x in planner_backend_vals if x == "model") / len(planner_backend_vals))
                if planner_backend_vals
                else float("nan")
            )
            planner_fallback_rate = (
                float(sum(planner_fallback_vals) / len(planner_fallback_vals)) if planner_fallback_vals else float("nan")
            )
            out[(uid, budget.key)] = {
                "status": "ok" if summary_rows else "missing",
                "mrr_strict": _mean_metric(summary_rows, ("mrr", "mrr_strict")),
                "top1_in_distractor_rate": _mean_metric(summary_rows, ("top1_in_distractor_rate", "top1_in_distractor")),
                "critical_fn_rate": _load_safety_rate(uid_dir / "safety_report.json"),
                "planner_fallback_rate": planner_fallback_rate,
                "planner_backend_used_rate": planner_backend_model_rate,
                "latency_p95_ms": float("nan"),
            }
    return out


def _safe_delta(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return float(b - a)


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], cols: list[str], summary: list[str]) -> None:
    lines = [
        "# Planner Backend Compare",
        "",
        *summary,
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    by_budget: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        sec = float(_to_float(row.get("budget_seconds")) or 0.0)
        by_budget.setdefault(sec, []).append(row)
    xs = sorted(by_budget.keys())
    if not xs:
        return out

    def _mean(col: str, sec: float) -> float:
        vals = [_to_float(r.get(col)) for r in by_budget.get(sec, [])]
        nums = [float(v) for v in vals if v is not None]
        return float(sum(nums) / len(nums)) if nums else 0.0

    fig1 = out_dir / "fig_planner_backend_delta"
    plt.figure(figsize=(8.0, 4.4))
    y_mrr = [_mean("delta_mrr_strict", x) for x in xs]
    y_dist = [_mean("delta_top1_in_distractor_rate", x) for x in xs]
    y_fb = [_mean("delta_planner_fallback_rate", x) for x in xs]
    plt.plot(xs, y_mrr, marker="o", label="delta_mrr_strict")
    plt.plot(xs, y_dist, marker="s", label="delta_top1_in_distractor_rate")
    plt.plot(xs, y_fb, marker="^", label="delta_planner_fallback_rate")
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("budget_seconds")
    plt.ylabel("delta (B-A)")
    plt.title("Planner Backend Delta")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig1.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()

    fig2 = out_dir / "fig_planner_backend_tradeoff"
    plt.figure(figsize=(8.0, 4.4))
    xa = [_mean("top1_in_distractor_rate_a", x) for x in xs]
    xb = [_mean("top1_in_distractor_rate_b", x) for x in xs]
    ya = [_mean("mrr_strict_a", x) for x in xs]
    yb = [_mean("mrr_strict_b", x) for x in xs]
    plt.scatter(xa, ya, marker="o", label="A-heuristic")
    plt.scatter(xb, yb, marker="x", label="B-model")
    plt.xlabel("top1_in_distractor_rate")
    plt.ylabel("mrr_strict")
    plt.title("Planner Backend Tradeoff")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig2.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()
    return out


def main() -> int:
    args = parse_args()
    json_dir = Path(args.pov_json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = _parse_budgets(args.budgets)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    selected_jsons, selection = _select_jsons(json_dir, args.uids_file)

    run_a = out_dir / "run_A"
    run_b = out_dir / "run_B"
    compare_dir = out_dir / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    compare_dir.mkdir(parents=True, exist_ok=True)
    commands_file = compare_dir / "commands.sh"
    commands_file.write_text("# run_planner_backend_compare commands\n\n", encoding="utf-8")

    variants = [
        ("A", str(args.planner_a), run_a),
        ("B", str(args.planner_b), run_b),
    ]
    for code, planner, run_dir in variants:
        run_dir.mkdir(parents=True, exist_ok=True)
        for budget in budgets:
            for json_path in selected_jsons:
                uid = _uid_from_json_path(json_path)
                uid_out = run_dir / budget.tag / uid
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "eval_nlq.py"),
                    "--json",
                    str(json_path),
                    "--out_dir",
                    str(uid_out),
                    "--mode",
                    str(args.mode),
                    "--n",
                    str(int(args.queries_total)),
                    "--seed",
                    str(int(args.seed)),
                    "--top-k",
                    str(int(args.top_k)),
                    "--planner-backend",
                    str(planner),
                    "--retrieval-plan",
                    str(args.retrieval_plan),
                    "--summary-topk",
                    str(int(args.summary_topk)),
                    "--budget-max-total-s",
                    str(float(budget.max_total_s)),
                    "--budget-max-tokens",
                    str(int(budget.max_tokens)),
                    "--budget-max-decisions",
                    str(int(budget.max_decisions)),
                ]
                if args.index_dir:
                    cand = Path(args.index_dir) / uid
                    cmd.extend(["--index", str(cand)])
                if code == "B":
                    cmd.extend([
                        "--planner-provider",
                        str(args.planner_b_provider),
                        "--planner-model",
                        str(args.planner_b_model),
                        "--planner-api-mode",
                        str(args.planner_b_api_mode),
                    ])
                    if args.planner_b_base_url:
                        cmd.extend(["--planner-base-url", str(args.planner_b_base_url)])
                    if args.planner_b_api_key_env:
                        cmd.extend(["--planner-api-key-env", str(args.planner_b_api_key_env)])
                rc = _run(cmd, cwd=ROOT, commands_file=commands_file, log_prefix=run_dir / "logs" / budget.tag / uid)
                if rc != 0:
                    return rc

    metrics_a = _collect_metrics(run_a, budgets, selected_jsons)
    metrics_b = _collect_metrics(run_b, budgets, selected_jsons)

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for json_path in selected_jsons:
            uid = _uid_from_json_path(json_path)
            key = (uid, budget.key)
            ma = dict(metrics_a.get(key, {}))
            mb = dict(metrics_b.get(key, {}))
            row = {
                "uid": uid,
                "budget_key": budget.key,
                "budget_seconds": float(budget.max_total_s),
                "status_a": str(ma.get("status", "missing")),
                "status_b": str(mb.get("status", "missing")),
                "planner_a": str(args.planner_a),
                "planner_b": str(args.planner_b),
                "mrr_strict_a": float(ma.get("mrr_strict", float("nan"))),
                "mrr_strict_b": float(mb.get("mrr_strict", float("nan"))),
                "delta_mrr_strict": _safe_delta(
                    float(ma.get("mrr_strict", float("nan"))),
                    float(mb.get("mrr_strict", float("nan"))),
                ),
                "top1_in_distractor_rate_a": float(ma.get("top1_in_distractor_rate", float("nan"))),
                "top1_in_distractor_rate_b": float(mb.get("top1_in_distractor_rate", float("nan"))),
                "delta_top1_in_distractor_rate": _safe_delta(
                    float(ma.get("top1_in_distractor_rate", float("nan"))),
                    float(mb.get("top1_in_distractor_rate", float("nan"))),
                ),
                "critical_fn_rate_a": float(ma.get("critical_fn_rate", float("nan"))),
                "critical_fn_rate_b": float(mb.get("critical_fn_rate", float("nan"))),
                "delta_critical_fn_rate": _safe_delta(
                    float(ma.get("critical_fn_rate", float("nan"))),
                    float(mb.get("critical_fn_rate", float("nan"))),
                ),
                "planner_fallback_rate_a": float(ma.get("planner_fallback_rate", float("nan"))),
                "planner_fallback_rate_b": float(mb.get("planner_fallback_rate", float("nan"))),
                "delta_planner_fallback_rate": _safe_delta(
                    float(ma.get("planner_fallback_rate", float("nan"))),
                    float(mb.get("planner_fallback_rate", float("nan"))),
                ),
            }
            rows.append(row)

    columns = [
        "uid",
        "budget_key",
        "budget_seconds",
        "status_a",
        "status_b",
        "planner_a",
        "planner_b",
        "mrr_strict_a",
        "mrr_strict_b",
        "delta_mrr_strict",
        "top1_in_distractor_rate_a",
        "top1_in_distractor_rate_b",
        "delta_top1_in_distractor_rate",
        "critical_fn_rate_a",
        "critical_fn_rate_b",
        "delta_critical_fn_rate",
        "planner_fallback_rate_a",
        "planner_fallback_rate_b",
        "delta_planner_fallback_rate",
    ]
    table_csv = tables_dir / "table_planner_backend_compare.csv"
    table_md = tables_dir / "table_planner_backend_compare.md"
    _write_csv(table_csv, rows, columns)
    summary_lines = [
        f"- selection_mode: {selection.get('selection_mode', 'unknown')}",
        f"- uids_total: {len(selected_jsons)}",
        f"- planner_a: {args.planner_a}",
        f"- planner_b: {args.planner_b}",
    ]
    _write_md(table_md, rows, columns, summary_lines)

    figure_paths = _make_figures(rows, figures_dir, formats)

    budgets_matched = sorted({str(r.get("budget_key", "")) for r in rows if str(r.get("budget_key", ""))})
    summary_rows: list[dict[str, Any]] = []
    for bkey in budgets_matched:
        b_rows = [r for r in rows if str(r.get("budget_key", "")) == bkey]
        def _m(col: str) -> float:
            vals = [_to_float(x.get(col)) for x in b_rows]
            nums = [float(v) for v in vals if v is not None]
            return float(sum(nums) / len(nums)) if nums else float("nan")
        summary_rows.append({
            "budget_key": bkey,
            "mrr_strict_a": _m("mrr_strict_a"),
            "mrr_strict_b": _m("mrr_strict_b"),
            "delta_mrr_strict": _m("delta_mrr_strict"),
            "top1_in_distractor_rate_a": _m("top1_in_distractor_rate_a"),
            "top1_in_distractor_rate_b": _m("top1_in_distractor_rate_b"),
            "delta_top1_in_distractor_rate": _m("delta_top1_in_distractor_rate"),
            "planner_fallback_rate_a": _m("planner_fallback_rate_a"),
            "planner_fallback_rate_b": _m("planner_fallback_rate_b"),
            "delta_planner_fallback_rate": _m("delta_planner_fallback_rate"),
        })

    compare_summary = {
        "selection": selection,
        "uids_total": len(selected_jsons),
        "budgets": [b.key for b in budgets],
        "budgets_matched": budgets_matched,
        "planner_a": str(args.planner_a),
        "planner_b": str(args.planner_b),
        "planner_b_provider": str(args.planner_b_provider),
        "planner_b_model": str(args.planner_b_model),
        "rows": len(rows),
        "per_budget": summary_rows,
    }
    compare_summary_path = compare_dir / "compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(json_dir),
            "uids_file": str(args.uids_file or ""),
            "mode": str(args.mode),
            "queries_total": int(args.queries_total),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "retrieval_plan": str(args.retrieval_plan),
            "summary_topk": int(args.summary_topk),
            "planner_a": str(args.planner_a),
            "planner_b": str(args.planner_b),
            "planner_b_provider": str(args.planner_b_provider),
            "planner_b_model": str(args.planner_b_model),
        },
        "selection": selection,
        "artifacts": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
            "compare_summary": str(compare_summary_path),
            "commands": str(commands_file),
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_path = compare_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Planner Backend Compare",
                "",
                "This compare runs the same UID/budget set with two planner backends:",
                f"- A: {args.planner_a}",
                f"- B: {args.planner_b} ({args.planner_b_provider}/{args.planner_b_model})",
                "",
                "Outputs:",
                "- tables/table_planner_backend_compare.csv",
                "- tables/table_planner_backend_compare.md",
                "- figures/fig_planner_backend_delta.(png/pdf)",
                "- figures/fig_planner_backend_tradeoff.(png/pdf)",
                "- compare_summary.json",
                "- snapshot.json",
                "- commands.sh",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved_run_A={run_a}")
    print(f"saved_run_B={run_b}")
    print(f"saved_compare={compare_dir}")
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
