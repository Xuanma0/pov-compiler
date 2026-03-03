from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
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
    parser = argparse.ArgumentParser(description="Compare baseline vs summary-first retrieval planning")
    parser.add_argument("--pov-json-dir", required=True)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="20/50/4,60/200/12")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
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


def _variant_full_rows(summary_csv: Path) -> list[dict[str, str]]:
    rows = _read_csv(summary_csv)
    out = [r for r in rows if str(r.get("variant", "")).strip().lower() in {"full", ""}]
    return out or rows


def _mean_metric(rows: list[dict[str, str]], keys: tuple[str, ...]) -> float:
    values: list[float] = []
    for row in rows:
        for key in keys:
            v = _to_float(row.get(key))
            if v is not None:
                values.append(v)
                break
    return float(sum(values) / len(values)) if values else float("nan")


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


def _collect_run_metrics(run_dir: Path, budgets: list[BudgetPoint], uid_paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for budget in budgets:
        for json_path in uid_paths:
            uid = _uid_from_json_path(json_path)
            uid_dir = run_dir / budget.tag / uid
            summary_rows = _variant_full_rows(uid_dir / "nlq_summary.csv")
            results_rows = _read_csv(uid_dir / "nlq_results.csv")
            metrics = {
                "status": "ok" if summary_rows else "missing",
                "mrr_strict": _mean_metric(summary_rows, ("mrr", "mrr_strict")),
                "distractor_rate": _mean_metric(summary_rows, ("top1_in_distractor_rate", "top1_in_distractor")),
                "hit_at_k_strict": _mean_metric(summary_rows, ("hit_at_k_strict",)),
                "critical_fn_rate": _load_safety_rate(uid_dir / "safety_report.json"),
                "latency_p95_ms": float("nan"),
                "stage0_hit_rate": _mean_metric(results_rows, ("stage0_summary_hit",)),
                "stage1_candidate_reduction_ratio": _mean_metric(results_rows, ("stage1_candidate_reduction_ratio",)),
            }
            out[(uid, budget.key)] = metrics
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], summary: list[str]) -> None:
    lines = [
        "# Repo Summary Retrieval Compare",
        "",
        *summary,
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
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

    def _mean_for(sec: float, col: str) -> float:
        vals = [_to_float(r.get(col)) for r in by_budget.get(sec, [])]
        nums = [float(v) for v in vals if v is not None]
        return float(sum(nums) / len(nums)) if nums else 0.0

    fig1 = out_dir / "fig_repo_summary_retrieval_quality_vs_budget_seconds"
    plt.figure(figsize=(8.0, 4.4))
    ya = [_mean_for(x, "mrr_strict_a") for x in xs]
    yb = [_mean_for(x, "mrr_strict_b") for x in xs]
    plt.plot(xs, ya, marker="o", label="baseline")
    plt.plot(xs, yb, marker="o", linestyle="--", label="summary_then_token")
    plt.xlabel("budget_seconds")
    plt.ylabel("mrr_strict")
    plt.title("Summary-first Retrieval Quality vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig1.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()

    fig2 = out_dir / "fig_repo_summary_retrieval_delta"
    plt.figure(figsize=(8.0, 4.4))
    y_delta_mrr = [_mean_for(x, "delta_mrr_strict") for x in xs]
    y_delta_dist = [_mean_for(x, "delta_distractor_rate") for x in xs]
    plt.plot(xs, y_delta_mrr, marker="o", label="delta_mrr_strict")
    plt.plot(xs, y_delta_dist, marker="s", label="delta_distractor_rate")
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("budget_seconds")
    plt.ylabel("delta (B-A)")
    plt.title("Summary-first Retrieval Delta")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig2.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()

    fig3 = out_dir / "fig_repo_summary_retrieval_candidate_scale"
    plt.figure(figsize=(8.0, 4.4))
    yca = [_mean_for(x, "stage1_candidate_reduction_ratio_a") for x in xs]
    ycb = [_mean_for(x, "stage1_candidate_reduction_ratio_b") for x in xs]
    plt.plot(xs, yca, marker="o", label="baseline")
    plt.plot(xs, ycb, marker="o", linestyle="--", label="summary_then_token")
    plt.xlabel("budget_seconds")
    plt.ylabel("stage1_candidate_reduction_ratio")
    plt.title("Summary-first Candidate Scale vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = fig3.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()
    return out


def _budget_stats(rows: list[dict[str, Any]], budget_key: str) -> dict[str, Any]:
    budget_rows = [r for r in rows if str(r.get("budget_key", "")) == str(budget_key)]
    if not budget_rows:
        return {}

    def _mean(col: str) -> float:
        vals = [_to_float(r.get(col)) for r in budget_rows]
        nums = [float(v) for v in vals if v is not None]
        return float(sum(nums) / len(nums)) if nums else 0.0

    return {
        "budget_key": str(budget_key),
        "budget_seconds": float(_to_float(budget_rows[0].get("budget_seconds")) or 0.0),
        "mrr_strict_a": _mean("mrr_strict_a"),
        "mrr_strict_b": _mean("mrr_strict_b"),
        "delta_mrr_strict": _mean("delta_mrr_strict"),
        "distractor_rate_a": _mean("distractor_rate_a"),
        "distractor_rate_b": _mean("distractor_rate_b"),
        "delta_distractor_rate": _mean("delta_distractor_rate"),
        "critical_fn_rate_a": _mean("critical_fn_rate_a"),
        "critical_fn_rate_b": _mean("critical_fn_rate_b"),
        "delta_critical_fn_rate": _mean("delta_critical_fn_rate"),
        "stage0_hit_rate_a": _mean("stage0_hit_rate_a"),
        "stage0_hit_rate_b": _mean("stage0_hit_rate_b"),
        "stage1_candidate_reduction_ratio_a": _mean("stage1_candidate_reduction_ratio_a"),
        "stage1_candidate_reduction_ratio_b": _mean("stage1_candidate_reduction_ratio_b"),
    }


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
    commands_file.write_text("# run_repo_summary_retrieval_compare commands\n\n", encoding="utf-8")

    variants = [("A", "baseline", run_a), ("B", "summary_then_token", run_b)]
    for code, plan, run_dir in variants:
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
                    str(int(args.n)),
                    "--seed",
                    str(int(args.seed)),
                    "--top-k",
                    str(int(args.top_k)),
                    "--budget-max-total-s",
                    str(float(budget.max_total_s)),
                    "--budget-max-tokens",
                    str(int(budget.max_tokens)),
                    "--budget-max-decisions",
                    str(int(budget.max_decisions)),
                    "--retrieval-plan",
                    str(plan),
                    "--summary-topk",
                    str(int(args.summary_topk)),
                ]
                if args.index_dir:
                    cmd.extend(["--index", str(Path(args.index_dir) / uid)])
                rc = _run(
                    cmd,
                    cwd=ROOT,
                    commands_file=commands_file,
                    log_prefix=compare_dir / "logs" / f"{code}_{budget.tag}_{uid}",
                )
                if rc != 0:
                    print(f"error=eval_failed variant={code} uid={uid} budget={budget.key}")
                    return rc

    metrics_a = _collect_run_metrics(run_a, budgets, selected_jsons)
    metrics_b = _collect_run_metrics(run_b, budgets, selected_jsons)
    all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()), key=lambda x: (x[0], float(x[1].split("/")[0])))
    rows: list[dict[str, Any]] = []
    for uid, budget_key in all_keys:
        a = metrics_a.get((uid, budget_key), {})
        b = metrics_b.get((uid, budget_key), {})
        sec = float(budget_key.split("/")[0]) if "/" in budget_key else 0.0
        row = {
            "uid": uid,
            "budget_key": budget_key,
            "budget_seconds": sec,
            "status_a": str(a.get("status", "missing")),
            "status_b": str(b.get("status", "missing")),
            "mrr_strict_a": a.get("mrr_strict", float("nan")),
            "mrr_strict_b": b.get("mrr_strict", float("nan")),
            "delta_mrr_strict": (
                float(b.get("mrr_strict")) - float(a.get("mrr_strict"))
                if _to_float(a.get("mrr_strict")) is not None and _to_float(b.get("mrr_strict")) is not None
                else float("nan")
            ),
            "distractor_rate_a": a.get("distractor_rate", float("nan")),
            "distractor_rate_b": b.get("distractor_rate", float("nan")),
            "delta_distractor_rate": (
                float(b.get("distractor_rate")) - float(a.get("distractor_rate"))
                if _to_float(a.get("distractor_rate")) is not None and _to_float(b.get("distractor_rate")) is not None
                else float("nan")
            ),
            "critical_fn_rate_a": a.get("critical_fn_rate", float("nan")),
            "critical_fn_rate_b": b.get("critical_fn_rate", float("nan")),
            "delta_critical_fn_rate": (
                float(b.get("critical_fn_rate")) - float(a.get("critical_fn_rate"))
                if _to_float(a.get("critical_fn_rate")) is not None and _to_float(b.get("critical_fn_rate")) is not None
                else float("nan")
            ),
            "latency_p95_ms_a": a.get("latency_p95_ms", float("nan")),
            "latency_p95_ms_b": b.get("latency_p95_ms", float("nan")),
            "delta_latency_p95_ms": float("nan"),
            "stage0_hit_rate_a": a.get("stage0_hit_rate", float("nan")),
            "stage0_hit_rate_b": b.get("stage0_hit_rate", float("nan")),
            "stage1_candidate_reduction_ratio_a": a.get("stage1_candidate_reduction_ratio", float("nan")),
            "stage1_candidate_reduction_ratio_b": b.get("stage1_candidate_reduction_ratio", float("nan")),
        }
        rows.append(row)

    columns = [
        "uid",
        "budget_key",
        "budget_seconds",
        "status_a",
        "status_b",
        "mrr_strict_a",
        "mrr_strict_b",
        "delta_mrr_strict",
        "distractor_rate_a",
        "distractor_rate_b",
        "delta_distractor_rate",
        "critical_fn_rate_a",
        "critical_fn_rate_b",
        "delta_critical_fn_rate",
        "latency_p95_ms_a",
        "latency_p95_ms_b",
        "delta_latency_p95_ms",
        "stage0_hit_rate_a",
        "stage0_hit_rate_b",
        "stage1_candidate_reduction_ratio_a",
        "stage1_candidate_reduction_ratio_b",
    ]
    table_csv = tables_dir / "table_repo_summary_retrieval_compare.csv"
    table_md = tables_dir / "table_repo_summary_retrieval_compare.md"
    _write_csv(table_csv, rows, columns)
    _write_md(
        table_md,
        rows,
        columns,
        [
            f"- plan_A: baseline",
            f"- plan_B: summary_then_token",
            f"- uids_total: {len(selected_jsons)}",
            f"- budgets: `{','.join(b.key for b in budgets)}`",
        ],
    )
    figure_paths = _make_figures(rows, figures_dir, formats=formats)

    per_budget = [_budget_stats(rows, b.key) for b in budgets]
    summary_payload = {
        "selection": selection,
        "uids_total": len(selected_jsons),
        "budgets": [b.key for b in budgets],
        "budgets_matched": len([x for x in per_budget if x]),
        "plan_A": "baseline",
        "plan_B": "summary_then_token",
        "per_budget": per_budget,
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": list(figure_paths),
        },
    }
    (compare_dir / "compare_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "budgets": [b.key for b in budgets],
            "mode": str(args.mode),
            "n": int(args.n),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "summary_topk": int(args.summary_topk),
        },
        "selection": selection,
        "paths": {
            "run_A": str(run_a),
            "run_B": str(run_b),
            "compare": str(compare_dir),
            "commands": str(commands_file),
        },
    }
    (compare_dir / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = compare_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Repo Summary Retrieval Compare",
                "",
                f"- run_A (baseline): `{run_a}`",
                f"- run_B (summary_then_token): `{run_b}`",
                f"- table: `{table_csv}`",
                f"- quality figure: `{figures_dir / 'fig_repo_summary_retrieval_quality_vs_budget_seconds.png'}`",
                f"- delta figure: `{figures_dir / 'fig_repo_summary_retrieval_delta.png'}`",
                f"- candidate scale: `{figures_dir / 'fig_repo_summary_retrieval_candidate_scale.png'}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"selection_mode={selection.get('selection_mode', '')}")
    print(f"selected_uids={len(selected_jsons)}")
    print(f"budgets={[b.key for b in budgets]}")
    print(f"saved_run_A={run_a}")
    print(f"saved_run_B={run_b}")
    print(f"saved_compare={compare_dir}")
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={compare_dir / 'snapshot.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
