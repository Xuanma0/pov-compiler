from __future__ import annotations

import argparse
import csv
import json
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
        s = int(round(float(self.max_total_s)))
        return f"{s}/{int(self.max_tokens)}/{int(self.max_decisions)}"


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLQ evaluation across explicit budget points")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--budgets", required=True, help='e.g. "60/200/12,40/100/8,20/50/4"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    _parse_bool_with_neg(parser, "allow-gt-fallback", default=False)
    _parse_bool_with_neg(parser, "hard-constraints", default=True)
    _parse_bool_with_neg(parser, "safety-report", default=True)
    _parse_bool_with_neg(parser, "strict-uids", default=True)
    parser.add_argument("--allow-fallback-all-uids", action="store_true")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--eval-script", default=str(ROOT / "scripts" / "eval_nlq.py"))
    return parser.parse_args()


def _normalize_uid(text: str) -> str:
    token = str(text).replace("\ufeff", "").strip()
    if not token:
        return ""
    if token.lower().endswith(".mp4"):
        token = token[:-4]
    return token.strip().lower()


def uid_from_pov_json_path(p: Path) -> str:
    stem = p.stem
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    cleaned = re.sub(r"(?i)_v\d+_token$", "", cleaned)
    cleaned = re.sub(r"(?i)_token$", "", cleaned)
    uuid_re = re.compile(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
    match = uuid_re.search(cleaned) or uuid_re.search(stem)
    if match:
        return str(match.group(1)).lower()
    return cleaned


def _read_uids_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "")
        if "#" in text:
            text = text.split("#", 1)[0]
        text = text.strip()
        if not text:
            continue
        for token in re.split(r"[,\s]+", text):
            norm = _normalize_uid(token)
            if norm:
                out.append(norm)
    return out


def _discover_json_by_uid(json_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    files = sorted(json_dir.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(json_dir.glob("*.json"), key=lambda p: p.name.lower())
    for p in files:
        uid = _normalize_uid(uid_from_pov_json_path(p))
        if uid and uid not in out:
            out[uid] = p
    return out


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        chunks = [x.strip() for x in text.split("/") if x.strip()]
        if len(chunks) != 3:
            raise ValueError(f"invalid budget {text}")
        out.append(BudgetPoint(max_total_s=float(chunks[0]), max_tokens=int(chunks[1]), max_decisions=int(chunks[2])))
    if not out:
        raise ValueError("no budgets parsed")
    return out


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NLQ Budget Sweep",
        "",
        "## Selection",
        "",
        f"- selection_mode: {selection.get('selection_mode', '')}",
        f"- uids_file_path: {selection.get('uids_file_path', '')}",
        f"- uids_requested: {selection.get('uids_requested', 0)}",
        f"- uids_found: {selection.get('uids_found', 0)}",
        f"- uids_missing_count: {selection.get('uids_missing_count', 0)}",
        f"- uids_missing_sample: {selection.get('uids_missing_sample', [])}",
        f"- dir_uids_sample: {selection.get('dir_uids_sample', [])}",
        "",
    ]
    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _extract_uid_metrics(uid_dir: Path, include_safety: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    results_rows = _read_csv(uid_dir / "nlq_results.csv")
    full_rows = [r for r in results_rows if str(r.get("variant", "")) == "full"]
    out["nlq_full_hit_at_k_strict"] = _mean([_to_float(r.get("hit_at_k_strict")) or 0.0 for r in full_rows])
    out["nlq_full_hit_at_1_strict"] = _mean([_to_float(r.get("hit_at_1_strict")) or 0.0 for r in full_rows])
    out["nlq_full_top1_in_distractor_rate"] = _mean(
        [_to_float(r.get("top1_in_distractor")) or 0.0 for r in full_rows]
    )
    out["nlq_full_fp_rate"] = out["nlq_full_top1_in_distractor_rate"]
    out["nlq_full_mrr"] = _mean([_to_float(r.get("mrr")) or 0.0 for r in full_rows])
    if include_safety:
        safety_path = uid_dir / "safety_report.json"
        safety_rate = 0.0
        if safety_path.exists():
            try:
                payload = json.loads(safety_path.read_text(encoding="utf-8"))
                var_stats = payload.get("variant_stats", {}) if isinstance(payload, dict) else {}
                if isinstance(var_stats, dict) and isinstance(var_stats.get("full"), dict):
                    safety_rate = float(var_stats["full"].get("critical_fn_rate", 0.0))
                else:
                    safety_rate = float(payload.get("critical_fn_rate", 0.0)) if isinstance(payload, dict) else 0.0
            except Exception:
                safety_rate = 0.0
        out["safety_critical_fn_rate_full"] = float(safety_rate)
    return out


def _make_figures(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: float(r.get("budget_seconds", 0.0)))
    xs = [float(r.get("budget_seconds", 0.0)) for r in sorted_rows]
    y_quality = [float(r.get("nlq_full_mrr", 0.0)) for r in sorted_rows]
    y_hitk = [float(r.get("nlq_full_hit_at_k_strict", 0.0)) for r in sorted_rows]
    y_fp = [float(r.get("nlq_full_fp_rate", 0.0)) for r in sorted_rows]

    out_paths: list[str] = []

    p1 = out_dir / "fig_nlq_quality_vs_budget_seconds"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, y_quality, marker="o")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("nlq_full_mrr")
    plt.title("NLQ Quality vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        path = p1.with_suffix(f".{ext}")
        plt.savefig(path)
        out_paths.append(str(path))
    plt.close()

    p2 = out_dir / "fig_nlq_strict_vs_budget_seconds"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, y_hitk, marker="o", label="nlq_full_hit_at_k_strict")
    plt.plot(xs, y_fp, marker="o", label="nlq_full_fp_rate")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("Metric")
    plt.title("NLQ Strict Metrics vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        path = p2.with_suffix(f".{ext}")
        plt.savefig(path)
        out_paths.append(str(path))
    plt.close()

    return out_paths


def main() -> int:
    args = parse_args()
    json_dir = Path(args.json_dir)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    per_budget_root = out_dir / "per_budget"
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_budget_root.mkdir(parents=True, exist_ok=True)

    budgets = _parse_budgets(args.budgets)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    discovered = _discover_json_by_uid(json_dir)
    if not discovered:
        print(f"error=no json found in {json_dir}")
        return 2
    dir_uids = sorted(discovered.keys())

    selection: dict[str, Any] = {
        "selection_mode": "all_json",
        "uids_file_path": None,
        "uids_requested": 0,
        "uids_found": 0,
        "uids_missing_count": 0,
        "uids_missing_sample": [],
        "dir_uids_sample": dir_uids[:5],
    }
    if args.uids_file:
        uids_file = Path(args.uids_file)
        requested = _read_uids_file(uids_file)
        selected_uids: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for uid in requested:
            norm = _normalize_uid(uid)
            if norm in discovered:
                if norm not in seen:
                    selected_uids.append(norm)
                    seen.add(norm)
            else:
                missing.append(uid)
        selection.update(
            {
                "selection_mode": "uids_file",
                "uids_file_path": str(uids_file),
                "uids_requested": len(requested),
                "uids_found": len(selected_uids),
                "uids_missing_count": len(missing),
                "uids_missing_sample": missing[:10],
            }
        )
        if bool(args.strict_uids):
            if not selected_uids or missing:
                print("error=uid selection failed under strict mode")
                print(f"uids_file_path={uids_file}")
                print(f"uids_requested={len(requested)}")
                print(f"uids_found={len(selected_uids)}")
                print(f"uids_missing_count={len(missing)}")
                print(f"uids_requested_sample={requested[:10]}")
                print(f"uids_missing_sample={missing[:10]}")
                print(f"dir_uid_sample={dir_uids[:5]}")
                return 2
        else:
            if not selected_uids:
                if bool(args.allow_fallback_all_uids):
                    selected_uids = list(dir_uids)
                    selection["selection_mode"] = "fallback_all_json"
                else:
                    print("error=no uid matched and fallback is disabled")
                    return 2
            elif missing:
                selection["selection_mode"] = "uids_file_partial"
    else:
        selected_uids = list(dir_uids)
        selection["uids_found"] = len(selected_uids)

    if bool(args.allow_fallback_all_uids) and bool(args.strict_uids):
        print("error=--allow-fallback-all-uids requires --no-strict-uids")
        return 2
    if not selected_uids:
        print("error=no selected uids")
        return 2

    print(f"selection_mode={selection.get('selection_mode')}")
    print(f"selected_uids={len(selected_uids)}")
    print(f"uids_file_path={selection.get('uids_file_path')}")
    print(f"uids_missing_count={selection.get('uids_missing_count')}")

    eval_script = Path(args.eval_script)
    if not eval_script.exists():
        print(f"error=eval script not found: {eval_script}")
        return 2

    run_rows: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    for budget in budgets:
        budget_dir = per_budget_root / budget.key.replace("/", "_")
        for uid in selected_uids:
            json_path = discovered[uid]
            uid_out = budget_dir / uid
            uid_out.mkdir(parents=True, exist_ok=True)
            index_prefix = index_dir / uid

            cmd = [
                sys.executable,
                str(eval_script),
                "--json",
                str(json_path),
                "--index",
                str(index_prefix),
                "--out_dir",
                str(uid_out),
                "--mode",
                str(args.mode),
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
            ]
            if bool(args.allow_gt_fallback):
                cmd.append("--allow-gt-fallback")
            else:
                cmd.append("--no-allow-gt-fallback")
            cmd.extend(["--hard-constraints", "on" if bool(args.hard_constraints) else "off"])
            cmd.append("--no-safety-gate")

            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
            (uid_out / "runner.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (uid_out / "runner.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

            row: dict[str, Any] = {
                "video_uid": uid,
                "budget_key": budget.key,
                "budget_seconds": float(budget.max_total_s),
                "budget_max_total_s": float(budget.max_total_s),
                "budget_max_tokens": int(budget.max_tokens),
                "budget_max_decisions": int(budget.max_decisions),
                "returncode": int(proc.returncode),
            }
            if proc.returncode == 0:
                row.update(_extract_uid_metrics(uid_out, include_safety=bool(args.safety_report)))
            run_rows.append(row)
            run_meta.append(
                {
                    "uid": uid,
                    "budget_key": budget.key,
                    "returncode": int(proc.returncode),
                    "stdout_log": str(uid_out / "runner.stdout.log"),
                    "stderr_log": str(uid_out / "runner.stderr.log"),
                    "out_dir": str(uid_out),
                }
            )

    by_budget: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        by_budget.setdefault(str(row["budget_key"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for budget in budgets:
        key = budget.key
        rows = by_budget.get(key, [])
        agg: dict[str, Any] = {
            "budget_key": key,
            "budget_seconds": float(budget.max_total_s),
            "budget_max_total_s": float(budget.max_total_s),
            "budget_max_tokens": int(budget.max_tokens),
            "budget_max_decisions": int(budget.max_decisions),
            "num_uids": len(selected_uids),
            "runs_total": len(rows),
            "runs_ok": sum(1 for r in rows if int(r.get("returncode", 1)) == 0),
        }
        numeric_keys = [
            "nlq_full_hit_at_k_strict",
            "nlq_full_hit_at_1_strict",
            "nlq_full_top1_in_distractor_rate",
            "nlq_full_fp_rate",
            "nlq_full_mrr",
            "safety_critical_fn_rate_full",
        ]
        for nk in numeric_keys:
            vals = [float(r[nk]) for r in rows if isinstance(r.get(nk), (int, float))]
            if vals:
                agg[nk] = _mean(vals)
        aggregate_rows.append(agg)

    metrics_csv = agg_dir / "metrics_by_budget.csv"
    metrics_md = agg_dir / "metrics_by_budget.md"
    _write_csv(metrics_csv, aggregate_rows)
    _write_md(metrics_md, aggregate_rows, selection)
    figure_paths = _make_figures(aggregate_rows, fig_dir, formats)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json_dir": str(json_dir),
            "index_dir": str(index_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "mode": str(args.mode),
            "budgets": [b.__dict__ for b in budgets],
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "allow_gt_fallback": bool(args.allow_gt_fallback),
            "hard_constraints": bool(args.hard_constraints),
            "safety_report": bool(args.safety_report),
            "strict_uids": bool(args.strict_uids),
            "allow_fallback_all_uids": bool(args.allow_fallback_all_uids),
            "eval_script": str(eval_script),
        },
        "selection": selection,
        "selected_uids": list(selected_uids),
        "runs": run_meta,
        "outputs": {
            "metrics_by_budget_csv": str(metrics_csv),
            "metrics_by_budget_md": str(metrics_md),
            "figures": figure_paths,
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"budgets={len(budgets)}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

