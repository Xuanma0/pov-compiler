from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.budget_policy import BudgetSpec, parse_budget_keys
from pov_compiler.streaming.runner import StreamingConfig, run_streaming


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming budget sweep (fixed/adaptive) over selected UIDs")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", required=True, help='e.g. "20/50/4,40/100/8,60/200/12"')
    parser.add_argument("--step-s", type=float, default=8.0)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--policy", choices=["fixed", "adaptive"], default="fixed")
    _parse_bool_with_neg(parser, "context-use-repo", default=False)
    parser.add_argument("--repo-read-policy", default="budgeted_topk")
    parser.add_argument("--repo-budget", default=None)
    parser.add_argument("--adaptive-gates-json", default=None)
    parser.add_argument("--adaptive-targets-json", default=None)
    _parse_bool_with_neg(parser, "strict-uids", default=True)
    parser.add_argument("--allow-fallback-all-uids", action="store_true")
    parser.add_argument("--formats", default="png,pdf")
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


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _parse_json_dict(raw: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if raw is None:
        return dict(default)
    text = str(raw).strip()
    if not text:
        return dict(default)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    p = Path(text)
    if p.exists():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return dict(default)


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
    lines = [
        "# Streaming Budget Sweep",
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_quality_figure(rows: list[dict[str, Any]], out_prefix: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda r: float(r.get("budget_seconds", 0.0)))
    xs = [float(r.get("budget_seconds", 0.0)) for r in sorted_rows]
    ys = [float(_to_float(r.get("hit@k_strict")) or 0.0) for r in sorted_rows]
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("hit@k_strict")
    plt.title("Streaming Quality vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    out: list[str] = []
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = out_prefix.with_suffix(f".{ext}")
        plt.savefig(p)
        out.append(str(p))
    plt.close()
    return out


def _collect_uid_metric_rows(
    *,
    uid: str,
    json_path: Path,
    budgets: list[BudgetSpec],
    step_s: float,
    mode: str,
    top_k: int,
    policy: str,
    adaptive_gates: dict[str, Any],
    adaptive_targets: dict[str, Any],
    context_use_repo: bool,
    repo_read_policy: str,
    repo_budget: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if policy == "adaptive":
        payload = run_streaming(
            json_path,
            config=StreamingConfig(
                step_s=float(step_s),
                top_k=int(top_k),
                budgets=budgets,
                budget_policy="adaptive",
                policy_gates=adaptive_gates,
                policy_targets=adaptive_targets,
                context_use_repo=bool(context_use_repo),
                repo_read_policy=str(repo_read_policy),
                repo_budget=str(repo_budget) if repo_budget else None,
                allow_gt_fallback=False,
                nlq_mode=str(mode),
                mode="budgeted",
            ),
        )
        summary = dict(payload.get("summary", {}))
        step_rows = list(payload.get("step_rows", []))
        query_rows = list(payload.get("query_rows", []))
        last_step = step_rows[-1] if step_rows else {}
        rows.append(
            {
                "video_uid": uid,
                "budget_key": "adaptive",
                "budget_seconds": float(max(b.max_total_s for b in budgets)) if budgets else 0.0,
                "hit@k_strict": float(last_step.get("hit_at_k_strict", summary.get("hit_at_k_strict", 0.0))),
                "hit@1_strict": float(last_step.get("hit_at_1_strict", summary.get("hit_at_1_strict", 0.0))),
                "top1_in_distractor_rate": float(
                    last_step.get("top1_in_distractor_rate", summary.get("top1_in_distractor_rate", 0.0))
                ),
                "fp_rate": float(last_step.get("fp_rate", summary.get("fp_rate", 0.0))),
                "e2e_ms_p50": float(summary.get("e2e_latency_p50_ms", 0.0)),
                "e2e_ms_p95": float(summary.get("e2e_latency_p95_ms", 0.0)),
                "retrieval_ms_p50": float(summary.get("retrieval_latency_p50_ms", 0.0)),
                "retrieval_ms_p95": float(summary.get("retrieval_latency_p95_ms", 0.0)),
                "queries_total": int(summary.get("queries_total", 0)),
                "steps": int(summary.get("steps", 0)),
                "avg_trials_per_query": float(summary.get("avg_trials_per_query", 0.0)),
                "avg_chosen_budget_seconds": _mean(
                    [float(_to_float(r.get("chosen_budget_seconds")) or 0.0) for r in query_rows]
                ),
            }
        )
        return rows

    for budget in budgets:
        payload = run_streaming(
            json_path,
            config=StreamingConfig(
                step_s=float(step_s),
                top_k=int(top_k),
                budgets=[budget],
                budget_policy="fixed",
                fixed_budget=budget.key,
                context_use_repo=bool(context_use_repo),
                repo_read_policy=str(repo_read_policy),
                repo_budget=str(repo_budget) if repo_budget else None,
                allow_gt_fallback=False,
                nlq_mode=str(mode),
                mode="budgeted",
            ),
        )
        summary = dict(payload.get("summary", {}))
        step_rows = list(payload.get("step_rows", []))
        query_rows = list(payload.get("query_rows", []))
        last_step = step_rows[-1] if step_rows else {}
        rows.append(
            {
                "video_uid": uid,
                "budget_key": str(budget.key),
                "budget_seconds": float(budget.max_total_s),
                "hit@k_strict": float(last_step.get("hit_at_k_strict", summary.get("hit_at_k_strict", 0.0))),
                "hit@1_strict": float(last_step.get("hit_at_1_strict", summary.get("hit_at_1_strict", 0.0))),
                "top1_in_distractor_rate": float(
                    last_step.get("top1_in_distractor_rate", summary.get("top1_in_distractor_rate", 0.0))
                ),
                "fp_rate": float(last_step.get("fp_rate", summary.get("fp_rate", 0.0))),
                "e2e_ms_p50": float(summary.get("e2e_latency_p50_ms", 0.0)),
                "e2e_ms_p95": float(summary.get("e2e_latency_p95_ms", 0.0)),
                "retrieval_ms_p50": float(summary.get("retrieval_latency_p50_ms", 0.0)),
                "retrieval_ms_p95": float(summary.get("retrieval_latency_p95_ms", 0.0)),
                "queries_total": int(summary.get("queries_total", 0)),
                "steps": int(summary.get("steps", 0)),
                "avg_trials_per_query": float(summary.get("avg_trials_per_query", 0.0)),
                "avg_chosen_budget_seconds": _mean(
                    [float(_to_float(r.get("chosen_budget_seconds")) or 0.0) for r in query_rows]
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    json_dir = Path(args.json_dir)
    out_dir = Path(args.out_dir)
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = parse_budget_keys(args.budgets)
    if not budgets:
        print("error=no budgets parsed")
        return 2
    formats = [x.strip().lower() for x in str(args.formats).split(",") if x.strip()]
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

    default_gates = {
        "top1_in_distractor_rate": {"op": "<=", "value": 0.20},
        "fp_rate": {"op": "<=", "value": 0.20},
    }
    default_targets = {"hit_at_k_strict": {"op": ">=", "value": 0.60}}
    adaptive_gates = _parse_json_dict(args.adaptive_gates_json, default_gates)
    adaptive_targets = _parse_json_dict(args.adaptive_targets_json, default_targets)

    per_uid_rows: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    for uid in selected_uids:
        uid_rows = _collect_uid_metric_rows(
            uid=uid,
            json_path=discovered[uid],
            budgets=budgets,
            step_s=float(args.step_s),
            mode=str(args.mode),
            top_k=int(args.top_k),
            policy=str(args.policy),
            adaptive_gates=adaptive_gates,
            adaptive_targets=adaptive_targets,
            context_use_repo=bool(args.context_use_repo),
            repo_read_policy=str(args.repo_read_policy),
            repo_budget=str(args.repo_budget) if args.repo_budget else None,
        )
        per_uid_rows.extend(uid_rows)
        run_meta.extend(
            [
                {
                    "uid": uid,
                    "budget_key": str(r.get("budget_key", "")),
                    "queries_total": int(r.get("queries_total", 0)),
                    "steps": int(r.get("steps", 0)),
                }
                for r in uid_rows
            ]
        )

    by_budget: dict[str, list[dict[str, Any]]] = {}
    for row in per_uid_rows:
        by_budget.setdefault(str(row["budget_key"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    budget_keys_order = [b.key for b in budgets] if str(args.policy) == "fixed" else ["adaptive"]
    for bkey in budget_keys_order:
        rows = by_budget.get(bkey, [])
        if not rows:
            continue
        out_row: dict[str, Any] = {
            "budget_key": bkey,
            "budget_seconds": _mean([float(_to_float(r.get("budget_seconds")) or 0.0) for r in rows]),
            "num_uids": len(selected_uids),
            "runs_total": len(rows),
            "policy": str(args.policy),
            "context_use_repo": bool(args.context_use_repo),
            "repo_read_policy": str(args.repo_read_policy),
            "repo_budget": str(args.repo_budget) if args.repo_budget else "",
        }
        metric_cols = [
            "hit@k_strict",
            "hit@1_strict",
            "top1_in_distractor_rate",
            "fp_rate",
            "e2e_ms_p50",
            "e2e_ms_p95",
            "retrieval_ms_p50",
            "retrieval_ms_p95",
            "queries_total",
            "steps",
            "avg_trials_per_query",
            "avg_chosen_budget_seconds",
        ]
        for col in metric_cols:
            vals = [float(_to_float(r.get(col)) or 0.0) for r in rows if _to_float(r.get(col)) is not None]
            if vals:
                out_row[col] = _mean(vals)
        aggregate_rows.append(out_row)

    metrics_csv = agg_dir / "metrics_by_budget.csv"
    metrics_md = agg_dir / "metrics_by_budget.md"
    _write_csv(metrics_csv, aggregate_rows)
    _write_md(metrics_md, aggregate_rows, selection)
    figure_paths = _make_quality_figure(
        aggregate_rows,
        fig_dir / "fig_streaming_quality_vs_budget_seconds",
        formats,
    )

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json_dir": str(json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "budgets": [b.key for b in budgets],
            "step_s": float(args.step_s),
            "mode": str(args.mode),
            "top_k": int(args.top_k),
            "policy": str(args.policy),
            "context_use_repo": bool(args.context_use_repo),
            "repo_read_policy": str(args.repo_read_policy),
            "repo_budget": str(args.repo_budget) if args.repo_budget else None,
            "adaptive_gates": adaptive_gates,
            "adaptive_targets": adaptive_targets,
            "strict_uids": bool(args.strict_uids),
            "allow_fallback_all_uids": bool(args.allow_fallback_all_uids),
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

    print(f"selection_mode={selection.get('selection_mode')}")
    print(f"selected_uids={len(selected_uids)}")
    print(f"budgets={len(budgets)}")
    print(f"context_use_repo={str(bool(args.context_use_repo)).lower()}")
    print(f"repo_read_policy={str(args.repo_read_policy)}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    for fig in figure_paths:
        print(f"saved_figure={fig}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
