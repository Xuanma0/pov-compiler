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
    parser = argparse.ArgumentParser(description="Sweep fixed-K streaming codec and report quality/safety/latency curves.")
    parser.add_argument("--json", default=None, help="Single *_v03_decisions.json path")
    parser.add_argument("--json_dir", default=None, help="Directory containing *_v03_decisions.json")
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--k-list", required=True, help='e.g. "4,8,16,32"')
    parser.add_argument("--budgets", default="20/50/4,40/100/8,60/200/12")
    parser.add_argument("--step-s", type=float, default=8.0)
    parser.add_argument(
        "--policy",
        choices=["fixed", "recommend", "adaptive", "safety_latency", "safety_latency_intervention"],
        default="safety_latency_intervention",
    )
    parser.add_argument("--fixed-budget", default="40/100/8")
    parser.add_argument("--recommend-dir", default=None)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
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
    uuid_re = re.compile(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
    match = uuid_re.search(cleaned) or uuid_re.search(stem)
    if match:
        return str(match.group(1)).lower()
    return cleaned


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
            uid = _normalize_uid(token)
            if uid:
                out.append(uid)
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _parse_ks(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            k = int(token)
        except Exception:
            continue
        if k > 0:
            out.append(k)
    uniq = sorted(set(out))
    return uniq


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
        "# Streaming Fixed-K Codec Sweep",
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


def _make_figures(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda r: int(r.get("codec_k", 0)))
    xs = [int(r.get("codec_k", 0)) for r in sorted_rows]
    quality = [float(_to_float(r.get("objective_combo")) or 0.0) for r in sorted_rows]
    strict = [float(_to_float(r.get("hit_at_k_strict")) or 0.0) for r in sorted_rows]
    safety = [float(_to_float(r.get("safety_critical_fn_rate")) or 0.0) for r in sorted_rows]
    latency_p50 = [float(_to_float(r.get("e2e_ms_p50")) or 0.0) for r in sorted_rows]
    latency_p95 = [float(_to_float(r.get("e2e_ms_p95")) or 0.0) for r in sorted_rows]
    out: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "fig_streaming_quality_vs_k"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, quality, marker="o", label="objective_combo")
    plt.plot(xs, strict, marker="s", linestyle="--", label="hit@k_strict")
    plt.xlabel("K")
    plt.ylabel("Quality")
    plt.title("Streaming Quality vs Fixed K")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        fp = p1.with_suffix(f".{ext}")
        plt.savefig(fp)
        out.append(str(fp))
    plt.close()

    p2 = out_dir / "fig_streaming_safety_vs_k"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, safety, marker="o")
    plt.xlabel("K")
    plt.ylabel("critical_fn_rate")
    plt.title("Streaming Safety vs Fixed K")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        fp = p2.with_suffix(f".{ext}")
        plt.savefig(fp)
        out.append(str(fp))
    plt.close()

    p3 = out_dir / "fig_streaming_latency_vs_k"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, latency_p50, marker="o", label="e2e_p50")
    plt.plot(xs, latency_p95, marker="s", linestyle="--", label="e2e_p95")
    plt.xlabel("K")
    plt.ylabel("Latency (ms)")
    plt.title("Streaming Latency vs Fixed K")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        fp = p3.with_suffix(f".{ext}")
        plt.savefig(fp)
        out.append(str(fp))
    plt.close()
    return out


def main() -> int:
    args = parse_args()
    if not args.json and not args.json_dir:
        print("error=one of --json/--json_dir is required")
        return 2
    ks = _parse_ks(args.k_list)
    if not ks:
        print("error=empty k-list")
        return 2
    budgets = parse_budget_keys(args.budgets)
    if not budgets:
        print("error=no budgets parsed")
        return 2
    out_dir = Path(args.out_dir)
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.json:
        p = Path(args.json)
        if not p.exists():
            print(f"error=json_not_found {p}")
            return 2
        uid = _normalize_uid(uid_from_pov_json_path(p))
        discovered = {uid: p}
    else:
        discovered = _discover_json_by_uid(Path(args.json_dir))
    if not discovered:
        print("error=no json discovered")
        return 2

    dir_uids = sorted(discovered.keys())
    selection: dict[str, Any] = {
        "selection_mode": "all_json",
        "uids_file_path": None,
        "uids_requested": 0,
        "uids_found": len(dir_uids),
        "uids_missing_count": 0,
        "uids_missing_sample": [],
        "dir_uids_sample": dir_uids[:5],
    }
    if args.uids_file:
        requested = _read_uids_file(Path(args.uids_file))
        selected_uids: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for uid in requested:
            if uid in discovered:
                if uid not in seen:
                    selected_uids.append(uid)
                    seen.add(uid)
            else:
                missing.append(uid)
        selection.update(
            {
                "selection_mode": "uids_file",
                "uids_file_path": str(args.uids_file),
                "uids_requested": len(requested),
                "uids_found": len(selected_uids),
                "uids_missing_count": len(missing),
                "uids_missing_sample": missing[:10],
            }
        )
        if bool(args.strict_uids):
            if not selected_uids or missing:
                print("error=uid selection failed under strict mode")
                print(f"uids_file_path={args.uids_file}")
                print(f"uids_requested={len(requested)}")
                print(f"uids_found={len(selected_uids)}")
                print(f"uids_missing_count={len(missing)}")
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

    run_rows: list[dict[str, Any]] = []
    for k in ks:
        for uid in selected_uids:
            path = discovered[uid]
            payload = run_streaming(
                path,
                config=StreamingConfig(
                    step_s=float(args.step_s),
                    top_k=int(args.top_k),
                    queries=[str(x) for x in list(args.query or []) if str(x).strip()],
                    budgets=budgets,
                    budget_policy=str(args.policy),
                    fixed_budget=str(args.fixed_budget),
                    recommend_dir=args.recommend_dir,
                    allow_gt_fallback=False,
                    nlq_mode=str(args.mode),
                    nlq_seed=int(args.seed),
                    mode="budgeted",
                    codec_name="fixed_k",
                    codec_k=int(k),
                ),
            )
            summary = dict(payload.get("summary", {}))
            step_rows = list(payload.get("step_rows", []))
            run_rows.append(
                {
                    "video_uid": uid,
                    "codec_k": int(k),
                    "policy": str(args.policy),
                    "hit_at_k_strict": float(summary.get("hit_at_k_strict", 0.0)),
                    "mrr": float(summary.get("mrr", 0.0)),
                    "top1_in_distractor_rate": float(summary.get("top1_in_distractor_rate", 0.0)),
                    "safety_critical_fn_rate": float(summary.get("safety_critical_fn_rate", 0.0)),
                    "e2e_ms_p50": float(summary.get("e2e_latency_p50_ms", 0.0)),
                    "e2e_ms_p95": float(summary.get("e2e_latency_p95_ms", 0.0)),
                    "retrieval_ms_p50": float(summary.get("retrieval_latency_p50_ms", 0.0)),
                    "retrieval_ms_p95": float(summary.get("retrieval_latency_p95_ms", 0.0)),
                    "avg_trials_per_query": float(summary.get("avg_trials_per_query", 0.0)),
                    "steps": int(summary.get("steps", 0)),
                    "queries_total": int(summary.get("queries_total", 0)),
                    "items_written_mean": _mean([float(_to_float(r.get("items_written")) or 0.0) for r in step_rows]),
                    "candidates_in_step_mean": _mean(
                        [float(_to_float(r.get("candidates_in_step")) or 0.0) for r in step_rows]
                    ),
                    "mean_item_score": _mean([float(_to_float(r.get("mean_item_score")) or 0.0) for r in step_rows]),
                }
            )

    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in run_rows:
        by_k.setdefault(int(row["codec_k"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for k in sorted(by_k.keys()):
        rows = by_k[k]
        out = {
            "codec_k": int(k),
            "num_uids": int(len(selected_uids)),
            "policy": str(args.policy),
            "hit_at_k_strict": _mean([float(r["hit_at_k_strict"]) for r in rows]),
            "mrr": _mean([float(r["mrr"]) for r in rows]),
            "top1_in_distractor_rate": _mean([float(r["top1_in_distractor_rate"]) for r in rows]),
            "safety_critical_fn_rate": _mean([float(r["safety_critical_fn_rate"]) for r in rows]),
            "e2e_ms_p50": _mean([float(r["e2e_ms_p50"]) for r in rows]),
            "e2e_ms_p95": _mean([float(r["e2e_ms_p95"]) for r in rows]),
            "retrieval_ms_p50": _mean([float(r["retrieval_ms_p50"]) for r in rows]),
            "retrieval_ms_p95": _mean([float(r["retrieval_ms_p95"]) for r in rows]),
            "avg_trials_per_query": _mean([float(r["avg_trials_per_query"]) for r in rows]),
            "items_written_mean": _mean([float(r["items_written_mean"]) for r in rows]),
            "candidates_in_step_mean": _mean([float(r["candidates_in_step_mean"]) for r in rows]),
            "mean_item_score": _mean([float(r["mean_item_score"]) for r in rows]),
            "queries_total": _mean([float(r["queries_total"]) for r in rows]),
            "steps": _mean([float(r["steps"]) for r in rows]),
        }
        out["objective_combo"] = float(
            out["hit_at_k_strict"] - out["top1_in_distractor_rate"] - (0.5 * out["safety_critical_fn_rate"])
        )
        aggregate_rows.append(out)

    metrics_csv = agg_dir / "metrics_by_k.csv"
    metrics_md = agg_dir / "metrics_by_k.md"
    _write_csv(metrics_csv, aggregate_rows)
    _write_md(metrics_md, aggregate_rows, selection)
    formats = [x.strip().lower() for x in str(args.formats).split(",") if x.strip()]
    figure_paths = _make_figures(aggregate_rows, fig_dir, formats)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json": str(args.json) if args.json else None,
            "json_dir": str(args.json_dir) if args.json_dir else None,
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "k_list": [int(x) for x in ks],
            "budgets": [b.key for b in budgets],
            "step_s": float(args.step_s),
            "policy": str(args.policy),
            "fixed_budget": str(args.fixed_budget),
            "recommend_dir": str(args.recommend_dir) if args.recommend_dir else None,
            "queries": [str(x) for x in list(args.query or []) if str(x).strip()],
            "mode": str(args.mode),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "strict_uids": bool(args.strict_uids),
        },
        "selection": selection,
        "selected_uids": list(selected_uids),
        "outputs": {
            "metrics_by_k_csv": str(metrics_csv),
            "metrics_by_k_md": str(metrics_md),
            "figures": figure_paths,
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selection_mode={selection.get('selection_mode')}")
    print(f"selected_uids={len(selected_uids)}")
    print(f"k_points={len(ks)}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    for fp in figure_paths:
        print(f"saved_figure={fp}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
