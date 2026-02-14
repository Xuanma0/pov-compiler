from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config

WEIGHT_FIELDS = [
    "w_trigger",
    "w_action",
    "w_constraint",
    "w_outcome",
    "w_evidence",
    "w_semantic",
]


def _parse_float_list(raw: str | None) -> list[float]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    out: list[float] = []
    for item in text.split(","):
        part = item.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_grid(raw: str | None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if raw is None:
        return out
    text = str(raw).strip()
    if not text:
        return out
    for seg in text.split(";"):
        s = seg.strip()
        if not s or "=" not in s:
            continue
        key, val = s.split("=", 1)
        k = key.strip()
        if k not in WEIGHT_FIELDS:
            continue
        vals = _parse_float_list(val)
        if vals:
            out[k] = vals
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    except Exception:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def _render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _discover_inputs(run_dir: Path | None, json_path: Path | None, index_prefix: Path | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if run_dir is not None:
        json_dir = run_dir / "json"
        cache_dir = run_dir / "cache"
        if not json_dir.exists():
            return []
        for p in sorted(json_dir.glob("*_v03_decisions.json")):
            uid = p.stem.replace("_v03_decisions", "")
            prefix = cache_dir / uid
            if not (Path(str(prefix) + ".index.npz").exists() and Path(str(prefix) + ".index_meta.json").exists()):
                prefix = None
            items.append(
                {
                    "video_uid": uid,
                    "json_path": p,
                    "index_prefix": prefix,
                }
            )
        return items

    if json_path is None:
        return []
    uid = json_path.stem.replace("_v03_decisions", "")
    items.append(
        {
            "video_uid": uid,
            "json_path": json_path,
            "index_prefix": index_prefix,
        }
    )
    return items


def _collect_metrics(eval_out_dir: Path) -> dict[str, Any]:
    rows_path = eval_out_dir / "nlq_results.csv"
    safety_path = eval_out_dir / "safety_report.json"
    strict_values: list[float] = []
    hit1_values: list[float] = []
    mrr_values: list[float] = []
    mrr_strict_values: list[float] = []
    distractor_values: list[float] = []
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("variant", "")) != "full":
                    continue
                strict = float(row.get("hit_at_k_strict", 0.0) or 0.0)
                hit1 = float(row.get("hit_at_1_strict", 0.0) or 0.0)
                mrr = float(row.get("mrr", 0.0) or 0.0)
                dist = float(row.get("top1_in_distractor", 0.0) or 0.0)
                strict_values.append(strict)
                hit1_values.append(hit1)
                mrr_values.append(mrr)
                mrr_strict_values.append(mrr * strict)
                distractor_values.append(dist)

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    critical_fn_rate = 0.0
    reason_counts: dict[str, int] = {}
    if safety_path.exists():
        try:
            payload = json.loads(safety_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                critical_fn_rate = float(payload.get("critical_fn_rate", 0.0) or 0.0)
                rc = payload.get("reason_counts", {})
                if isinstance(rc, dict):
                    reason_counts = {str(k): int(v) for k, v in rc.items()}
        except Exception:
            pass

    return {
        "strict_success_rate": _mean(strict_values),
        "hit_at_1_strict": _mean(hit1_values),
        "mrr": _mean(mrr_values),
        "mrr_strict": _mean(mrr_strict_values),
        "top1_in_distractor_rate": _mean(distractor_values),
        "critical_fn_rate": float(critical_fn_rate),
        "reason_counts": reason_counts,
        "rows_full": len(strict_values),
    }


def _objective(mrr_strict: float, distractor_rate: float, critical_fn_rate: float, alpha: float, beta: float) -> float:
    return float(mrr_strict - alpha * distractor_rate - beta * critical_fn_rate)


def _make_figures(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: int(r.get("weights_id", 0)))
    best = max(ordered, key=lambda r: float(r.get("objective", -1e9)))

    paths: list[str] = []
    # objective vs weights id
    x = [int(r.get("weights_id", 0)) for r in ordered]
    y = [float(r.get("objective", 0.0)) for r in ordered]
    p1 = out_dir / "fig_objective_vs_weights_id"
    plt.figure(figsize=(8.0, 4.2))
    plt.plot(x, y, marker="o")
    plt.scatter([int(best.get("weights_id", 0))], [float(best.get("objective", 0.0))], marker="*", s=150, label="best")
    plt.xlabel("weights_id")
    plt.ylabel("objective")
    plt.title("Reranker Sweep Objective")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # trade-off strict vs distractor
    p2 = out_dir / "fig_tradeoff_strict_vs_distractor"
    plt.figure(figsize=(7.4, 4.4))
    xs = [float(r.get("top1_in_distractor_rate", 0.0)) for r in ordered]
    ys = [float(r.get("mrr_strict", 0.0)) for r in ordered]
    sizes = [max(20.0, 400.0 * float(r.get("critical_fn_rate", 0.0) + 0.02)) for r in ordered]
    plt.scatter(xs, ys, s=sizes)
    plt.scatter(
        [float(best.get("top1_in_distractor_rate", 0.0))],
        [float(best.get("mrr_strict", 0.0))],
        marker="*",
        s=180,
        label="best",
    )
    plt.xlabel("top1_in_distractor_rate")
    plt.ylabel("mrr_strict")
    plt.title("Strict vs Distractor Trade-off")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()
    return paths


def _build_grid_candidates(base: WeightConfig, args: argparse.Namespace) -> list[WeightConfig]:
    grid = _parse_grid(args.grid)
    explicit_lists = {
        "w_trigger": _parse_float_list(args.w_trigger_list),
        "w_action": _parse_float_list(args.w_action_list),
        "w_constraint": _parse_float_list(args.w_constraint_list),
        "w_outcome": _parse_float_list(args.w_outcome_list),
        "w_evidence": _parse_float_list(args.w_evidence_list),
        "w_semantic": _parse_float_list(args.w_semantic_list),
    }
    values: dict[str, list[float]] = {}
    for key in WEIGHT_FIELDS:
        if key in grid and grid[key]:
            values[key] = list(grid[key])
        elif explicit_lists[key]:
            values[key] = list(explicit_lists[key])
        else:
            values[key] = [float(getattr(base, key))]

    candidates: list[WeightConfig] = []
    combos = itertools.product(*(values[k] for k in WEIGHT_FIELDS))
    max_cfg = max(1, int(args.max_configs))
    for i, combo in enumerate(combos, start=1):
        data = base.to_dict()
        for key, val in zip(WEIGHT_FIELDS, combo):
            data[key] = float(val)
        data["name"] = f"weights_{i:04d}"
        candidates.append(WeightConfig.from_dict(data))
        if len(candidates) >= max_cfg:
            break
    return candidates


def _build_random_candidates(base: WeightConfig, args: argparse.Namespace) -> list[WeightConfig]:
    rng = random.Random(int(args.seed))
    out = [WeightConfig.from_dict(base.to_dict())]
    out[0].name = "default"
    for i in range(max(0, int(args.trials) - 1)):
        data = base.to_dict()
        data["name"] = f"rand_{i + 1:04d}"
        data["w_trigger"] = round(rng.uniform(0.0, 1.2), 4)
        data["w_action"] = round(rng.uniform(0.0, 1.2), 4)
        data["w_constraint"] = round(rng.uniform(0.0, 1.2), 4)
        data["w_outcome"] = round(rng.uniform(0.0, 1.2), 4)
        data["w_evidence"] = round(rng.uniform(0.0, 1.2), 4)
        data["w_semantic"] = round(rng.uniform(0.6, 1.4), 4)
        out.append(WeightConfig.from_dict(data))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep decision-aligned reranker weights with strict+distractor objective")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", default=None, help="Run directory containing json/ and cache/")
    src.add_argument("--json", dest="json_path", default=None, help="Single *_v03_decisions.json input")
    parser.add_argument("--index", default=None, help="Single index prefix for --json mode")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--nlq-mode", default="hard_pseudo_nlq", choices=["hard_pseudo_nlq", "pseudo_nlq", "mock", "ego4d"])
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search", choices=["grid", "random"], default="grid")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--grid", default="", help="Grid string, e.g. w_trigger=0.2,0.4;w_action=0.3,0.6")
    parser.add_argument("--w-trigger-list", default="")
    parser.add_argument("--w-action-list", default="")
    parser.add_argument("--w-constraint-list", default="")
    parser.add_argument("--w-outcome-list", default="")
    parser.add_argument("--w-evidence-list", default="")
    parser.add_argument("--w-semantic-list", default="")
    parser.add_argument("--max-configs", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5, help="Penalty coefficient for top1_in_distractor_rate")
    parser.add_argument("--beta", type=float, default=0.5, help="Penalty coefficient for safety critical_fn_rate")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--eval-script", default=str(ROOT / "scripts" / "eval_nlq.py"))
    return parser.parse_args()


def _load_base_cfg(config_path: Path) -> WeightConfig:
    if not config_path.exists():
        return WeightConfig()
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            reranker = payload.get("reranker", {})
            if isinstance(reranker, dict):
                return resolve_weight_config(reranker)
    except Exception:
        pass
    return WeightConfig()


def _git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return str(res.stdout).strip()
    except Exception:
        pass
    return ""


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    run_root = out_dir / "runs"
    cfg_root = out_dir / "configs"
    agg_root = out_dir / "aggregate"
    fig_root = out_dir / "figures"
    logs_root = out_dir / "logs"
    for d in (out_dir, run_root, cfg_root, agg_root, fig_root, logs_root):
        d.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir) if args.run_dir else None
    json_path = Path(args.json_path) if args.json_path else None
    index_prefix = Path(args.index) if args.index else None
    eval_script = Path(args.eval_script)
    inputs = _discover_inputs(run_dir, json_path, index_prefix)
    if not inputs:
        print("error=no_inputs")
        return 2
    if not eval_script.exists():
        print(f"error=eval_script_missing path={eval_script}")
        return 2

    base_cfg = _load_base_cfg(Path(args.config))
    base_cfg.name = str(base_cfg.name or "default")
    if str(args.search).lower() == "random":
        candidates = _build_random_candidates(base_cfg, args)
    else:
        candidates = _build_grid_candidates(base_cfg, args)
        if not candidates:
            candidates = [base_cfg]
            candidates[0].name = "default"

    commands_path = logs_root / "commands.sh"
    commands_path.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    default_objective: float | None = None
    for idx, cfg in enumerate(candidates, start=1):
        cfg_id = f"weights_{idx:04d}"
        cfg_path = cfg_root / f"{cfg_id}.yaml"
        _write_yaml(cfg_path, cfg.to_dict())

        strict_values: list[float] = []
        hit1_values: list[float] = []
        mrr_values: list[float] = []
        mrr_strict_values: list[float] = []
        dist_values: list[float] = []
        crit_values: list[float] = []
        reason_totals: dict[str, int] = {}
        rc_all = 0

        for item in inputs:
            uid = str(item["video_uid"])
            per_out = run_root / cfg_id / uid
            per_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(eval_script),
                "--json",
                str(item["json_path"]),
                "--out_dir",
                str(per_out),
                "--mode",
                str(args.nlq_mode),
                "--seed",
                str(int(args.seed)),
                "--top-k",
                str(int(args.top_k)),
                "--rerank-cfg",
                str(cfg_path),
                "--no-allow-gt-fallback",
                "--no-safety-gate",
            ]
            if item.get("index_prefix") is not None:
                cmd.extend(["--index", str(item["index_prefix"])])
            with commands_path.open("a", encoding="utf-8") as f:
                f.write(f"# {cfg_id}::{uid}\n{_render_cmd(cmd)}\n\n")
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
            (logs_root / f"{cfg_id}__{uid}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (logs_root / f"{cfg_id}__{uid}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            rc_all = max(rc_all, int(proc.returncode))
            metrics = _collect_metrics(per_out)
            strict_values.append(float(metrics["strict_success_rate"]))
            hit1_values.append(float(metrics["hit_at_1_strict"]))
            mrr_values.append(float(metrics["mrr"]))
            mrr_strict_values.append(float(metrics["mrr_strict"]))
            dist_values.append(float(metrics["top1_in_distractor_rate"]))
            crit_values.append(float(metrics["critical_fn_rate"]))
            for k, v in dict(metrics.get("reason_counts", {})).items():
                reason_totals[str(k)] = reason_totals.get(str(k), 0) + int(v)

        def _mean(values: list[float]) -> float:
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        strict = _mean(strict_values)
        hit1 = _mean(hit1_values)
        mrr = _mean(mrr_values)
        mrr_strict = _mean(mrr_strict_values)
        dist = _mean(dist_values)
        crit = _mean(crit_values)
        obj = _objective(mrr_strict, dist, crit, alpha=float(args.alpha), beta=float(args.beta))
        if str(cfg.name) == "default":
            default_objective = float(obj)
        row = {
            "weights_id": idx,
            "cfg_id": cfg_id,
            "cfg_name": str(cfg.name),
            "cfg_hash": str(cfg.short_hash()),
            "objective": float(obj),
            "strict_success_rate": float(strict),
            "hit_at_1_strict": float(hit1),
            "mrr": float(mrr),
            "mrr_strict": float(mrr_strict),
            "top1_in_distractor_rate": float(dist),
            "critical_fn_rate": float(crit),
            "num_videos": len(inputs),
            "returncode": int(rc_all),
            "reason_counts_json": json.dumps(reason_totals, ensure_ascii=False, sort_keys=True),
        }
        for field in WEIGHT_FIELDS:
            row[field] = float(getattr(cfg, field))
        rows.append(row)
        print(
            f"weights_id={idx} cfg={cfg.name} objective={obj:.6f} "
            f"mrr_strict={mrr_strict:.4f} distractor={dist:.4f} critical_fn={crit:.4f}"
        )

    rows.sort(key=lambda r: (-float(r.get("objective", -1e9)), int(r.get("weights_id", 0))))
    best = rows[0]
    best_cfg = next((cfg for cfg in candidates if cfg.short_hash() == str(best["cfg_hash"])), candidates[0])
    if default_objective is None:
        default_objective = float(rows[-1]["objective"])

    metrics_csv = agg_root / "metrics_by_weights.csv"
    _write_csv(metrics_csv, rows)
    metrics_md = agg_root / "metrics_by_weights.md"
    md_lines = [
        "# Reranker Decision-Align Sweep",
        "",
        f"- inputs: {len(inputs)}",
        f"- nlq_mode: {args.nlq_mode}",
        f"- alpha: {float(args.alpha):.4f}",
        f"- beta: {float(args.beta):.4f}",
        f"- objective: mrr_strict - alpha*top1_in_distractor_rate - beta*critical_fn_rate",
        "",
        "| weights_id | cfg_name | objective | mrr_strict | top1_in_distractor_rate | critical_fn_rate | "
        "w_trigger | w_action | w_constraint | w_outcome | w_evidence | w_semantic |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {int(row['weights_id'])} | {row['cfg_name']} | {float(row['objective']):.6f} | "
            f"{float(row['mrr_strict']):.4f} | {float(row['top1_in_distractor_rate']):.4f} | "
            f"{float(row['critical_fn_rate']):.4f} | {float(row['w_trigger']):.3f} | {float(row['w_action']):.3f} | "
            f"{float(row['w_constraint']):.3f} | {float(row['w_outcome']):.3f} | "
            f"{float(row['w_evidence']):.3f} | {float(row['w_semantic']):.3f} |"
        )
    metrics_md.write_text("\n".join(md_lines), encoding="utf-8")

    best_cfg_path = out_dir / "best_weights.yaml"
    _write_yaml(best_cfg_path, best_cfg.to_dict())

    best_reason_counts = json.loads(str(best.get("reason_counts_json", "{}")))
    top_reason_pairs = sorted(best_reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:5]
    best_report = out_dir / "best_report.md"
    report_lines = [
        "# Best Reranker Weights Report",
        "",
        f"- best_cfg_name: {best_cfg.name}",
        f"- best_cfg_hash: {best_cfg.short_hash()}",
        f"- default_objective: {float(default_objective):.6f}",
        f"- best_objective: {float(best['objective']):.6f}",
        f"- delta: {float(best['objective']) - float(default_objective):+.6f}",
        "",
        "## Best Metrics",
        "",
        f"- mrr_strict: {float(best['mrr_strict']):.4f}",
        f"- top1_in_distractor_rate: {float(best['top1_in_distractor_rate']):.4f}",
        f"- critical_fn_rate: {float(best['critical_fn_rate']):.4f}",
        "",
        "## Failure Reasons Top-N",
        "",
    ]
    if top_reason_pairs:
        report_lines.append("| reason | count |")
        report_lines.append("|---|---:|")
        for reason, count in top_reason_pairs:
            report_lines.append(f"| {reason} | {int(count)} |")
    else:
        report_lines.append("- none")
    report_lines.extend(["", "## Best Weights", "", "```yaml"])
    try:
        import yaml  # type: ignore

        report_lines.append(yaml.safe_dump(best_cfg.to_dict(), sort_keys=False, allow_unicode=True).strip())
    except Exception:
        report_lines.append(json.dumps(best_cfg.to_dict(), ensure_ascii=False, indent=2))
    report_lines.append("```")
    best_report.write_text("\n".join(report_lines), encoding="utf-8")

    figure_paths = _make_figures(rows, fig_root)
    snapshot_path = out_dir / "snapshot.json"
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "inputs": {
            "run_dir": str(run_dir) if run_dir else None,
            "json": str(json_path) if json_path else None,
            "index": str(index_prefix) if index_prefix else None,
            "nlq_mode": str(args.nlq_mode),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "search": str(args.search),
            "trials": int(args.trials),
            "grid": str(args.grid),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "eval_script": str(eval_script),
        },
        "weights": {
            "fields": list(WEIGHT_FIELDS),
            "num_candidates": len(candidates),
        },
        "best": {
            "cfg_name": str(best_cfg.name),
            "cfg_hash": str(best_cfg.short_hash()),
            "objective": float(best["objective"]),
            "row": best,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "metrics_md": str(metrics_md),
            "best_weights": str(best_cfg_path),
            "best_report": str(best_report),
            "figures": list(figure_paths),
            "commands_log": str(commands_path),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"default_objective={float(default_objective):.6f}")
    print(f"best_objective={float(best['objective']):.6f}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    print(f"saved_best_config={best_cfg_path}")
    print(f"saved_best_report={best_report}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
