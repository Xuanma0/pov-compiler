from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.streaming.intervention_config import InterventionConfig


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _read_summary_from_snapshot(run_dir: Path) -> dict[str, Any]:
    snap = run_dir / "snapshot.json"
    if not snap.exists():
        return {}
    try:
        payload = json.loads(snap.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary", {})
    return dict(summary) if isinstance(summary, dict) else {}


def _extract_metrics(run_dir: Path) -> dict[str, float]:
    summary = _read_summary_from_snapshot(run_dir)
    return {
        "strict_success_rate": float(_to_float(summary.get("hit_at_k_strict")) or 0.0),
        "critical_fn_rate": float(_to_float(summary.get("safety_critical_fn_rate")) or 0.0),
        "latency_p95_e2e_ms": float(_to_float(summary.get("e2e_latency_p95_ms")) or 0.0),
        "avg_trials_per_query": float(_to_float(summary.get("avg_trials_per_query")) or 0.0),
    }


def _objective(metrics: dict[str, float]) -> float:
    # Fixed global objective (not from intervention config).
    w1, w2, w3, w4 = 1.0, 1.0, 0.2, 0.1
    return (
        w1 * float(metrics.get("strict_success_rate", 0.0))
        - w2 * float(metrics.get("critical_fn_rate", 0.0))
        - w3 * (float(metrics.get("latency_p95_e2e_ms", 0.0)) / 1000.0)
        - w4 * float(metrics.get("avg_trials_per_query", 0.0))
    )


def _random_cfg(base: InterventionConfig, rng: random.Random, idx: int, max_trials_cap: int) -> InterventionConfig:
    return InterventionConfig(
        name=f"rand_{idx:04d}",
        w_safety=max(0.05, base.w_safety * rng.uniform(0.5, 1.8)),
        w_latency=max(0.01, base.w_latency * rng.uniform(0.5, 2.0)),
        w_trials=max(0.01, base.w_trials * rng.uniform(0.5, 2.0)),
        penalty_budget_up=max(0.0, base.penalty_budget_up * rng.uniform(0.5, 2.0)),
        penalty_retry=max(0.0, base.penalty_retry * rng.uniform(0.5, 2.0)),
        penalty_relax=max(0.0, base.penalty_relax * rng.uniform(0.5, 2.0)),
        max_trials_cap=max(1, min(int(max_trials_cap), rng.randint(2, max(2, int(max_trials_cap))))),
    )


def _grid_cfgs(base: InterventionConfig, max_trials_cap: int) -> list[InterventionConfig]:
    scales = [0.8, 1.0, 1.2]
    out: list[InterventionConfig] = []
    for i, s1 in enumerate(scales):
        for j, s2 in enumerate(scales):
            cfg = InterventionConfig(
                name=f"grid_{i}{j}",
                w_safety=max(0.05, base.w_safety * s1),
                w_latency=max(0.01, base.w_latency * s2),
                w_trials=max(0.01, base.w_trials * (scales[(i + j) % len(scales)])),
                penalty_budget_up=max(0.0, base.penalty_budget_up * s2),
                penalty_retry=max(0.0, base.penalty_retry * s1),
                penalty_relax=max(0.0, base.penalty_relax * scales[(i * 2 + j) % len(scales)]),
                max_trials_cap=max(1, min(int(max_trials_cap), int(base.max_trials_cap))),
            )
            out.append(cfg)
    return out


def _run_trial(
    *,
    json_path: str,
    out_dir: Path,
    step_s: float,
    budgets: str,
    queries: list[str],
    seed: int,
    cfg_path: Path,
    max_trials_cap: int,
) -> tuple[int, list[str]]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "streaming_budget_smoke.py"),
        "--json",
        str(json_path),
        "--out_dir",
        str(out_dir),
        "--step-s",
        str(float(step_s)),
        "--budgets",
        str(budgets),
        "--policy",
        "safety_latency_intervention",
        "--intervention-cfg",
        str(cfg_path),
        "--max-trials",
        str(int(max_trials_cap)),
        "--seed",
        str(int(seed)),
    ]
    for q in queries:
        cmd.extend(["--query", str(q)])

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    logs: list[str] = []
    logs.append(_render_cmd(cmd))
    logs.append(result.stdout or "")
    logs.append(result.stderr or "")
    return int(result.returncode), logs


def _make_figures(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    good = [r for r in rows if int(r.get("returncode", 1)) == 0]
    if not good:
        return []

    xs = [float(r.get("latency_p95_e2e_ms", 0.0)) for r in good]
    ys = [float(r.get("objective", -1e9)) for r in good]
    stricts = [float(r.get("strict_success_rate", 0.0)) for r in good]
    labels = [str(r.get("cfg_id", "")) for r in good]

    best_idx = max(range(len(good)), key=lambda i: float(good[i].get("objective", -1e9)))
    paths: list[str] = []

    p1 = out_dir / "fig_objective_vs_latency"
    plt.figure(figsize=(7.2, 4.4))
    plt.scatter(xs, ys)
    plt.scatter([xs[best_idx]], [ys[best_idx]], marker="*", s=140, label="best")
    plt.xlabel("latency_p95_e2e_ms")
    plt.ylabel("objective")
    plt.title("Streaming Intervention Sweep: Objective vs Latency")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    # Pareto frontier on latency (min) and strict_success_rate (max)
    points = list(zip(xs, stricts, labels))
    pareto: list[tuple[float, float, str]] = []
    for i, (x, y, lab) in enumerate(points):
        dominated = False
        for j, (x2, y2, _) in enumerate(points):
            if i == j:
                continue
            if x2 <= x and y2 >= y and (x2 < x or y2 > y):
                dominated = True
                break
        if not dominated:
            pareto.append((x, y, lab))
    pareto.sort(key=lambda t: (t[0], -t[1], t[2]))

    p2 = out_dir / "fig_pareto_frontier"
    plt.figure(figsize=(7.2, 4.4))
    plt.scatter(xs, stricts, label="configs")
    if pareto:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        plt.plot(px, py, marker="o", linestyle="--", label="pareto")
    plt.scatter([xs[best_idx]], [stricts[best_idx]], marker="*", s=140, label="best")
    plt.xlabel("latency_p95_e2e_ms")
    plt.ylabel("strict_success_rate")
    plt.title("Streaming Intervention Sweep: Pareto Frontier")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        paths.append(str(p))
    plt.close()

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep streaming intervention configs")
    parser.add_argument("--json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--step-s", type=float, default=8.0)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--search", choices=["random", "grid"], default="random")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-cfg", default=str(ROOT / "configs" / "streaming_intervention_default.yaml"))
    parser.add_argument("--objective", choices=["combo"], default="combo")
    parser.add_argument("--max-trials-cap", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cfg_dir = out_dir / "configs"
    run_root = out_dir / "runs"
    fig_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    queries = [str(x).strip() for x in list(args.query or []) if str(x).strip()]
    base_cfg = InterventionConfig.from_yaml(args.base_cfg)
    rng = random.Random(int(args.seed))

    candidates: list[InterventionConfig] = [InterventionConfig.from_dict(base_cfg.to_dict())]
    candidates[0].name = str(base_cfg.name or "default")

    trial_target = max(1, int(args.trials))
    if str(args.search).lower() == "grid":
        grid = _grid_cfgs(base_cfg, max_trials_cap=max(1, int(args.max_trials_cap)))
        for g in grid:
            if len(candidates) >= trial_target:
                break
            candidates.append(g)
    else:
        while len(candidates) < trial_target:
            candidates.append(_random_cfg(base_cfg, rng, len(candidates), max_trials_cap=max(1, int(args.max_trials_cap))))

    rows: list[dict[str, Any]] = []
    command_log_path = logs_dir / "commands.sh"
    command_log_path.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    for idx, cfg in enumerate(candidates):
        cfg_id = f"cfg_{idx:04d}"
        cfg_path = cfg_dir / f"{cfg_id}.yaml"
        cfg.to_yaml(cfg_path)

        run_dir = run_root / cfg_id
        rc, logs = _run_trial(
            json_path=str(args.json),
            out_dir=run_dir,
            step_s=float(args.step_s),
            budgets=str(args.budgets),
            queries=queries,
            seed=int(args.seed),
            cfg_path=cfg_path,
            max_trials_cap=max(1, min(int(args.max_trials_cap), int(cfg.max_trials_cap))),
        )
        with command_log_path.open("a", encoding="utf-8") as f:
            f.write(f"# {cfg_id}\n{logs[0]}\n\n")
        (logs_dir / f"{cfg_id}.stdout.log").write_text(logs[1], encoding="utf-8")
        (logs_dir / f"{cfg_id}.stderr.log").write_text(logs[2], encoding="utf-8")

        metrics = _extract_metrics(run_dir) if rc == 0 else {
            "strict_success_rate": 0.0,
            "critical_fn_rate": 1.0,
            "latency_p95_e2e_ms": 999999.0,
            "avg_trials_per_query": float(max(1, int(cfg.max_trials_cap))),
        }
        obj = _objective(metrics)
        row = {
            "cfg_id": cfg_id,
            "cfg_name": str(cfg.name),
            "cfg_hash": str(cfg.stable_hash()),
            "cfg_path": str(cfg_path),
            "run_dir": str(run_dir),
            "returncode": int(rc),
            "objective": float(obj),
            **metrics,
            **{f"cfg_{k}": v for k, v in cfg.to_dict().items()},
        }
        rows.append(row)

    _write_csv(out_dir / "results_sweep.csv", rows)

    good = [r for r in rows if int(r.get("returncode", 1)) == 0]
    if not good:
        print("error=no successful sweep trials")
        return 2
    best = max(good, key=lambda r: float(r.get("objective", -1e9)))
    default = rows[0]

    best_cfg = InterventionConfig.from_yaml(best["cfg_path"])
    best_cfg_path = out_dir / "best_config.yaml"
    best_cfg.to_yaml(best_cfg_path)

    figure_paths = _make_figures(rows, fig_dir)

    best_report = out_dir / "best_report.md"
    report_lines = [
        "# Streaming Intervention Sweep Report",
        "",
        f"- default_cfg: `{default.get('cfg_name')}` hash={default.get('cfg_hash')}",
        f"- best_cfg: `{best.get('cfg_name')}` hash={best.get('cfg_hash')}",
        f"- default_objective: {float(default.get('objective', 0.0)):.6f}",
        f"- best_objective: {float(best.get('objective', 0.0)):.6f}",
        f"- delta: {float(best.get('objective', 0.0) - default.get('objective', 0.0)):.6f}",
        "",
        "## Metrics (default vs best)",
        "",
        "| metric | default | best |",
        "|---|---:|---:|",
    ]
    for m in ("strict_success_rate", "critical_fn_rate", "latency_p95_e2e_ms", "avg_trials_per_query", "objective"):
        report_lines.append(
            f"| {m} | {float(default.get(m, 0.0)):.6f} | {float(best.get(m, 0.0)):.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Best Config",
            "",
            "```yaml",
            Path(best["cfg_path"]).read_text(encoding="utf-8"),
            "```",
        ]
    )
    best_report.write_text("\n".join(report_lines), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json": str(args.json),
            "out_dir": str(out_dir),
            "step_s": float(args.step_s),
            "budgets": str(args.budgets),
            "queries": queries,
            "search": str(args.search),
            "trials": int(args.trials),
            "seed": int(args.seed),
            "base_cfg": str(args.base_cfg),
            "objective": str(args.objective),
            "max_trials_cap": int(args.max_trials_cap),
        },
        "best": {
            "cfg_id": str(best.get("cfg_id", "")),
            "cfg_name": str(best.get("cfg_name", "")),
            "cfg_hash": str(best.get("cfg_hash", "")),
            "cfg_path": str(best.get("cfg_path", "")),
            "objective": float(best.get("objective", 0.0)),
        },
        "default": {
            "cfg_id": str(default.get("cfg_id", "")),
            "cfg_name": str(default.get("cfg_name", "")),
            "cfg_hash": str(default.get("cfg_hash", "")),
            "objective": float(default.get("objective", 0.0)),
        },
        "outputs": {
            "results_csv": str(out_dir / "results_sweep.csv"),
            "best_config": str(best_cfg_path),
            "best_report": str(best_report),
            "figures": figure_paths,
            "command_log": str(command_log_path),
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also keep a copy of best report/config under a predictable folder for paper_ready copy.
    publish_dir = out_dir / "streaming_intervention_sweep"
    publish_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_cfg_path, publish_dir / "best_config.yaml")
    shutil.copyfile(best_report, publish_dir / "best_report.md")

    default_obj = float(default.get("objective", 0.0))
    best_obj = float(best.get("objective", 0.0))
    print(f"default_objective={default_obj:.6f}")
    print(f"best_objective={best_obj:.6f}")
    print(f"delta={(best_obj - default_obj):.6f}")
    print(f"saved_results={out_dir / 'results_sweep.csv'}")
    print(f"saved_best_config={best_cfg_path}")
    print(f"saved_best_report={best_report}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
