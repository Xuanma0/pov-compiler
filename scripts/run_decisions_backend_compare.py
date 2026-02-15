from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return float(out)


def _render_cmd(cmd: list[str]) -> str:
    redacted: list[str] = []
    prev = ""
    for token in cmd:
        redacted.append(shlex.quote(_redact_token(str(token), prev)))
        prev = str(token)
    return " ".join(redacted)


def _redact_token(token: str, prev: str) -> str:
    p = str(prev or "").strip().lower()
    value = str(token)
    if p in {"--model-base-url", "--base-url", "--base_url"}:
        return re.sub(r"([?&](?:key|api_key|token|secret)=)[^&\\s]+", r"\1***", value, flags=re.IGNORECASE)
    if p in {"--model-api-key-env", "--api-key-env", "--api_key_env"}:
        return "***ENV***"
    if re.search(r"(authorization|bearer|api[_-]?key|token|secret)", value, flags=re.IGNORECASE):
        if "=" in value:
            k, _, _ = value.partition("=")
            return f"{k}=***"
    return value


def _run(cmd: list[str], *, cwd: Path, log_prefix: Path, commands_path: Path) -> int:
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    with commands_path.open("a", encoding="utf-8") as f:
        f.write(f"# {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        f.write(_render_cmd(cmd) + "\n\n")
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


def _read_summary_map(run_dir: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    rows = _read_csv(run_dir / "summary.csv")
    for row in rows:
        uid = str(row.get("video_uid", "")).strip()
        if uid:
            out[uid] = row
    return out


def _read_nlq_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    rows = _read_csv(run_dir / "nlq_summary_all.csv")
    by_uid: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        uid = str(row.get("video_uid", "")).strip()
        if not uid:
            uid = str(row.get("video_id", "")).strip()
        if not uid:
            continue
        if str(row.get("variant", "")).strip().lower() not in {"", "full"}:
            continue
        by_uid.setdefault(uid, []).append(row)

    out: dict[str, dict[str, float]] = {}
    for uid, uid_rows in by_uid.items():
        mrr_vals: list[float] = []
        dist_vals: list[float] = []
        crit_vals: list[float] = []
        for row in uid_rows:
            for key in ("mrr", "mrr_strict", "nlq_full_mrr"):
                val = _to_float(row.get(key))
                if val is not None:
                    mrr_vals.append(val)
                    break
            for key in ("top1_in_distractor_rate", "top1_in_distractor", "nlq_full_top1_in_distractor_rate"):
                val = _to_float(row.get(key))
                if val is not None:
                    dist_vals.append(val)
                    break
            for key in ("critical_fn_rate", "safety_critical_fn_rate"):
                val = _to_float(row.get(key))
                if val is not None:
                    crit_vals.append(val)
                    break
        out[uid] = {
            "mrr_strict": float(sum(mrr_vals) / len(mrr_vals)) if mrr_vals else math.nan,
            "distractor_rate": float(sum(dist_vals) / len(dist_vals)) if dist_vals else math.nan,
            "critical_fn_rate": float(sum(crit_vals) / len(crit_vals)) if crit_vals else math.nan,
        }
    return out


def _pick_latency(row: dict[str, str]) -> float | None:
    for key in ("latency_p95_ms", "e2e_latency_p95_ms", "latency_p95_e2e_ms", "e2e_ms_p95"):
        v = _to_float(row.get(key))
        if v is not None:
            return float(v)
    return None


def _metric_or_nan(row: dict[str, str], keys: tuple[str, ...]) -> float:
    for key in keys:
        v = _to_float(row.get(key))
        if v is not None:
            return float(v)
    return float("nan")


def _maybe(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _safe_delta(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return float(b - a)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], summary: list[str]) -> None:
    lines = [
        "# Decisions Backend Compare",
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
    fig_paths: list[str] = []

    # Delta bar by metric (mean across UIDs).
    metrics = [
        "delta_mrr_strict",
        "delta_distractor_rate",
        "delta_critical_fn_rate",
        "delta_latency_p95_ms",
    ]
    means: list[float] = []
    labels: list[str] = []
    for m in metrics:
        vals = [float(r.get(m)) for r in rows if _to_float(r.get(m)) is not None]
        if vals:
            means.append(float(sum(vals) / len(vals)))
        else:
            means.append(0.0)
        labels.append(m.replace("delta_", ""))

    p1 = out_dir / "fig_decisions_backend_delta"
    plt.figure(figsize=(8.0, 4.4))
    xs = list(range(len(labels)))
    plt.bar(xs, means)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xticks(xs, labels, rotation=20, ha="right")
    plt.ylabel("delta (real - stub)")
    plt.title("Decisions Backend Delta")
    plt.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()

    # Tradeoff scatter.
    p2 = out_dir / "fig_decisions_backend_tradeoff"
    plt.figure(figsize=(8.0, 4.4))
    xa = [float(_to_float(r.get("latency_p95_ms_stub")) or 0.0) for r in rows]
    xb = [float(_to_float(r.get("latency_p95_ms_real")) or 0.0) for r in rows]
    ya = [float(_to_float(r.get("mrr_strict_stub")) or 0.0) for r in rows]
    yb = [float(_to_float(r.get("mrr_strict_real")) or 0.0) for r in rows]
    plt.scatter(xa, ya, marker="o", label="stub")
    plt.scatter(xb, yb, marker="x", label="real")
    plt.xlabel("latency_p95_ms")
    plt.ylabel("mrr_strict")
    plt.title("Decisions Backend Tradeoff")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()
    return fig_paths


def _compare_runs(run_a: Path, run_b: Path, out_dir: Path, a_label: str, b_label: str) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    summary_a = _read_summary_map(run_a)
    summary_b = _read_summary_map(run_b)
    nlq_a = _read_nlq_metrics(run_a)
    nlq_b = _read_nlq_metrics(run_b)
    uids = sorted(set(summary_a.keys()) | set(summary_b.keys()))

    rows: list[dict[str, Any]] = []
    for uid in uids:
        sa = summary_a.get(uid, {})
        sb = summary_b.get(uid, {})
        na = nlq_a.get(uid, {})
        nb = nlq_b.get(uid, {})

        mrr_a = _to_float(na.get("mrr_strict"))
        if mrr_a is None:
            mrr_a = _to_float(sa.get("mrr"))
        mrr_b = _to_float(nb.get("mrr_strict"))
        if mrr_b is None:
            mrr_b = _to_float(sb.get("mrr"))

        dist_a = _to_float(na.get("distractor_rate"))
        if dist_a is None:
            dist_a = _metric_or_nan(sa, ("top1_in_distractor_rate", "distractor_rate"))
        dist_b = _to_float(nb.get("distractor_rate"))
        if dist_b is None:
            dist_b = _metric_or_nan(sb, ("top1_in_distractor_rate", "distractor_rate"))

        crit_a = _to_float(na.get("critical_fn_rate"))
        if crit_a is None:
            crit_a = _metric_or_nan(sa, ("critical_fn_rate",))
        crit_b = _to_float(nb.get("critical_fn_rate"))
        if crit_b is None:
            crit_b = _metric_or_nan(sb, ("critical_fn_rate",))

        lat_a = _pick_latency(sa)
        lat_b = _pick_latency(sb)

        row = {
            "uid": uid,
            "status_stub": str(sa.get("status", "missing")) if sa else "missing",
            "status_real": str(sb.get("status", "missing")) if sb else "missing",
            "decisions_backend_stub": str(sa.get("decisions_backend", "")) if sa else "",
            "decisions_backend_real": str(sb.get("decisions_backend", "")) if sb else "",
            "decision_pool_kind_stub": str(sa.get("decision_pool_kind", "")) if sa else "",
            "decision_pool_kind_real": str(sb.get("decision_pool_kind", "")) if sb else "",
            "mrr_strict_stub": _maybe(mrr_a),
            "mrr_strict_real": _maybe(mrr_b),
            "delta_mrr_strict": _safe_delta(_maybe(mrr_a), _maybe(mrr_b)),
            "distractor_rate_stub": _maybe(dist_a),
            "distractor_rate_real": _maybe(dist_b),
            "delta_distractor_rate": _safe_delta(_maybe(dist_a), _maybe(dist_b)),
            "critical_fn_rate_stub": _maybe(crit_a),
            "critical_fn_rate_real": _maybe(crit_b),
            "delta_critical_fn_rate": _safe_delta(_maybe(crit_a), _maybe(crit_b)),
            "latency_p95_ms_stub": _maybe(lat_a),
            "latency_p95_ms_real": _maybe(lat_b),
            "delta_latency_p95_ms": _safe_delta(_maybe(lat_a), _maybe(lat_b)),
        }
        rows.append(row)

    columns = [
        "uid",
        "status_stub",
        "status_real",
        "decisions_backend_stub",
        "decisions_backend_real",
        "decision_pool_kind_stub",
        "decision_pool_kind_real",
        "mrr_strict_stub",
        "mrr_strict_real",
        "delta_mrr_strict",
        "distractor_rate_stub",
        "distractor_rate_real",
        "delta_distractor_rate",
        "critical_fn_rate_stub",
        "critical_fn_rate_real",
        "delta_critical_fn_rate",
        "latency_p95_ms_stub",
        "latency_p95_ms_real",
        "delta_latency_p95_ms",
    ]

    tables = out_dir / "tables"
    figures = out_dir / "figures"
    table_csv = tables / "table_decisions_backend_compare.csv"
    table_md = tables / "table_decisions_backend_compare.md"
    _write_csv(table_csv, rows, columns)
    _write_md(
        table_md,
        rows,
        columns,
        [
            f"- run_a: `{run_a}` ({a_label})",
            f"- run_b: `{run_b}` ({b_label})",
            f"- uids_total: {len(uids)}",
        ],
    )
    fig_paths = _make_figures(rows, figures, formats=["png", "pdf"])

    def _stats(key: str) -> dict[str, Any]:
        vals = [float(v) for v in (_to_float(r.get(key)) for r in rows) if v is not None]
        return {
            "count": len(vals),
            "mean": statistics.mean(vals) if vals else None,
            "median": statistics.median(vals) if vals else None,
        }

    summary = {
        "labels": {"a": a_label, "b": b_label},
        "uids_total": len(uids),
        "uids_with_a": len(summary_a),
        "uids_with_b": len(summary_b),
        "uids_with_both": sum(1 for uid in uids if uid in summary_a and uid in summary_b),
        "delta_stats": {
            "mrr_strict": _stats("delta_mrr_strict"),
            "distractor_rate": _stats("delta_distractor_rate"),
            "critical_fn_rate": _stats("delta_critical_fn_rate"),
            "latency_p95_ms": _stats("delta_latency_p95_ms"),
        },
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": fig_paths,
        },
    }
    (out_dir / "compare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, fig_paths, summary


def _build_smoke_cmd(
    *,
    root: str,
    out_dir: Path,
    uids_file: str | None,
    jobs: int,
    n: int | None,
    min_size_bytes: int,
    probe_candidates: int,
    decisions_backend: str,
    model_provider: str,
    model_name: str | None,
    model_base_url: str | None,
    model_api_key_env: str | None,
    fake_mode: str,
    nlq_mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ego4d_smoke.py"),
        "--root",
        str(root),
        "--out_dir",
        str(out_dir),
        "--jobs",
        str(int(jobs)),
        "--min-size-bytes",
        str(int(min_size_bytes)),
        "--probe-candidates",
        str(int(probe_candidates)),
        "--no-proxy",
        "--resume",
        "--run-eval",
        "--run-nlq",
        "--nlq-mode",
        str(nlq_mode),
        "--no-run-perception",
        "--decisions-backend",
        str(decisions_backend),
    ]
    if uids_file:
        cmd.extend(["--uids-file", str(uids_file)])
    if n is not None:
        cmd.extend(["--n", str(int(n))])
    if str(decisions_backend).strip().lower() == "model":
        cmd.extend(["--model-provider", str(model_provider)])
        if model_name:
            cmd.extend(["--model-name", str(model_name)])
        if model_base_url:
            cmd.extend(["--model-base-url", str(model_base_url)])
        if model_api_key_env:
            cmd.extend(["--model-api-key-env", str(model_api_key_env)])
        cmd.extend(["--model-fake-mode", str(fake_mode)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare decisions backend heuristic vs model")
    parser.add_argument("--run-a-dir", default=None)
    parser.add_argument("--run-b-dir", default=None)
    parser.add_argument("--a-label", default="stub")
    parser.add_argument("--b-label", default="real")

    parser.add_argument("--root", default=None)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-size-bytes", type=int, default=0)
    parser.add_argument("--probe-candidates", type=int, default=0)
    parser.add_argument("--nlq-mode", default="hard_pseudo_nlq")

    parser.add_argument("--model-provider", choices=["fake", "openai_compat", "gemini", "qwen", "deepseek", "glm"], default="fake")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-base-url", default=None)
    parser.add_argument("--model-api-key-env", default=None)
    parser.add_argument("--fake-mode", choices=["minimal", "diverse"], default="diverse")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_only_mode = bool(args.run_a_dir and args.run_b_dir)
    compare_dir = out_dir if compare_only_mode else (out_dir / "compare")
    compare_dir.mkdir(parents=True, exist_ok=True)
    commands_path = compare_dir / "commands.sh"
    if not commands_path.exists():
        commands_path.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    run_a = Path(args.run_a_dir) if args.run_a_dir else (out_dir / "run_A")
    run_b = Path(args.run_b_dir) if args.run_b_dir else (out_dir / "run_B")

    cmd_a: list[str] | None = None
    cmd_b: list[str] | None = None
    if compare_only_mode:
        if not run_a.exists() or not run_b.exists():
            print("error=--run-a-dir/--run-b-dir must exist")
            return 2
    else:
        if not args.root:
            print("error=--root is required when --run-a-dir/--run-b-dir are not provided")
            return 2
        run_a.mkdir(parents=True, exist_ok=True)
        run_b.mkdir(parents=True, exist_ok=True)
        cmd_a = _build_smoke_cmd(
            root=str(args.root),
            out_dir=run_a,
            uids_file=str(args.uids_file) if args.uids_file else None,
            jobs=int(args.jobs),
            n=args.n,
            min_size_bytes=int(args.min_size_bytes),
            probe_candidates=int(args.probe_candidates),
            decisions_backend="heuristic",
            model_provider=str(args.model_provider),
            model_name=str(args.model_name) if args.model_name else None,
            model_base_url=str(args.model_base_url) if args.model_base_url else None,
            model_api_key_env=str(args.model_api_key_env) if args.model_api_key_env else None,
            fake_mode=str(args.fake_mode),
            nlq_mode=str(args.nlq_mode),
        )
        rc = _run(cmd_a, cwd=ROOT, log_prefix=compare_dir / "run_A", commands_path=commands_path)
        if rc != 0:
            return rc

        cmd_b = _build_smoke_cmd(
            root=str(args.root),
            out_dir=run_b,
            uids_file=str(args.uids_file) if args.uids_file else None,
            jobs=int(args.jobs),
            n=args.n,
            min_size_bytes=int(args.min_size_bytes),
            probe_candidates=int(args.probe_candidates),
            decisions_backend="model",
            model_provider=str(args.model_provider),
            model_name=str(args.model_name) if args.model_name else None,
            model_base_url=str(args.model_base_url) if args.model_base_url else None,
            model_api_key_env=str(args.model_api_key_env) if args.model_api_key_env else None,
            fake_mode=str(args.fake_mode),
            nlq_mode=str(args.nlq_mode),
        )
        rc = _run(cmd_b, cwd=ROOT, log_prefix=compare_dir / "run_B", commands_path=commands_path)
        if rc != 0:
            return rc

    rows, fig_paths, summary = _compare_runs(run_a, run_b, compare_dir, str(args.a_label), str(args.b_label))

    snapshot = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "root": str(args.root) if args.root else None,
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "jobs": int(args.jobs),
            "n": int(args.n) if args.n is not None else None,
            "seed": int(args.seed),
            "nlq_mode": str(args.nlq_mode),
            "model_provider": str(args.model_provider),
            "model_name": str(args.model_name) if args.model_name else None,
            "model_base_url": _redact_token(str(args.model_base_url), "--model-base-url") if args.model_base_url else None,
            "model_api_key_env": "***ENV***" if args.model_api_key_env else None,
            "fake_mode": str(args.fake_mode),
        },
        "runs": {"run_A": str(run_a), "run_B": str(run_b)},
        "outputs": {
            "compare_dir": str(compare_dir),
            "table_csv": str(compare_dir / "tables" / "table_decisions_backend_compare.csv"),
            "table_md": str(compare_dir / "tables" / "table_decisions_backend_compare.md"),
            "figures": fig_paths,
            "summary": str(compare_dir / "compare_summary.json"),
        },
        "summary": summary,
        "commands": {
            "run_A": _render_cmd(cmd_a) if cmd_a else None,
            "run_B": _render_cmd(cmd_b) if cmd_b else None,
        },
    }
    (compare_dir / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_lines = [
        "# Decisions Backend Compare",
        "",
        f"- run_A: `{run_a}`",
        f"- run_B: `{run_b}`",
        f"- compare: `{compare_dir}`",
        "",
        "## Outputs",
        "",
        f"- `{compare_dir / 'tables' / 'table_decisions_backend_compare.csv'}`",
        f"- `{compare_dir / 'tables' / 'table_decisions_backend_compare.md'}`",
        f"- `{compare_dir / 'figures' / 'fig_decisions_backend_delta.png'}`",
        f"- `{compare_dir / 'figures' / 'fig_decisions_backend_tradeoff.png'}`",
        f"- `{compare_dir / 'compare_summary.json'}`",
        f"- `{compare_dir / 'snapshot.json'}`",
        f"- `{compare_dir / 'commands.sh'}`",
    ]
    (compare_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    if cmd_a is not None:
        print(f"saved_run_A={run_a}")
    if cmd_b is not None:
        print(f"saved_run_B={run_b}")
    print(f"saved_compare={compare_dir}")
    print(f"saved_table={[str(compare_dir / 'tables' / 'table_decisions_backend_compare.csv'), str(compare_dir / 'tables' / 'table_decisions_backend_compare.md')]}")
    print(f"saved_figures={fig_paths}")
    print(f"saved_snapshot={compare_dir / 'snapshot.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
