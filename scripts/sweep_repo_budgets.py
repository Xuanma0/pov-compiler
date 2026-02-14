from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.context.context_builder import build_context
from pov_compiler.schemas import Output


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        s = int(round(float(self.max_total_s)))
        return f"{s}/{int(self.max_tokens)}/{int(self.max_decisions)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoV0 budget sweep (repo_only / events_plus_repo)")
    parser.add_argument("--json", required=True, help="Input *_v03_decisions.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="20/50/4,40/100/8,60/200/12")
    parser.add_argument("--query", default="", help="Optional repo query")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--repo-strategy", choices=["importance_greedy", "recency_greedy"], default="importance_greedy")
    return parser.parse_args()


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    points: list[BudgetPoint] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        chunks = [x.strip() for x in token.split("/") if x.strip()]
        if len(chunks) != 3:
            raise ValueError(f"invalid budget: {token}")
        points.append(BudgetPoint(float(chunks[0]), int(chunks[1]), int(chunks[2])))
    if not points:
        raise ValueError("no budgets parsed")
    return points


def _read_output(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        output = Output.model_validate(payload)  # type: ignore[attr-defined]
    else:
        output = Output.parse_obj(payload)
    if hasattr(output, "model_dump"):
        return output.model_dump()
    return output.dict()


def _interval_union_seconds(items: list[dict[str, Any]]) -> float:
    intervals: list[tuple[float, float]] = []
    for item in items:
        try:
            t0 = float(item.get("t0", 0.0))
            t1 = float(item.get("t1", 0.0))
        except Exception:
            continue
        if t1 > t0:
            intervals.append((t0, t1))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float]] = []
    s0, s1 = intervals[0]
    for a0, a1 in intervals[1:]:
        if a0 <= s1:
            s1 = max(s1, a1)
            continue
        merged.append((s0, s1))
        s0, s1 = a0, a1
    merged.append((s0, s1))
    return float(sum(max(0.0, b - a) for a, b in merged))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RepoV0 Budget Sweep",
        "",
        f"- json: `{meta.get('json', '')}`",
        f"- video_id: `{meta.get('video_id', '')}`",
        f"- query: `{meta.get('query', '')}`",
        f"- budgets: `{meta.get('budgets', '')}`",
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


def _plot_quality(rows: list[dict[str, Any]], out_prefix: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    for variant in ("repo_only", "events_plus_repo"):
        v_rows = sorted(
            [r for r in rows if str(r.get("variant", "")) == variant],
            key=lambda x: float(x.get("budget_seconds", 0.0)),
        )
        xs = [float(r.get("budget_seconds", 0.0)) for r in v_rows]
        ys = [float(r.get("repo_quality_proxy", 0.0)) for r in v_rows]
        plt.plot(xs, ys, marker="o", label=variant)
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("repo_quality_proxy")
    plt.title("RepoV0 Quality Proxy vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        out = out_prefix.with_suffix(f".{ext}")
        plt.savefig(out)
        out_paths.append(str(out))
    plt.close()
    return out_paths


def main() -> int:
    args = parse_args()
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _read_output(json_path)
    budgets = _parse_budgets(args.budgets)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    duration_s = _to_float(payload.get("meta", {}).get("duration_s", 0.0), 0.0)
    video_id = str(payload.get("video_id", json_path.stem))
    repo_summary = payload.get("repository", {}).get("summary", {}) if isinstance(payload.get("repository", {}), dict) else {}
    before_dedup = int(repo_summary.get("chunks_before_dedup", 0) or 0)
    after_dedup = int(repo_summary.get("chunks_after_dedup", 0) or 0)
    dedup_rate = float(1.0 - (after_dedup / max(1, before_dedup))) if before_dedup > 0 else 0.0

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        base_budget = {
            "max_seconds": float(budget.max_total_s),
            "max_tokens": int(budget.max_tokens),
            "max_decisions": int(budget.max_decisions),
            "max_repo_chunks": max(4, min(64, int(budget.max_tokens // 8) if budget.max_tokens > 0 else 16)),
            "max_repo_chars": int(max(1200, budget.max_tokens * 36)),
            "repo_strategy": str(args.repo_strategy),
            "repo_query": str(args.query or ""),
        }
        for variant in ("repo_only", "events_plus_repo"):
            context = build_context(payload, mode=variant, budget=base_budget)
            repo_chunks = list(context.get("repo_chunks", []))
            repo_chars = int(sum(len(str(c.get("text", ""))) for c in repo_chunks))
            repo_coverage_s = _interval_union_seconds(repo_chunks)
            repo_coverage_ratio = float(repo_coverage_s / max(1e-6, duration_s)) if duration_s > 0 else 0.0
            repo_importance_mean = float(
                sum(_to_float(c.get("importance", 0.0), 0.0) for c in repo_chunks) / max(1, len(repo_chunks))
            )
            quality_proxy = float(
                0.45 * min(1.0, repo_coverage_ratio)
                + 0.35 * min(1.0, repo_importance_mean)
                + 0.20 * min(1.0, len(repo_chunks) / max(1, base_budget["max_repo_chunks"]))
            )
            rows.append(
                {
                    "video_id": video_id,
                    "variant": variant,
                    "budget_key": budget.key,
                    "budget_seconds": float(budget.max_total_s),
                    "budget_max_total_s": float(budget.max_total_s),
                    "budget_max_tokens": int(budget.max_tokens),
                    "budget_max_decisions": int(budget.max_decisions),
                    "max_repo_chunks": int(base_budget["max_repo_chunks"]),
                    "max_repo_chars": int(base_budget["max_repo_chars"]),
                    "repo_chunks": len(repo_chunks),
                    "repo_chars": repo_chars,
                    "repo_coverage_s": float(repo_coverage_s),
                    "repo_coverage_ratio": float(repo_coverage_ratio),
                    "repo_importance_mean": float(repo_importance_mean),
                    "repo_quality_proxy": float(quality_proxy),
                    "events_count": len(context.get("events", [])),
                    "highlights_count": len(context.get("highlights", [])),
                    "decisions_count": len(context.get("decision_points", [])),
                    "tokens_count": len(context.get("tokens", [])),
                    "repo_trace_after": int(context.get("repo_trace", {}).get("repo_after", len(repo_chunks))),
                    "repo_trace_chars_after": int(context.get("repo_trace", {}).get("repo_chars_after", repo_chars)),
                    "source_chunks_before_dedup": before_dedup,
                    "source_chunks_after_dedup": after_dedup,
                    "source_dedup_rate": float(dedup_rate),
                }
            )

    rows.sort(key=lambda r: (str(r["variant"]), float(r["budget_seconds"]), str(r["budget_key"])))
    metrics_csv = agg_dir / "metrics_by_budget.csv"
    metrics_md = agg_dir / "metrics_by_budget.md"
    _write_csv(metrics_csv, rows)
    _write_md(
        metrics_md,
        rows,
        {
            "json": str(json_path),
            "video_id": video_id,
            "query": str(args.query or ""),
            "budgets": str(args.budgets),
        },
    )
    figure_paths = _plot_quality(rows, fig_dir / "fig_repo_quality_vs_budget_seconds", formats)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json": str(json_path),
            "out_dir": str(out_dir),
            "budgets": [b.__dict__ for b in budgets],
            "query": str(args.query or ""),
            "repo_strategy": str(args.repo_strategy),
        },
        "source": {
            "video_id": video_id,
            "duration_s": duration_s,
            "repository_summary": repo_summary,
        },
        "outputs": {
            "metrics_by_budget_csv": str(metrics_csv),
            "metrics_by_budget_md": str(metrics_md),
            "figures": figure_paths,
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"video_id={video_id}")
    print(f"budgets={len(budgets)}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
