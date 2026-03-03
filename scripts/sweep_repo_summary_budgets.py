from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from pov_compiler.repository import build_repo_chunks, deduplicate_chunks
from pov_compiler.schemas import Output


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        return f"{int(round(self.max_total_s))}/{int(self.max_tokens)}/{int(self.max_decisions)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep repo summary policy across budgets")
    parser.add_argument("--pov-json-dir", required=True)
    parser.add_argument("--uids-file", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="20/50/4,60/200/12,120/400/24")
    parser.add_argument("--provider", default="fake", choices=["fake", "openai", "openai_compat", "gemini", "qwen", "deepseek", "glm"])
    parser.add_argument("--model", default="fake-summary-v0")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--query", default="anchor=turn_head top_k=6")
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        chunks = [x.strip() for x in token.split("/") if x.strip()]
        if len(chunks) != 3:
            raise ValueError(f"invalid budget: {token}")
        out.append(BudgetPoint(float(chunks[0]), int(chunks[1]), int(chunks[2])))
    if not out:
        raise ValueError("no budget parsed")
    return out


def _read_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _uid_from_path(path: Path) -> str:
    stem = str(path.stem)
    for suffix in ("_v03_decisions", "_decisions"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _select_json_files(json_dir: Path, uids_file: str) -> tuple[list[Path], dict[str, Any]]:
    all_json = sorted(json_dir.glob("*_decisions.json"))
    if not uids_file:
        return all_json, {"selection_mode": "all_json", "uids_requested": len(all_json), "uids_found": len(all_json), "uids_missing_count": 0}
    requested: list[str] = []
    for line in Path(uids_file).read_text(encoding="utf-8").splitlines():
        clean = line.lstrip("\ufeff").split("#", 1)[0].strip()
        if not clean:
            continue
        for token in clean.replace(",", " ").split():
            uid = str(token).strip().replace(".mp4", "")
            if uid:
                requested.append(uid)
    by_uid = {_uid_from_path(p): p for p in all_json}
    chosen = [by_uid[uid] for uid in requested if uid in by_uid]
    missing = [uid for uid in requested if uid not in by_uid]
    if not chosen:
        raise RuntimeError("uids-file provided but no uid matched pov-json-dir")
    return chosen, {
        "selection_mode": "uids_file",
        "uids_file_path": str(uids_file),
        "uids_requested": len(requested),
        "uids_found": len(chosen),
        "uids_missing_count": len(missing),
        "uids_missing_sample": missing[:10],
    }


def _interval_union(rows: list[dict[str, Any]]) -> float:
    intervals: list[tuple[float, float]] = []
    for row in rows:
        try:
            t0 = float(row.get("t0", 0.0))
            t1 = float(row.get("t1", 0.0))
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
    lines = [
        "# Repo Summary Sweep",
        "",
        f"- pov_json_dir: `{meta.get('pov_json_dir', '')}`",
        f"- budgets: `{meta.get('budgets', '')}`",
        f"- selection_mode: `{meta.get('selection_mode', '')}`",
        "",
    ]
    if rows:
        cols = list(rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(rows: list[dict[str, Any]], fig_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    policies = sorted({str(r.get("policy")) for r in rows})

    def _rows(policy: str) -> list[dict[str, Any]]:
        return sorted([r for r in rows if str(r.get("policy")) == policy], key=lambda x: float(x.get("budget_seconds", 0.0)))

    # quality
    p1 = fig_dir / "fig_repo_summary_quality_vs_budget_seconds"
    plt.figure(figsize=(7.2, 4.2))
    for p in policies:
        rws = _rows(p)
        xs = [float(r.get("budget_seconds", 0.0)) for r in rws]
        ys = [float(r.get("repo_quality_proxy", 0.0)) for r in rws]
        plt.plot(xs, ys, marker="o", label=p)
    plt.xlabel("Budget Seconds")
    plt.ylabel("repo_quality_proxy")
    plt.title("Repo Summary Quality vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        fp = p1.with_suffix(f".{ext}")
        plt.savefig(fp)
        out.append(str(fp))
    plt.close()

    # size
    p2 = fig_dir / "fig_repo_summary_size_vs_budget_seconds"
    plt.figure(figsize=(7.2, 4.2))
    for p in policies:
        rws = _rows(p)
        xs = [float(r.get("budget_seconds", 0.0)) for r in rws]
        ys = [float(r.get("repo_chars_selected", 0.0)) for r in rws]
        plt.plot(xs, ys, marker="o", label=p)
    plt.xlabel("Budget Seconds")
    plt.ylabel("repo_chars_selected")
    plt.title("Repo Context Size vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        fp = p2.with_suffix(f".{ext}")
        plt.savefig(fp)
        out.append(str(fp))
    plt.close()

    # delta summary-baseline
    p3 = fig_dir / "fig_repo_summary_delta_vs_budget_seconds"
    base = {str(r.get("budget_key")): r for r in rows if str(r.get("policy")) == "baseline"}
    summ = {str(r.get("budget_key")): r for r in rows if str(r.get("policy")) == "summary_v0"}
    keys = sorted(set(base.keys()) & set(summ.keys()), key=lambda x: float(x.split("/")[0]))
    xs = [float(k.split("/")[0]) for k in keys]
    yd = [float(summ[k].get("repo_quality_proxy", 0.0)) - float(base[k].get("repo_quality_proxy", 0.0)) for k in keys]
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xs, yd, marker="o")
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("Budget Seconds")
    plt.ylabel("delta repo_quality_proxy (summary-baseline)")
    plt.title("Repo Summary Delta")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        fp = p3.with_suffix(f".{ext}")
        plt.savefig(fp)
        out.append(str(fp))
    plt.close()
    return out


def main() -> int:
    args = parse_args()
    json_dir = Path(args.pov_json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = _parse_budgets(args.budgets)
    selected_jsons, selection = _select_json_files(json_dir, str(args.uids_file or ""))
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]

    policies = {
        "baseline": {"write_policy": {"name": "fixed_interval", "chunk_step_s": 0.0}},
        "summary_v0": {
            "write_policy": {"name": "multiscale+summary_v0", "chunk_step_s": 0.0, "summary_window_s": 60.0},
            "summary": {
                "enabled": True,
                "window_s": 60.0,
                "model": {
                    "enabled": not bool(args.dry_run),
                    "provider": str(args.provider),
                    "model": str(args.model),
                    "base_url": args.base_url,
                    "api_key_env": str(args.api_key_env or ""),
                    "timeout_s": 60,
                    "max_retries": 1,
                    "max_tokens": 400,
                    "temperature": 0.2,
                    "model_cache_enabled": True,
                    "model_cache_dir": "data/outputs/model_cache",
                },
            },
        },
    }

    per_run_rows: list[dict[str, Any]] = []
    for json_path in selected_jsons:
        output = _read_output(json_path)
        video_id = str(output.video_id)
        duration_s = float(output.meta.get("duration_s", 0.0) or 0.0)
        for budget in budgets:
            budget_cfg = {
                "use_repo": True,
                "max_total_s": float(budget.max_total_s),
                "max_tokens": int(budget.max_tokens),
                "max_decisions": int(budget.max_decisions),
                "max_seconds": float(budget.max_total_s),
                "max_repo_chunks": max(4, min(64, int(budget.max_tokens // 8) if budget.max_tokens > 0 else 16)),
                "max_repo_chars": int(max(1200, budget.max_tokens * 36)),
                "max_repo_tokens": int(budget.max_tokens),
                "repo_read_policy": "query_aware",
                "repo_strategy": "importance_greedy",
                "repo_query": str(args.query),
            }
            for policy_name, cfg_patch in policies.items():
                cfg = {
                    "scales": {"event": True, "decision": True, "place": True, "window": True, "segment": True},
                    "window_s": 30.0,
                    "min_segment_s": 5.0,
                    "dedup": {"iou_thresh": 0.6, "sim_thresh": 0.9, "cross_scale": True, "keep_best_importance": True},
                }
                cfg.update(cfg_patch)
                raw = build_repo_chunks(output, cfg=cfg)
                deduped = deduplicate_chunks(raw, cfg=cfg.get("dedup", {}))
                rows = [
                    c.model_dump() if hasattr(c, "model_dump") else c.dict()  # type: ignore[union-attr]
                    for c in deduped
                ]
                output.repository = {"chunks": rows, "summary": {"chunks_before_dedup": len(raw), "chunks_after_dedup": len(deduped)}}
                context = build_context(
                    output,
                    mode="repo_only",
                    budget=budget_cfg,
                    query_info={"query": str(args.query), "top_k": 6},
                )
                selected = list(context.get("repo_chunks", []))
                selected_chars = int(sum(len(str(r.get("text", ""))) for r in selected))
                selected_cov = _interval_union(selected)
                selected_imp = 0.0
                if selected:
                    selected_imp = float(sum(float(r.get("importance", 0.0) or 0.0) for r in selected) / len(selected))
                quality = float(
                    0.45 * min(1.0, selected_cov / max(1e-6, duration_s if duration_s > 0 else selected_cov + 1.0))
                    + 0.35 * min(1.0, selected_imp)
                    + 0.20 * min(1.0, len(selected) / max(1, budget_cfg["max_repo_chunks"]))
                )
                summary_selected = sum(
                    1 for r in selected if str(r.get("level", r.get("scale", ""))).strip().lower() == "summary"
                )
                per_run_rows.append(
                    {
                        "video_id": video_id,
                        "policy": policy_name,
                        "budget_key": budget.key,
                        "budget_seconds": float(budget.max_total_s),
                        "budget_max_tokens": int(budget.max_tokens),
                        "budget_max_decisions": int(budget.max_decisions),
                        "repo_selected_chunks": int(len(selected)),
                        "repo_selected_summary_chunks": int(summary_selected),
                        "repo_chars_selected": int(selected_chars),
                        "repo_coverage_s": float(selected_cov),
                        "repo_quality_proxy": float(quality),
                        "mrr_strict": float("nan"),
                        "top1_in_distractor_rate": float("nan"),
                        "critical_fn_rate": float("nan"),
                    }
                )

    # aggregate by (policy, budget)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_run_rows:
        key = (str(row["policy"]), str(row["budget_key"]))
        grouped.setdefault(key, []).append(row)
    agg_rows: list[dict[str, Any]] = []
    for (policy, budget_key), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], float(kv[0][1].split("/")[0]))):
        values = lambda col: [float(r.get(col, 0.0)) for r in rows]
        agg_rows.append(
            {
                "policy": policy,
                "budget_key": budget_key,
                "budget_seconds": float(budget_key.split("/")[0]),
                "num_uids": len(rows),
                "repo_selected_chunks": statistics.mean(values("repo_selected_chunks")),
                "repo_selected_summary_chunks": statistics.mean(values("repo_selected_summary_chunks")),
                "repo_chars_selected": statistics.mean(values("repo_chars_selected")),
                "repo_coverage_s": statistics.mean(values("repo_coverage_s")),
                "repo_quality_proxy": statistics.mean(values("repo_quality_proxy")),
                "mrr_strict": "",
                "top1_in_distractor_rate": "",
                "critical_fn_rate": "",
            }
        )

    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    csv_path = agg_dir / "metrics_by_policy_budget.csv"
    md_path = agg_dir / "metrics_by_policy_budget.md"
    _write_csv(csv_path, agg_rows)
    _write_md(md_path, agg_rows, {"pov_json_dir": str(json_dir), "budgets": str(args.budgets), **selection})
    figure_paths = _plot(agg_rows, fig_dir, formats=formats)
    snapshot_path = out_dir / "snapshot.json"
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(json_dir),
            "uids_file": str(args.uids_file or ""),
            "budgets": str(args.budgets),
            "provider": str(args.provider),
            "model": str(args.model),
            "dry_run": bool(args.dry_run),
        },
        "selection": selection,
        "outputs": {
            "metrics_by_policy_budget_csv": str(csv_path),
            "metrics_by_policy_budget_md": str(md_path),
            "figures": figure_paths,
            "rows": len(agg_rows),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selection_mode={selection.get('selection_mode', 'all_json')}")
    print(f"selected_uids={selection.get('uids_found', len(selected_jsons))}")
    print(f"budgets={len(budgets)}")
    print(f"saved_metrics_csv={csv_path}")
    print(f"saved_metrics_md={md_path}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
