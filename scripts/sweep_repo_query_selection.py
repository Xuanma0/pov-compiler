from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.fixed_queries import FixedQuery, generate_fixed_queries, load_queries_jsonl
from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.repository import build_repo_chunks, deduplicate_chunks, policy_cfg_hash, select_chunks_for_query
from pov_compiler.repository.schema import RepoChunk
from pov_compiler.retrieval.query_parser import parse_query
from pov_compiler.retrieval.query_planner import plan as plan_query
from pov_compiler.schemas import Output


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        return f"{int(round(self.max_total_s))}/{int(self.max_tokens)}/{int(self.max_decisions)}"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for token in str(raw).split(","):
        text = token.strip()
        if not text:
            continue
        parts = [x.strip() for x in text.split("/") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"invalid budget token: {text}")
        out.append(BudgetPoint(float(parts[0]), int(parts[1]), int(parts[2])))
    if not out:
        raise ValueError("no budgets parsed")
    return out


def _parse_csv_tokens(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _normalize_uid(text: str) -> str:
    token = str(text).replace("\ufeff", "").strip()
    if token.lower().endswith(".mp4"):
        token = token[:-4]
    return token.lower()


def _uid_from_json(path: Path) -> str:
    stem = path.stem
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    uuid_re = re.compile(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
    m = uuid_re.search(cleaned) or uuid_re.search(stem)
    if m:
        return _normalize_uid(m.group(1))
    return _normalize_uid(cleaned)


def _discover_jsons(pov_json_dir: Path, uids_file: str | None, strict_uids: bool) -> tuple[list[Path], dict[str, Any]]:
    if not pov_json_dir.exists():
        raise FileNotFoundError(str(pov_json_dir))
    files = sorted(pov_json_dir.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(pov_json_dir.glob("*.json"), key=lambda p: p.name.lower())
    by_uid: dict[str, Path] = {}
    for p in files:
        uid = _uid_from_json(p)
        if uid and uid not in by_uid:
            by_uid[uid] = p
    all_uids = sorted(by_uid.keys())
    selection = {
        "selection_mode": "all_json",
        "uids_file_path": "",
        "uids_requested": 0,
        "uids_found": len(all_uids),
        "uids_missing_count": 0,
        "uids_missing_sample": [],
        "dir_uids_sample": all_uids[:5],
    }
    if not uids_file:
        return [by_uid[u] for u in all_uids], selection

    uid_path = Path(str(uids_file))
    requested: list[str] = []
    for line in uid_path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "")
        if "#" in text:
            text = text.split("#", 1)[0]
        text = text.strip()
        if not text:
            continue
        for token in re.split(r"[,\s]+", text):
            uid = _normalize_uid(token)
            if uid:
                requested.append(uid)
    found = [uid for uid in requested if uid in by_uid]
    missing = [uid for uid in requested if uid not in by_uid]
    selection.update(
        {
            "selection_mode": "uids_file",
            "uids_file_path": str(uid_path),
            "uids_requested": len(requested),
            "uids_found": len(found),
            "uids_missing_count": len(missing),
            "uids_missing_sample": missing[:10],
        }
    )
    if strict_uids and (not found or missing):
        raise RuntimeError(
            f"strict uid matching failed: found={len(found)} missing={len(missing)} "
            f"uids_file={uid_path} sample_missing={missing[:5]} dir_uid_sample={all_uids[:5]}"
        )
    return [by_uid[uid] for uid in found], selection


def _model_validate_output(payload: dict[str, Any]) -> Output:
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _load_output(path: Path) -> Output:
    return ensure_events_v1(_model_validate_output(json.loads(path.read_text(encoding="utf-8"))))


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


def _write_md(path: Path, rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# Repo query selection sweep\n\nNo rows.\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [
        "# Repo query selection sweep",
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
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_query_info(query_text: str, top_k: int) -> dict[str, Any]:
    parsed = parse_query(query_text)
    plan = plan_query(query_text)
    constraints = dict(plan.constraints)
    if parsed.place is not None:
        constraints.setdefault("place", parsed.place)
    if parsed.place_segment_ids:
        constraints.setdefault("place_segment_id", list(parsed.place_segment_ids))
    if parsed.interaction_min is not None:
        constraints.setdefault("interaction_min", parsed.interaction_min)
    if parsed.interaction_object:
        constraints.setdefault("interaction_object", parsed.interaction_object)
    if parsed.anchor_types:
        constraints.setdefault("anchor_type", list(parsed.anchor_types))
    if parsed.decision_types:
        constraints.setdefault("decision_type", list(parsed.decision_types))
    if parsed.token_types:
        constraints.setdefault("token_type", list(parsed.token_types))
    if parsed.time_range is not None:
        constraints.setdefault("time_range", parsed.time_range)
    return {
        "query": str(query_text),
        "plan_intent": str(plan.intent),
        "parsed_constraints": constraints,
        "top_k": int(top_k),
    }


def _relevant_ids(query: FixedQuery) -> set[str]:
    relevant = dict(query.relevant)
    out: set[str] = set()
    for key in ("events", "highlights", "decisions", "tokens"):
        vals = relevant.get(key, [])
        if isinstance(vals, list):
            out.update(str(x) for x in vals if str(x))
    return out


def _level_counts(chunks: list[RepoChunk]) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in chunks:
        level = str(chunk.level or chunk.scale).lower()
        out[level] = out.get(level, 0) + 1
    return out


def _evaluate_one_query(
    *,
    chunks: list[RepoChunk],
    query: FixedQuery,
    policy_name: str,
    budget: BudgetPoint,
    top_k: int,
) -> dict[str, Any]:
    selected, trace = select_chunks_for_query(
        chunks,
        query=str(query.query),
        budget={
            "max_repo_chunks": max(4, min(64, int(budget.max_tokens // 8))),
            "max_tokens": int(budget.max_tokens),
            "max_seconds": float(budget.max_total_s),
            "repo_read_policy": str(policy_name),
        },
        cfg={"read_policy": {"name": str(policy_name)}},
        query_info=_parse_query_info(str(query.query), top_k=int(top_k)),
        return_trace=True,
    )
    selected_by_id = {str(c.id): c for c in selected}
    ranked_ids = [str(x) for x in trace.get("selected_chunk_ids", []) if str(x)]
    if not ranked_ids:
        ranked_ids = [str(c.id) for c in selected]
    relevant = _relevant_ids(query)
    rank_first: int | None = None
    for i, chunk_id in enumerate(ranked_ids[: max(1, int(query.top_k or top_k))], start=1):
        chunk = selected_by_id.get(chunk_id)
        if chunk is None:
            continue
        sources = {str(x) for x in (chunk.source_ids or []) if str(x)}
        if relevant.intersection(sources):
            rank_first = i
            break
    has_rel = bool(relevant)
    hit = 1.0 if has_rel and rank_first is not None else 0.0
    mrr = float(1.0 / rank_first) if has_rel and rank_first is not None else 0.0
    top1_id = ranked_ids[0] if ranked_ids else ""
    top1_chunk = selected_by_id.get(top1_id)
    top1_rel = False
    if top1_chunk is not None:
        sources_top1 = {str(x) for x in (top1_chunk.source_ids or []) if str(x)}
        top1_rel = bool(relevant.intersection(sources_top1))
    top1_distractor = 1.0 if has_rel and not top1_rel else 0.0
    context_tokens_used = int(sum(int(c.meta.get("token_est", max(1, len(str(c.text)) // 4))) for c in selected))
    by_level = _level_counts(selected)
    return {
        "hit@k_strict": float(hit),
        "hit_at_k_strict": float(hit),
        "mrr_strict": float(mrr),
        "top1_in_distractor_rate": float(top1_distractor),
        "context_tokens_used": int(context_tokens_used),
        "selected_chunks_count": int(len(selected)),
        "selected_chunks_by_level": by_level,
        "policy_hash": str(trace.get("policy_hash", "")),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _plot(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    policies = sorted({str(r.get("policy", "")) for r in rows if str(r.get("policy", ""))})

    def _save(fig, path_prefix: Path) -> None:
        for ext in formats:
            p = path_prefix.with_suffix(f".{ext}")
            fig.savefig(p)
            figure_paths.append(str(p))
        plt.close(fig)

    fig1 = plt.figure(figsize=(7.4, 4.2))
    for policy in policies:
        p_rows = sorted([r for r in rows if str(r.get("policy")) == policy], key=lambda x: float(x.get("budget_seconds", 0.0)))
        plt.plot(
            [float(r.get("budget_seconds", 0.0)) for r in p_rows],
            [float(r.get("mrr_strict", 0.0)) for r in p_rows],
            marker="o",
            label=policy,
        )
    plt.xlabel("Budget Seconds")
    plt.ylabel("mrr_strict")
    plt.title("Repo Query Selection Quality vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    _save(fig1, out_dir / "fig_repo_query_selection_quality_vs_budget")

    fig2 = plt.figure(figsize=(7.4, 4.2))
    for policy in policies:
        p_rows = sorted([r for r in rows if str(r.get("policy")) == policy], key=lambda x: float(x.get("budget_seconds", 0.0)))
        plt.plot(
            [float(r.get("budget_seconds", 0.0)) for r in p_rows],
            [float(r.get("top1_in_distractor_rate", 0.0)) for r in p_rows],
            marker="o",
            label=policy,
        )
    plt.xlabel("Budget Seconds")
    plt.ylabel("top1_in_distractor_rate")
    plt.title("Repo Query Selection Distractor vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    _save(fig2, out_dir / "fig_repo_query_selection_distractor_vs_budget")

    fig3 = plt.figure(figsize=(8.0, 4.4))
    max_budget = max((float(r.get("budget_seconds", 0.0)) for r in rows), default=0.0)
    ref_rows = [r for r in rows if float(r.get("budget_seconds", 0.0)) == max_budget]
    level_names = sorted(
        {
            key.replace("level_", "")
            for row in ref_rows
            for key in row.keys()
            if str(key).startswith("level_")
        }
    )
    x = range(len(level_names))
    width = 0.35
    for idx, policy in enumerate(policies):
        prow = next((r for r in ref_rows if str(r.get("policy")) == policy), None)
        vals = [float((prow or {}).get(f"level_{lvl}", 0.0)) for lvl in level_names]
        offset = -width / 2 if idx == 0 else width / 2
        pos = [float(i) + offset for i in x]
        plt.bar(pos, vals, width=width, label=policy)
    plt.xticks(list(x), level_names, rotation=20)
    plt.ylabel("avg selected chunks")
    plt.title("Repo Query Selection: chunks by level (max budget)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _save(fig3, out_dir / "fig_repo_query_selection_chunks_by_level")
    return figure_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep repo query-time selection policy vs budget")
    parser.add_argument("--pov-json-dir", required=True)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="20/50/4,60/200/12")
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--policies", default="baseline,query_aware")
    parser.add_argument("--use-repo", default="true")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-uids", dest="strict_uids", action="store_true", default=True)
    parser.add_argument("--no-strict-uids", dest="strict_uids", action="store_false")
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = _parse_budgets(args.budgets)
    policies_raw = _parse_csv_tokens(args.policies)
    policy_name_map = {"baseline": "budgeted_topk", "query_aware": "query_aware"}
    policies = [policy_name_map.get(x.lower(), x.lower()) for x in policies_raw]
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]

    json_paths, selection = _discover_jsons(Path(args.pov_json_dir), args.uids_file, bool(args.strict_uids))
    if not json_paths:
        raise RuntimeError("no json selected")

    all_rows: list[dict[str, Any]] = []
    for json_path in json_paths:
        output = _load_output(json_path)
        repo_chunks = deduplicate_chunks(build_repo_chunks(output, cfg={}), cfg={})
        if args.queries_file:
            queries = load_queries_jsonl(args.queries_file)
        else:
            queries = generate_fixed_queries(
                output,
                seed=int(args.seed),
                n_time=4,
                n_anchor=4,
                n_token=4,
                n_decision=4,
                n_hard_time=4,
                default_top_k=int(args.top_k),
            )
        if not queries:
            continue
        for policy in policies:
            for budget in budgets:
                query_rows: list[dict[str, Any]] = []
                level_acc: dict[str, list[float]] = {}
                policy_hash = policy_cfg_hash({"read_policy": {"name": policy}})
                for query in queries:
                    result = _evaluate_one_query(
                        chunks=repo_chunks,
                        query=query,
                        policy_name=policy,
                        budget=budget,
                        top_k=int(args.top_k),
                    )
                    query_rows.append(result)
                    policy_hash = str(result.get("policy_hash", policy_hash))
                    for level, count in dict(result.get("selected_chunks_by_level", {})).items():
                        level_acc.setdefault(str(level), []).append(float(count))
                row: dict[str, Any] = {
                    "video_id": str(output.video_id),
                    "policy": policy,
                    "budget_key": budget.key,
                    "budget_seconds": float(budget.max_total_s),
                    "budget_max_total_s": float(budget.max_total_s),
                    "budget_max_tokens": int(budget.max_tokens),
                    "budget_max_decisions": int(budget.max_decisions),
                    "num_queries": int(len(query_rows)),
                    "hit@k_strict": _mean([float(q["hit@k_strict"]) for q in query_rows]),
                    "hit_at_k_strict": _mean([float(q["hit_at_k_strict"]) for q in query_rows]),
                    "mrr_strict": _mean([float(q["mrr_strict"]) for q in query_rows]),
                    "top1_in_distractor_rate": _mean([float(q["top1_in_distractor_rate"]) for q in query_rows]),
                    "context_tokens_used": _mean([float(q["context_tokens_used"]) for q in query_rows]),
                    "selected_chunks_count": _mean([float(q["selected_chunks_count"]) for q in query_rows]),
                    "selected_chunks_by_level": json.dumps(
                        {k: _mean(v) for k, v in sorted(level_acc.items())},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "policy_hash": policy_hash,
                }
                row["objective"] = float(row["mrr_strict"]) - 0.5 * float(row["top1_in_distractor_rate"])
                for level, vals in sorted(level_acc.items()):
                    row[f"level_{level}"] = _mean(vals)
                all_rows.append(row)

    # macro average over videos
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        key = (str(row["policy"]), str(row["budget_key"]))
        grouped.setdefault(key, []).append(row)
    agg_rows: list[dict[str, Any]] = []
    for (policy, budget_key), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], _to_float(kv[1][0].get("budget_seconds", 0.0)))):
        sample = rows[0]
        out = {
            "policy": policy,
            "budget_key": budget_key,
            "budget_seconds": float(sample.get("budget_seconds", 0.0)),
            "budget_max_total_s": float(sample.get("budget_max_total_s", 0.0)),
            "budget_max_tokens": int(sample.get("budget_max_tokens", 0)),
            "budget_max_decisions": int(sample.get("budget_max_decisions", 0)),
            "num_uids": int(len(rows)),
            "mrr_strict": _mean([_to_float(r.get("mrr_strict", 0.0), 0.0) for r in rows]),
            "hit@k_strict": _mean([_to_float(r.get("hit@k_strict", 0.0), 0.0) for r in rows]),
            "hit_at_k_strict": _mean([_to_float(r.get("hit_at_k_strict", 0.0), 0.0) for r in rows]),
            "top1_in_distractor_rate": _mean([_to_float(r.get("top1_in_distractor_rate", 0.0), 0.0) for r in rows]),
            "context_tokens_used": _mean([_to_float(r.get("context_tokens_used", 0.0), 0.0) for r in rows]),
            "selected_chunks_count": _mean([_to_float(r.get("selected_chunks_count", 0.0), 0.0) for r in rows]),
            "objective": _mean([_to_float(r.get("objective", 0.0), 0.0) for r in rows]),
            "policy_hash": str(sample.get("policy_hash", "")),
        }
        level_cols = sorted({k for r in rows for k in r.keys() if str(k).startswith("level_")})
        by_level_payload: dict[str, float] = {}
        for col in level_cols:
            val = _mean([_to_float(r.get(col, 0.0), 0.0) for r in rows])
            out[col] = val
            by_level_payload[col.replace("level_", "")] = val
        out["selected_chunks_by_level"] = json.dumps(by_level_payload, ensure_ascii=False, sort_keys=True)
        agg_rows.append(out)

    metrics_csv = agg_dir / "metrics_by_policy_budget.csv"
    metrics_md = agg_dir / "metrics_by_policy_budget.md"
    _write_csv(metrics_csv, agg_rows)
    _write_md(metrics_md, agg_rows, selection=selection)
    figure_paths = _plot(agg_rows, fig_dir, formats)

    best = max(agg_rows, key=lambda r: float(r.get("objective", 0.0))) if agg_rows else {}
    baseline_rows = [r for r in agg_rows if str(r.get("policy", "")) == "budgeted_topk"]
    baseline_best = max(baseline_rows, key=lambda r: float(r.get("objective", 0.0))) if baseline_rows else {}
    best_report = out_dir / "best_report.md"
    lines = [
        "# Repo query-time selection best report",
        "",
        f"- best_policy: `{best.get('policy', '')}`",
        f"- best_budget: `{best.get('budget_key', '')}`",
        f"- best_objective: `{float(best.get('objective', 0.0)):.4f}`",
        f"- best_mrr_strict: `{float(best.get('mrr_strict', 0.0)):.4f}`",
        f"- best_top1_in_distractor_rate: `{float(best.get('top1_in_distractor_rate', 0.0)):.4f}`",
    ]
    if baseline_best:
        lines.extend(
            [
                "",
                "## Delta vs baseline (budgeted_topk)",
                "",
                f"- baseline_budget: `{baseline_best.get('budget_key', '')}`",
                f"- delta_mrr_strict: `{float(best.get('mrr_strict', 0.0)) - float(baseline_best.get('mrr_strict', 0.0)):.4f}`",
                f"- delta_top1_in_distractor_rate: `{float(best.get('top1_in_distractor_rate', 0.0)) - float(baseline_best.get('top1_in_distractor_rate', 0.0)):.4f}`",
                f"- delta_objective: `{float(best.get('objective', 0.0)) - float(baseline_best.get('objective', 0.0)):.4f}`",
            ]
        )
    best_report.write_text("\n".join(lines), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(args.pov_json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "selection": selection,
            "budgets": [b.__dict__ for b in budgets],
            "policies": policies_raw,
            "policy_name_map": policy_name_map,
            "top_k": int(args.top_k),
            "queries_file": str(args.queries_file) if args.queries_file else None,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "metrics_md": str(metrics_md),
            "figures": figure_paths,
            "best_report": str(best_report),
            "best": best,
            "baseline_best": baseline_best,
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selection_mode={selection.get('selection_mode', '')}")
    print(f"selected_uids={len(json_paths)}")
    print(f"budgets={len(budgets)}")
    print(f"policies={policies_raw}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_best_report={best_report}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
