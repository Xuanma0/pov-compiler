from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import NLQSample, load_hard_pseudo_nlq
from pov_compiler.bench.nlq.evaluator import evaluate_nlq_samples
from pov_compiler.bench.nlq.safety import SafetyGateConfig, build_safety_report
from pov_compiler.eval.eval_cross_variant import evaluate_cross_variant
from pov_compiler.eval.fixed_queries import FixedQuery, generate_fixed_queries
from pov_compiler.retrieval.constraints import HardConstraintConfig
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.schemas import Output
from pov_compiler.utils.media import get_duration_bucket


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _as_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _attach_perception_sidecar(output: Output, json_path: Path) -> Output:
    if isinstance(output.perception, dict) and output.perception:
        return output
    uid = str(output.video_id)
    run_root = json_path.parent.parent
    sidecar = run_root / "perception" / uid / "perception.json"
    if not sidecar.exists():
        return output
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return output
    if isinstance(payload, dict):
        output.perception = payload
    return output


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


def _group_mean(rows: list[dict[str, Any]], keys: tuple[str, ...], metric: str) -> dict[tuple[Any, ...], float]:
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        try:
            value = float(row.get(metric, 0.0))
        except Exception:
            value = 0.0
        buckets[key].append(value)
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}


def _load_ego4d_queries(ann_path: Path, video_id: str, top_k: int) -> list[FixedQuery]:
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation file not found: {ann_path}")
    text = ann_path.read_text(encoding="utf-8")
    payloads: list[dict[str, Any]] = []
    if ann_path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                payloads.append(item)
    else:
        root = json.loads(text)
        if isinstance(root, list):
            payloads.extend([x for x in root if isinstance(x, dict)])
        elif isinstance(root, dict):
            items = root.get("queries")
            if isinstance(items, list):
                payloads.extend([x for x in items if isinstance(x, dict)])

    queries: list[FixedQuery] = []
    for i, item in enumerate(payloads, start=1):
        ann_video = str(item.get("video_uid", item.get("video_id", "")))
        if ann_video and ann_video != str(video_id):
            continue
        q = str(item.get("query", "")).strip()
        if not q:
            continue
        rel = item.get("relevant", {})
        if not isinstance(rel, dict):
            rel = {}
        qtype = str(item.get("type", "ego4d"))
        queries.append(
            FixedQuery(
                qid=str(item.get("qid", f"q_{i:06d}")),
                type=qtype,
                query=q,
                top_k=int(item.get("top_k", top_k)),
                time=item.get("time") if isinstance(item.get("time"), dict) else None,
                relevant={
                    "highlights": [str(x) for x in rel.get("highlights", [])],
                    "events": [str(x) for x in rel.get("events", [])],
                    "decisions": [str(x) for x in rel.get("decisions", [])],
                    "tokens": [str(x) for x in rel.get("tokens", [])],
                },
                meta=item.get("meta", {}) if isinstance(item.get("meta"), dict) else {},
            ).normalized()
        )
    return queries


def _build_fixed_queries(
    output: Output,
    mode: str,
    n: int,
    seed: int,
    top_k: int,
    ann_path: Path | None,
) -> list[FixedQuery]:
    if mode == "ego4d":
        if ann_path is None:
            raise ValueError("--ann is required when --mode ego4d")
        queries = _load_ego4d_queries(ann_path=ann_path, video_id=output.video_id, top_k=top_k)
        if not queries:
            raise ValueError("No ego4d queries matched this video")
        return queries

    base = generate_fixed_queries(
        output=output,
        seed=seed,
        n_time=max(1, n),
        n_anchor=max(1, n // 3),
        n_token=max(1, n),
        n_decision=max(1, n),
        n_hard_time=max(1, n),
        time_window_s=8.0,
        default_top_k=top_k,
        hard_overlap_thresh=0.05,
    )
    if mode == "mock":
        remap = {
            "token": "mock_token",
            "decision": "mock_decision",
            "time": "mock_time",
            "hard_time": "mock_hard_time",
            "anchor": "mock_anchor",
        }
    else:
        remap = {
            "token": "pseudo_token",
            "decision": "pseudo_decision",
            "time": "pseudo_time",
            "hard_time": "pseudo_hard_time",
            "anchor": "pseudo_anchor",
        }

    out: list[FixedQuery] = []
    for q in base:
        q2 = q.normalized()
        q2.type = remap.get(q2.type, q2.type)
        out.append(q2)
    return out


def _pick_existing(query_types: set[str], candidates: list[str], default: str) -> str:
    for item in candidates:
        if item in query_types:
            return item
    return default


def _make_report(
    out_path: Path,
    mode: str,
    overall_rows: list[dict[str, Any]],
    by_type_rows: list[dict[str, Any]],
    per_query_rows: list[dict[str, Any]],
    allow_gt_fallback: bool,
    rerank_cfg_name: str = "default",
    rerank_cfg_hash: str = "",
    hard_constraints_enabled: bool = True,
    hard_constraints_cfg: dict[str, Any] | None = None,
    safety_report: dict[str, Any] | None = None,
) -> None:
    variants = sorted({str(r.get("variant", "")) for r in overall_rows if str(r.get("variant", ""))})
    query_types = sorted({str(r.get("query_type", "")) for r in by_type_rows if str(r.get("query_type", ""))})
    duration_buckets = sorted({str(r.get("duration_bucket", "")) for r in by_type_rows if str(r.get("duration_bucket", ""))})

    mean_hit = _group_mean(overall_rows, ("variant",), "hit_at_k")
    mean_hit1 = _group_mean(overall_rows, ("variant",), "hit_at_1")
    mean_hit1_strict = _group_mean(overall_rows, ("variant",), "hit_at_1_strict")
    mean_hitk_strict = _group_mean(overall_rows, ("variant",), "hit_at_k_strict")
    mean_fp = _group_mean(overall_rows, ("variant",), "top1_in_distractor_rate")
    mean_mrr = _group_mean(overall_rows, ("variant",), "mrr")
    q_hit = _group_mean(by_type_rows, ("query_type", "variant"), "hit_at_k")
    q_hitk_strict = _group_mean(by_type_rows, ("query_type", "variant"), "hit_at_k_strict")
    q_hit1_strict = _group_mean(by_type_rows, ("query_type", "variant"), "hit_at_1_strict")
    q_fp = _group_mean(by_type_rows, ("query_type", "variant"), "top1_in_distractor_rate")
    # some modes may not carry event_hit; fallback to hit_at_k.
    q_event = _group_mean(by_type_rows, ("query_type", "variant"), "hit_at_k_event")
    if not q_event:
        q_event = q_hit

    lines: list[str] = []
    lines.append("# NLQ Evaluation Report")
    lines.append("")
    lines.append(f"- mode: {mode}")
    lines.append(f"- allow_gt_fallback: {str(bool(allow_gt_fallback)).lower()}")
    lines.append(f"- rerank_cfg_name: {rerank_cfg_name}")
    lines.append(f"- rerank_cfg_hash: {rerank_cfg_hash}")
    lines.append(f"- hard_constraints_enabled: {str(bool(hard_constraints_enabled)).lower()}")
    if isinstance(hard_constraints_cfg, dict):
        lines.append(f"- hard_constraints_cfg: `{json.dumps(hard_constraints_cfg, ensure_ascii=False, sort_keys=True)}`")
    lines.append(f"- variants: {', '.join(variants)}")
    lines.append(f"- query_types: {', '.join(query_types)}")
    if duration_buckets:
        lines.append(f"- duration_buckets: {', '.join(duration_buckets)}")
    lines.append(f"- rows_overall: {len(overall_rows)}")
    lines.append(f"- rows_by_query_type: {len(by_type_rows)}")
    if isinstance(safety_report, dict):
        lines.append(
            f"- safety_count_granularity: {str(safety_report.get('count_granularity', 'row=(variant,budget,query)'))}"
        )
        lines.append(f"- safety_gate_enforced: {str(bool(safety_report.get('gate_enforced', False))).lower()}")
        lines.append(f"- safety_max_critical_fn: {int(safety_report.get('max_critical_fn', 0))}")
        lines.append(f"- safety_critical_fn_denominator: {int(safety_report.get('critical_fn_denominator', 0))}")
        lines.append(f"- safety_critical_fn_count: {int(safety_report.get('critical_fn_count', 0))}")
        lines.append(f"- safety_critical_fn_rate: {float(safety_report.get('critical_fn_rate', 0.0)):.4f}")
        lines.append(f"- safety_pass_gate: {str(bool(safety_report.get('pass_gate', True))).lower()}")
    lines.append("")

    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| variant | hit@k | hit@1 | hit@1_strict | hit@k_strict | top1_in_distractor_rate | mrr |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        lines.append(
            f"| {variant} | {mean_hit.get((variant,), 0.0):.4f} | {mean_hit1.get((variant,), 0.0):.4f} | "
            f"{mean_hit1_strict.get((variant,), 0.0):.4f} | {mean_hitk_strict.get((variant,), 0.0):.4f} | "
            f"{mean_fp.get((variant,), 0.0):.4f} | {mean_mrr.get((variant,), 0.0):.4f} |"
        )
    lines.append("")

    lines.append("## By Query Type")
    lines.append("")
    lines.append("| query_type | variant | hit@k | hit@k_strict | hit@1_strict | top1_in_distractor_rate | event_hit@k |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for qtype in query_types:
        for variant in variants:
            lines.append(
                f"| {qtype} | {variant} | {q_hit.get((qtype, variant), 0.0):.4f} | "
                f"{q_hitk_strict.get((qtype, variant), 0.0):.4f} | "
                f"{q_hit1_strict.get((qtype, variant), 0.0):.4f} | "
                f"{q_fp.get((qtype, variant), 0.0):.4f} | "
                f"{q_event.get((qtype, variant), 0.0):.4f} |"
            )
    lines.append("")

    if duration_buckets:
        lines.append("## By Duration Bucket")
        lines.append("")
        lines.append("| duration_bucket | variant | hit@k | hit@k_strict | top1_in_distractor_rate | mrr |")
        lines.append("|---|---|---:|---:|---:|---:|")
        bucket_hit = _group_mean(by_type_rows, ("duration_bucket", "variant"), "hit_at_k")
        bucket_hitk_strict = _group_mean(by_type_rows, ("duration_bucket", "variant"), "hit_at_k_strict")
        bucket_fp = _group_mean(by_type_rows, ("duration_bucket", "variant"), "top1_in_distractor_rate")
        bucket_mrr = _group_mean(by_type_rows, ("duration_bucket", "variant"), "mrr")
        for bucket in duration_buckets:
            for variant in variants:
                lines.append(
                    f"| {bucket} | {variant} | {bucket_hit.get((bucket, variant), 0.0):.4f} | "
                    f"{bucket_hitk_strict.get((bucket, variant), 0.0):.4f} | "
                    f"{bucket_fp.get((bucket, variant), 0.0):.4f} | "
                    f"{bucket_mrr.get((bucket, variant), 0.0):.4f} |"
                )
        lines.append("")

    lines.append("## Key Deltas")
    lines.append("")

    qset = set(query_types)
    if mode == "hard_pseudo_nlq":
        token_q = _pick_existing(qset, ["hard_pseudo_token", "pseudo_token", "token"], "hard_pseudo_token")
        decision_q = _pick_existing(qset, ["hard_pseudo_decision", "pseudo_decision", "decision"], "hard_pseudo_decision")
        hard_q = _pick_existing(qset, ["hard_pseudo_anchor", "pseudo_hard_time", "hard_time"], "hard_pseudo_anchor")
    elif mode == "pseudo_nlq":
        token_q = "pseudo_token"
        decision_q = "pseudo_decision"
        hard_q = "pseudo_hard_time"
    elif mode == "mock":
        token_q = "mock_token"
        decision_q = "mock_decision"
        hard_q = "mock_hard_time"
    else:
        token_q = _pick_existing(qset, ["token", "pseudo_token"], "token")
        decision_q = _pick_existing(qset, ["decision", "pseudo_decision"], "decision")
        hard_q = _pick_existing(qset, ["hard_time", "pseudo_hard_time"], "hard_time")

    def _delta(qtype: str, rhs: str) -> tuple[float, float, float]:
        base = q_hit.get((qtype, "highlights_only"), 0.0)
        val = q_hit.get((qtype, rhs), 0.0)
        return base, val, val - base

    tb, tf, td = _delta(token_q, "full")
    _, _, tpd = _delta(token_q, "highlights_plus_tokens")
    lines.append(
        f"- {token_q}: `full` vs `highlights_only` hit@k {tf:.4f} vs {tb:.4f} (delta {td:+.4f}); "
        f"`highlights_plus_tokens` delta {tpd:+.4f}."
    )
    token_base_fp = q_fp.get((token_q, "highlights_only"), 0.0)
    token_full_fp = q_fp.get((token_q, "full"), 0.0)
    lines.append(
        f"- {token_q} strict: `full` hit@k_strict {q_hitk_strict.get((token_q, 'full'), 0.0):.4f}, "
        f"`highlights_only` {q_hitk_strict.get((token_q, 'highlights_only'), 0.0):.4f}; "
        f"top1_in_distractor_rate delta {token_full_fp - token_base_fp:+.4f} (lower is better)."
    )

    db, df, dd = _delta(decision_q, "full")
    _, _, dpd = _delta(decision_q, "highlights_plus_decisions")
    lines.append(
        f"- {decision_q}: `full` vs `highlights_only` hit@k {df:.4f} vs {db:.4f} (delta {dd:+.4f}); "
        f"`highlights_plus_decisions` delta {dpd:+.4f}."
    )
    decision_base_fp = q_fp.get((decision_q, "highlights_only"), 0.0)
    decision_full_fp = q_fp.get((decision_q, "full"), 0.0)
    lines.append(
        f"- {decision_q} strict: `full` hit@k_strict {q_hitk_strict.get((decision_q, 'full'), 0.0):.4f}, "
        f"`highlights_only` {q_hitk_strict.get((decision_q, 'highlights_only'), 0.0):.4f}; "
        f"top1_in_distractor_rate delta {decision_full_fp - decision_base_fp:+.4f} (lower is better)."
    )

    hard_full = q_event.get((hard_q, "full"), 0.0)
    hard_raw = q_event.get((hard_q, "raw_events_only"), 0.0)
    lines.append(
        f"- {hard_q} (event hit@k): `full` {hard_full:.4f} vs `raw_events_only` {hard_raw:.4f} "
        f"(delta {hard_full - hard_raw:+.4f})."
    )
    if "hard_pseudo_contact" in qset:
        contact_full = q_hit.get(("hard_pseudo_contact", "full"), 0.0)
        contact_hl = q_hit.get(("hard_pseudo_contact", "highlights_only"), 0.0)
        contact_fp_full = q_fp.get(("hard_pseudo_contact", "full"), 0.0)
        contact_fp_hl = q_fp.get(("hard_pseudo_contact", "highlights_only"), 0.0)
        lines.append(
            f"- hard_pseudo_contact: `full` hit@k {contact_full:.4f} vs `highlights_only` {contact_hl:.4f} "
            f"(delta {contact_full - contact_hl:+.4f}); "
            f"top1_in_distractor_rate delta {contact_fp_full - contact_fp_hl:+.4f}. "
            "This query family depends on events_v1 contact/perception evidence."
        )
    lines.append("")

    lines.append("## Constraint Filtering Stats")
    lines.append("")
    if per_query_rows:
        n = float(len(per_query_rows))

        def _rate(pred) -> float:
            c = 0.0
            for row in per_query_rows:
                if pred(row):
                    c += 1.0
            return float(c / n)

        stats = {
            "present_after_scene_change_rate": _rate(lambda r: bool(r.get("present_after_scene_change", False))),
            "present_first_last_rate": _rate(lambda r: bool(r.get("present_first_last", False))),
            "present_type_match_rate": _rate(lambda r: bool(r.get("present_type_match", False))),
            "filtered_after_scene_change_rate": _rate(lambda r: bool(r.get("filtered_after_scene_change", False))),
            "filtered_first_last_rate": _rate(lambda r: bool(r.get("filtered_first_last", False))),
            "filtered_type_match_rate": _rate(lambda r: bool(r.get("filtered_type_match", False))),
            "relaxed_after_scene_change_rate": _rate(lambda r: bool(r.get("relaxed_after_scene_change", False))),
            "relaxed_first_last_rate": _rate(lambda r: bool(r.get("relaxed_first_last", False))),
            "relaxed_type_match_rate": _rate(lambda r: bool(r.get("relaxed_type_match", False))),
            "used_fallback_rate": _rate(lambda r: bool(r.get("used_fallback", False))),
            "avg_filtered_before": float(sum(float(r.get("filtered_hits_before", 0.0)) for r in per_query_rows) / n),
            "avg_filtered_after": float(sum(float(r.get("filtered_hits_after", 0.0)) for r in per_query_rows) / n),
        }
        lines.append("| stat | value |")
        lines.append("|---|---:|")
        for key in sorted(stats.keys()):
            lines.append(f"| {key} | {float(stats[key]):.4f} |")
    else:
        lines.append("- no per-query rows")
    lines.append("")

    if isinstance(safety_report, dict):
        lines.append("## Safety Gate")
        lines.append("")
        lines.append("| field | value |")
        lines.append("|---|---:|")
        lines.append(f"| count_granularity | {safety_report.get('count_granularity', '')} |")
        lines.append(f"| gate_enforced | {str(bool(safety_report.get('gate_enforced', False))).lower()} |")
        lines.append(f"| max_critical_fn | {int(safety_report.get('max_critical_fn', 0))} |")
        lines.append(f"| critical_fn_denominator | {int(safety_report.get('critical_fn_denominator', 0))} |")
        lines.append(f"| critical_fn_count | {int(safety_report.get('critical_fn_count', 0))} |")
        lines.append(f"| critical_fn_rate | {float(safety_report.get('critical_fn_rate', 0.0)):.4f} |")
        lines.append(f"| would_pass_gate | {str(bool(safety_report.get('would_pass_gate', True))).lower()} |")
        lines.append(f"| pass_gate | {str(bool(safety_report.get('pass_gate', True))).lower()} |")
        lines.append("")
        var_stats = safety_report.get("variant_stats", {})
        if isinstance(var_stats, dict) and var_stats:
            lines.append("### Safety By Variant")
            lines.append("")
            lines.append("| variant | critical_fn_count | critical_fn_denominator | critical_fn_rate |")
            lines.append("|---|---:|---:|---:|")
            for variant in sorted(var_stats.keys()):
                item = var_stats[variant] if isinstance(var_stats[variant], dict) else {}
                lines.append(
                    f"| {variant} | {int(item.get('critical_fn_count', 0))} | "
                    f"{int(item.get('critical_fn_denominator', 0))} | "
                    f"{float(item.get('critical_fn_rate', 0.0)):.4f} |"
                )
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _resolve_allow_gt_fallback(mode: str, cli_value: bool | None) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if mode == "hard_pseudo_nlq":
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLQ evaluation (mock/pseudo_nlq/hard_pseudo_nlq/ego4d)")
    parser.add_argument("--json", required=True, help="Pipeline output json")
    parser.add_argument("--index", default=None, help="Vector index prefix")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["mock", "pseudo_nlq", "hard_pseudo_nlq", "ego4d"], default="pseudo_nlq")
    parser.add_argument("--ann", default=None, help="Annotation path for mode=ego4d")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--sweep", action="store_true", help="Run budget sweep")
    parser.add_argument("--n", type=int, default=10, help="Query count knob")
    parser.add_argument("--seed", type=int, default=0, help="Query seed")
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=6, help="Query top_k")

    parser.set_defaults(allow_gt_fallback=None)
    parser.add_argument("--allow-gt-fallback", dest="allow_gt_fallback", action="store_true")
    parser.add_argument("--no-allow-gt-fallback", dest="allow_gt_fallback", action="store_false")
    parser.add_argument("--rerank-cfg", default=None, help="Path to reranker WeightConfig YAML/JSON")
    parser.add_argument("--hard-constraints", choices=["on", "off"], default="on")
    parser.add_argument(
        "--hard-constraints-cfg",
        default=str(ROOT / "configs" / "hard_constraints_default.yaml"),
        help="Path to hard constraint config YAML/JSON",
    )
    parser.set_defaults(safety_gate_enforced=False)
    parser.add_argument(
        "--safety-gate",
        "--enforce-safety-gate",
        dest="safety_gate_enforced",
        action="store_true",
        help="Enforce safety gate and fail with non-zero exit when threshold is exceeded",
    )
    parser.add_argument(
        "--no-safety-gate",
        "--report-only",
        dest="safety_gate_enforced",
        action="store_false",
        help="Report-only mode: always write safety report but do not fail process",
    )
    parser.add_argument("--max-critical-fn", type=int, default=None, help="Safety gate threshold")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    eval_cfg = dict(cfg.get("eval", {}))
    budgets_cfg = dict(eval_cfg.get("budgets", {}))
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    safety_cfg = dict(cfg.get("safety", {}))
    rerank_cfg_yaml = cfg.get("reranker", {})
    resolved_cfg: WeightConfig
    if args.rerank_cfg:
        resolved_cfg = resolve_weight_config(Path(args.rerank_cfg))
    elif isinstance(rerank_cfg_yaml, dict) and rerank_cfg_yaml:
        resolved_cfg = resolve_weight_config(rerank_cfg_yaml)
    else:
        resolved_cfg = WeightConfig()
    hard_cfg_yaml = cfg.get("hard_constraints", {})
    if str(args.hard_constraints).lower() == "off":
        resolved_hard_cfg = HardConstraintConfig(
            enable_after_scene_change=False,
            enable_first_last=False,
            enable_type_match=False,
            relax_on_empty=True,
            relax_order=["after_scene_change", "first_last", "type_match"],
        )
    elif args.hard_constraints_cfg and Path(args.hard_constraints_cfg).exists():
        resolved_hard_cfg = HardConstraintConfig.from_yaml(Path(args.hard_constraints_cfg))
    elif isinstance(hard_cfg_yaml, dict) and hard_cfg_yaml:
        resolved_hard_cfg = HardConstraintConfig.from_dict(hard_cfg_yaml)
    else:
        resolved_hard_cfg = HardConstraintConfig()

    json_path = Path(args.json)
    output = _attach_perception_sidecar(_as_output(json_path), json_path)

    budgets = budgets_cfg if bool(args.sweep) else {
        "max_total_s": [max(budgets_cfg.get("max_total_s", [60]))],
        "max_tokens": [max(budgets_cfg.get("max_tokens", [200]))],
        "max_decisions": [max(budgets_cfg.get("max_decisions", [12]))],
    }

    allow_gt_fallback = _resolve_allow_gt_fallback(str(args.mode), args.allow_gt_fallback)

    if str(args.mode) == "hard_pseudo_nlq":
        try:
            samples: list[NLQSample] = load_hard_pseudo_nlq(
                output,
                seed=int(args.seed),
                n_highlight=max(1, int(args.n)),
                n_token=max(1, int(args.n)),
                n_decision=max(1, int(args.n)),
                top_k=max(1, int(args.top_k)),
            )
        except Exception as exc:
            print(f"error=build_hard_queries_failed detail={exc}")
            return 1
        if not samples:
            print("error=no_hard_pseudo_queries")
            return 1

        result = evaluate_nlq_samples(
            output=output,
            samples=samples,
            budgets=budgets,
            sweep=bool(args.sweep),
            retriever_config=retrieval_cfg,
            index_prefix=args.index,
            rerank_cfg=resolved_cfg,
            hard_constraints_cfg=resolved_hard_cfg,
            allow_gt_fallback=allow_gt_fallback,
        )
        overall_rows = result["overall_rows"]
        by_type_rows = result["by_query_type_rows"]
        per_query_rows = result["per_query_rows"]
        queries_total = len(samples)
    else:
        try:
            queries = _build_fixed_queries(
                output=output,
                mode=str(args.mode),
                n=int(args.n),
                seed=int(args.seed),
                top_k=int(args.top_k),
                ann_path=Path(args.ann) if args.ann else None,
            )
        except Exception as exc:
            print(f"error=build_queries_failed detail={exc}")
            return 1
        if not queries:
            print("error=no_queries")
            return 1

        result = evaluate_cross_variant(
            full_output=output,
            queries=queries,
            budgets=budgets,
            sweep=bool(args.sweep),
            retriever_config=retrieval_cfg,
            index_prefix=args.index,
        )
        overall_rows = result["overall_rows"]
        by_type_rows = result["by_query_type_rows"]
        per_query_rows = result["per_query_rows"]
        queries_total = len(queries)

    duration_bucket = get_duration_bucket(output.meta.get("duration_s"))
    for rows in (overall_rows, by_type_rows, per_query_rows):
        for row in rows:
            row.setdefault("video_uid", row.get("video_id", output.video_id))
            row.setdefault("duration_bucket", duration_bucket)
            row.setdefault("rerank_cfg_name", resolved_cfg.name)
            row.setdefault("rerank_cfg_hash", resolved_cfg.short_hash())
            row.setdefault("hard_constraints_enabled", str(args.hard_constraints).lower() == "on")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "nlq_results.csv"
    summary_csv = out_dir / "nlq_summary.csv"
    report_md = out_dir / "nlq_report.md"
    safety_json = out_dir / "safety_report.json"
    _write_csv(results_csv, per_query_rows)
    _write_csv(summary_csv, by_type_rows)
    resolved_safety_cfg = SafetyGateConfig.from_dict(safety_cfg)
    if args.max_critical_fn is not None:
        resolved_safety_cfg.max_critical_fn = int(args.max_critical_fn)
    safety_report = build_safety_report(
        video_id=output.video_id,
        per_query_rows=per_query_rows,
        gate_cfg=resolved_safety_cfg,
        enforce_gate=bool(args.safety_gate_enforced),
    )
    safety_json.write_text(json.dumps(safety_report, ensure_ascii=False, indent=2), encoding="utf-8")

    _make_report(
        report_md,
        mode=str(args.mode),
        overall_rows=overall_rows,
        by_type_rows=by_type_rows,
        per_query_rows=per_query_rows,
        allow_gt_fallback=allow_gt_fallback,
        rerank_cfg_name=str(resolved_cfg.name),
        rerank_cfg_hash=str(resolved_cfg.short_hash()),
        hard_constraints_enabled=str(args.hard_constraints).lower() == "on",
        hard_constraints_cfg=resolved_hard_cfg.to_dict(),
        safety_report=safety_report,
    )

    print(f"video_id={output.video_id}")
    print(f"mode={args.mode}")
    print(f"allow_gt_fallback={str(bool(allow_gt_fallback)).lower()}")
    print(f"rerank_cfg_name={resolved_cfg.name}")
    print(f"rerank_cfg_hash={resolved_cfg.short_hash()}")
    print(f"hard_constraints={args.hard_constraints}")
    print(f"duration_bucket={duration_bucket}")
    print(f"queries_total={queries_total}")
    print(f"rows_results={len(per_query_rows)}")
    print(f"rows_summary={len(by_type_rows)}")
    print(f"safety_count_granularity={safety_report.get('count_granularity', 'row=(variant,budget,query)')}")
    print(f"safety_gate_enforced={str(bool(safety_report.get('gate_enforced', False))).lower()}")
    print(f"safety_threshold={int(safety_report.get('max_critical_fn', 0))}")
    print(f"safety_denominator={int(safety_report.get('critical_fn_denominator', 0))}")
    print(f"safety_rate={float(safety_report.get('critical_fn_rate', 0.0)):.4f}")
    print(f"safety_pass={str(bool(safety_report.get('pass_gate', True))).lower()}")
    print(f"safety_critical_fn={int(safety_report.get('critical_fn_count', 0))}")
    print(f"saved_results={results_csv}")
    print(f"saved_summary={summary_csv}")
    print(f"saved_report={report_md}")
    print(f"saved_safety={safety_json}")
    if bool(safety_report.get("gate_enforced", False)) and not bool(safety_report.get("pass_gate", True)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
