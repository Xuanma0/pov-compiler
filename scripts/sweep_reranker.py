from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import NLQSample, load_hard_pseudo_nlq
from pov_compiler.bench.nlq.evaluator import evaluate_nlq_samples
from pov_compiler.bench.nlq.sweep_utils import compute_objective, rank_rows_by_metric, summarize_variant_metrics
from pov_compiler.retrieval.constraints import HardConstraintConfig
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.schemas import Output


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _as_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


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


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    except Exception:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep reranker WeightConfig for strict NLQ metrics")
    parser.add_argument("--json_dir", required=True, help="Directory containing *_v03_decisions.json")
    parser.add_argument("--index_dir", required=True, help="Directory containing <video_uid>.index.npz/meta")
    parser.add_argument("--pattern", default="*_v03_decisions.json")
    parser.add_argument("--mode", choices=["hard_pseudo_nlq"], default="hard_pseudo_nlq")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--default-cfg", default=str(ROOT / "configs" / "rerank_default.yaml"))
    parser.add_argument("--hard-constraints", choices=["on", "off"], default="on")
    parser.add_argument(
        "--hard-constraints-cfg",
        default=str(ROOT / "configs" / "hard_constraints_default.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=6)
    parser.add_argument("--search", choices=["grid", "random"], default="random")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--metric",
        choices=["hit_at_k_strict", "hit_at_1_strict", "fp_rate", "objective_combo"],
        default="objective_combo",
    )
    parser.add_argument("--split-runs", type=int, default=5)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _discover_inputs(json_dir: Path, index_dir: Path, pattern: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for jpath in sorted(json_dir.glob(pattern)):
        uid = jpath.stem.replace("_v03_decisions", "")
        index_prefix = index_dir / uid
        if not (Path(str(index_prefix) + ".index.npz").exists() and Path(str(index_prefix) + ".index_meta.json").exists()):
            continue
        items.append(
            {
                "video_uid": uid,
                "json_path": jpath,
                "index_prefix": index_prefix,
            }
        )
    return items


def _budget_from_config(cfg_path: Path) -> dict[str, list[float | int]]:
    cfg = _load_yaml(cfg_path)
    budgets_cfg = dict(cfg.get("eval", {}).get("budgets", {}))
    max_total = max([float(x) for x in budgets_cfg.get("max_total_s", [60.0])])
    max_tokens = max([int(x) for x in budgets_cfg.get("max_tokens", [200])])
    max_decisions = max([int(x) for x in budgets_cfg.get("max_decisions", [12])])
    return {
        "max_total_s": [max_total],
        "max_tokens": [max_tokens],
        "max_decisions": [max_decisions],
    }


def _grid_candidates(base: WeightConfig, limit: int) -> list[WeightConfig]:
    values = {
        "bonus_intent_token_on_token": [0.8, 1.0, 1.2],
        "bonus_intent_decision_on_decision": [0.8, 1.0, 1.2],
        "penalty_distractor_near": [0.2, 0.3, 0.4],
        "distractor_near_window_s": [4.0, 5.0],
        "bonus_first": [0.3, 0.5, 0.7],
        "bonus_last": [0.3, 0.5, 0.7],
    }
    keys = list(values.keys())
    combos = list(itertools.product(*(values[k] for k in keys)))
    out: list[WeightConfig] = []
    for i, combo in enumerate(combos, start=1):
        data = base.to_dict()
        for k, v in zip(keys, combo):
            data[k] = v
        data["name"] = f"grid_{i:04d}"
        out.append(WeightConfig.from_dict(data))
        if len(out) >= max(1, int(limit)):
            break
    return out


def _random_candidates(base: WeightConfig, trials: int, seed: int) -> list[WeightConfig]:
    rng = random.Random(int(seed))
    out: list[WeightConfig] = []
    for i in range(max(0, int(trials))):
        data = base.to_dict()
        data["name"] = f"rand_{i + 1:04d}"
        data["bonus_intent_token_on_token"] = round(rng.uniform(0.5, 1.8), 4)
        data["bonus_intent_decision_on_decision"] = round(rng.uniform(0.5, 1.8), 4)
        data["bonus_intent_anchor_on_highlight"] = round(rng.uniform(0.6, 1.6), 4)
        data["bonus_first"] = round(rng.uniform(0.1, 1.0), 4)
        data["bonus_last"] = round(rng.uniform(0.1, 1.0), 4)
        data["penalty_distractor_near"] = round(rng.uniform(0.05, 0.8), 4)
        data["distractor_near_window_s"] = round(rng.uniform(2.0, 8.0), 4)
        data["bonus_conf_scale"] = round(rng.uniform(0.0, 0.4), 4)
        data["bonus_boundary_scale"] = round(rng.uniform(0.0, 0.4), 4)
        data["penalty_before_scene_change"] = round(rng.uniform(0.0, 2.0), 4)
        out.append(WeightConfig.from_dict(data))
    return out


def _split_stability(values_by_uid: dict[str, float], split_runs: int, seed: int) -> dict[str, float]:
    uids = sorted(values_by_uid.keys())
    if len(uids) < 2 or int(split_runs) <= 0:
        val = float(sum(values_by_uid.values()) / max(1, len(values_by_uid)))
        return {
            "split_train_mean": val,
            "split_train_std": 0.0,
            "split_test_mean": val,
            "split_test_std": 0.0,
        }
    rng = random.Random(int(seed))
    train_scores: list[float] = []
    test_scores: list[float] = []
    for _ in range(int(split_runs)):
        shuffled = list(uids)
        rng.shuffle(shuffled)
        cut = max(1, int(0.7 * len(shuffled)))
        train = shuffled[:cut]
        test = shuffled[cut:] if cut < len(shuffled) else shuffled[-1:]
        train_scores.append(float(sum(values_by_uid[u] for u in train) / max(1, len(train))))
        test_scores.append(float(sum(values_by_uid[u] for u in test) / max(1, len(test))))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    def _std(xs: list[float]) -> float:
        m = _mean(xs)
        return float((sum((x - m) ** 2 for x in xs) / max(1, len(xs))) ** 0.5)

    return {
        "split_train_mean": _mean(train_scores),
        "split_train_std": _std(train_scores),
        "split_test_mean": _mean(test_scores),
        "split_test_std": _std(test_scores),
    }


def _aggregate_by_key(rows: list[dict[str, Any]], key: str, variant: str = "full") -> dict[str, dict[str, float]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("variant", "")) != str(variant):
            continue
        k = str(row.get(key, ""))
        out.setdefault(k, []).append(row)
    result: dict[str, dict[str, float]] = {}
    for k, vals in out.items():
        n = float(len(vals))
        result[k] = {
            "hit_at_k_strict": float(sum(float(v.get("hit_at_k_strict", 0.0)) for v in vals) / n),
            "hit_at_1_strict": float(sum(float(v.get("hit_at_1_strict", 0.0)) for v in vals) / n),
            "fp_rate": float(sum(float(v.get("top1_in_distractor_rate", 0.0)) for v in vals) / n),
            "hit_at_k": float(sum(float(v.get("hit_at_k", 0.0)) for v in vals) / n),
            "mrr": float(sum(float(v.get("mrr", 0.0)) for v in vals) / n),
        }
    return result


def main() -> int:
    args = _parse_args()
    json_dir = Path(args.json_dir)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _discover_inputs(json_dir=json_dir, index_dir=index_dir, pattern=str(args.pattern))
    if not inputs:
        print("error=no_inputs_found")
        return 1

    budgets = _budget_from_config(Path(args.config))
    base_cfg = resolve_weight_config(Path(args.default_cfg))
    base_cfg.name = "default"
    if str(args.hard_constraints).lower() == "off":
        hard_cfg = HardConstraintConfig(
            enable_after_scene_change=False,
            enable_first_last=False,
            enable_type_match=False,
            relax_on_empty=True,
            relax_order=["after_scene_change", "first_last", "type_match"],
        )
    else:
        hard_cfg = HardConstraintConfig.from_yaml(Path(args.hard_constraints_cfg))

    prepared: list[dict[str, Any]] = []
    for item in inputs:
        output = _as_output(item["json_path"])
        samples: list[NLQSample] = load_hard_pseudo_nlq(
            output,
            seed=int(args.seed),
            n_highlight=max(1, int(args.n)),
            n_token=max(1, int(args.n)),
            n_decision=max(1, int(args.n)),
            top_k=max(1, int(args.top_k)),
        )
        if not samples:
            continue
        prepared.append(
            {
                **item,
                "output": output,
                "samples": samples,
            }
        )
    if not prepared:
        print("error=no_samples_generated")
        return 1

    candidates: list[WeightConfig] = [base_cfg]
    if str(args.search) == "grid":
        candidates.extend(_grid_candidates(base_cfg, limit=max(1, int(args.trials))))
    else:
        candidates.extend(_random_candidates(base_cfg, trials=max(1, int(args.trials) - 1), seed=int(args.seed)))

    results: list[dict[str, Any]] = []
    default_row: dict[str, Any] | None = None
    for idx, cfg in enumerate(candidates, start=1):
        per_video_objective: dict[str, float] = {}
        collected_overall: list[dict[str, Any]] = []
        collected_by_type: list[dict[str, Any]] = []
        for entry in prepared:
            output = entry["output"]
            samples = entry["samples"]
            result = evaluate_nlq_samples(
                output=output,
                samples=samples,
                budgets=budgets,
                sweep=False,
                retriever_config={},
                index_prefix=entry["index_prefix"],
                rerank_cfg=cfg,
                hard_constraints_cfg=hard_cfg,
                allow_gt_fallback=False,
            )
            overall_rows = result["overall_rows"]
            by_type_rows = result["by_query_type_rows"]
            collected_overall.extend(overall_rows)
            collected_by_type.extend(by_type_rows)

            m_full = summarize_variant_metrics(overall_rows, variant="full")
            objective = compute_objective(
                hit_at_k_strict=float(m_full["hit_at_k_strict"]),
                hit_at_1_strict=float(m_full["hit_at_1_strict"]),
                fp_rate=float(m_full["fp_rate"]),
                metric=str(args.metric),
            )
            per_video_objective[str(entry["video_uid"])] = float(objective)

        full = summarize_variant_metrics(collected_overall, variant="full")
        highlights_only = summarize_variant_metrics(collected_overall, variant="highlights_only")
        objective = compute_objective(
            hit_at_k_strict=float(full["hit_at_k_strict"]),
            hit_at_1_strict=float(full["hit_at_1_strict"]),
            fp_rate=float(full["fp_rate"]),
            metric=str(args.metric),
        )
        stability = _split_stability(per_video_objective, split_runs=int(args.split_runs), seed=int(args.seed) + idx)
        by_query = _aggregate_by_key(collected_by_type, "query_type", variant="full")
        by_bucket = _aggregate_by_key(collected_by_type, "duration_bucket", variant="full")
        fallback_rate = (
            float(sum(float(r.get("fallback_rate", 0.0)) for r in collected_overall) / max(1, len(collected_overall)))
            if collected_overall
            else 0.0
        )
        relax_after_scene = (
            float(
                sum(float(r.get("relax_rate_after_scene_change", 0.0)) for r in collected_overall)
                / max(1, len(collected_overall))
            )
            if collected_overall
            else 0.0
        )
        relax_first_last = (
            float(sum(float(r.get("relax_rate_first_last", 0.0)) for r in collected_overall) / max(1, len(collected_overall)))
            if collected_overall
            else 0.0
        )
        relax_type_match = (
            float(sum(float(r.get("relax_rate_type_match", 0.0)) for r in collected_overall) / max(1, len(collected_overall)))
            if collected_overall
            else 0.0
        )

        row = {
            "trial_id": idx,
            "search": str(args.search),
            "metric": str(args.metric),
            "cfg_name": str(cfg.name),
            "cfg_hash": str(cfg.short_hash()),
            "hard_constraints_enabled": str(args.hard_constraints).lower() == "on",
            "objective": float(objective),
            "full_hit_at_k_strict": float(full["hit_at_k_strict"]),
            "full_hit_at_1_strict": float(full["hit_at_1_strict"]),
            "full_fp_rate": float(full["fp_rate"]),
            "full_hit_at_k": float(full["hit_at_k"]),
            "full_mrr": float(full["mrr"]),
            "hl_only_hit_at_k_strict": float(highlights_only["hit_at_k_strict"]),
            "hl_only_fp_rate": float(highlights_only["fp_rate"]),
            "delta_hit_at_k_strict_vs_hl_only": float(full["hit_at_k_strict"] - highlights_only["hit_at_k_strict"]),
            "delta_fp_rate_vs_hl_only": float(full["fp_rate"] - highlights_only["fp_rate"]),
            "num_videos": len(prepared),
            "split_train_mean": float(stability["split_train_mean"]),
            "split_train_std": float(stability["split_train_std"]),
            "split_test_mean": float(stability["split_test_mean"]),
            "split_test_std": float(stability["split_test_std"]),
            "fallback_rate": float(fallback_rate),
            "relax_rate_after_scene_change": float(relax_after_scene),
            "relax_rate_first_last": float(relax_first_last),
            "relax_rate_type_match": float(relax_type_match),
            "by_query_type_json": json.dumps(by_query, ensure_ascii=False, sort_keys=True),
            "by_duration_bucket_json": json.dumps(by_bucket, ensure_ascii=False, sort_keys=True),
            "cfg_json": json.dumps(cfg.to_dict(), ensure_ascii=False, sort_keys=True),
            "hard_constraints_cfg_json": json.dumps(hard_cfg.to_dict(), ensure_ascii=False, sort_keys=True),
        }
        results.append(row)
        if str(cfg.name) == "default":
            default_row = row
        print(
            f"trial={idx}/{len(candidates)} cfg={cfg.name} objective={objective:.4f} "
            f"full_strict={full['hit_at_k_strict']:.4f} fp={full['fp_rate']:.4f}"
        )

    ranked = rank_rows_by_metric(results, metric=str(args.metric))
    best = ranked[0]
    best_cfg = WeightConfig.from_dict(json.loads(str(best["cfg_json"])))
    if not best_cfg.name:
        best_cfg.name = "best"

    results_csv = out_dir / "results_sweep.csv"
    _write_csv(results_csv, results)
    best_cfg_path = out_dir / "best_config.yaml"
    _write_yaml(best_cfg_path, best_cfg.to_dict())

    default_obj = float(default_row["objective"]) if default_row else 0.0
    best_obj = float(best["objective"])
    best_report = out_dir / "best_report.md"
    lines: list[str] = []
    lines.append("# Reranker Sweep Report")
    lines.append("")
    lines.append(f"- metric: {args.metric}")
    lines.append(f"- search: {args.search}")
    lines.append(f"- trials: {len(results)}")
    lines.append(f"- videos: {len(prepared)}")
    lines.append(f"- hard_constraints_enabled: {str(args.hard_constraints).lower() == 'on'}")
    lines.append(f"- hard_constraints_cfg: `{json.dumps(hard_cfg.to_dict(), ensure_ascii=False, sort_keys=True)}`")
    lines.append(f"- budget: {budgets}")
    lines.append(f"- default_objective: {default_obj:.4f}")
    lines.append(f"- best_objective: {best_obj:.4f}")
    lines.append(f"- delta_best_minus_default: {best_obj - default_obj:+.4f}")
    lines.append("")
    lines.append("## Best Config")
    lines.append("")
    lines.append(f"- name: {best_cfg.name}")
    lines.append(f"- hash: {best_cfg.short_hash()}")
    lines.append("```yaml")
    try:
        import yaml  # type: ignore

        lines.append(yaml.safe_dump(best_cfg.to_dict(), sort_keys=False, allow_unicode=True).strip())
    except Exception:
        lines.append(json.dumps(best_cfg.to_dict(), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Stability")
    lines.append("")
    lines.append(
        f"- split_train_mean/std: {float(best['split_train_mean']):.4f} / {float(best['split_train_std']):.4f}"
    )
    lines.append(f"- split_test_mean/std: {float(best['split_test_mean']):.4f} / {float(best['split_test_std']):.4f}")
    lines.append("")
    lines.append("## Best vs Default")
    lines.append("")
    lines.append("| metric | default | best | delta |")
    lines.append("|---|---:|---:|---:|")

    def _dv(key: str) -> float:
        return float(default_row.get(key, 0.0)) if default_row else 0.0

    def _bv(key: str) -> float:
        return float(best.get(key, 0.0))

    for key in ["full_hit_at_k_strict", "full_hit_at_1_strict", "full_fp_rate", "full_hit_at_k", "full_mrr", "objective"]:
        d = _dv(key)
        b = _bv(key)
        lines.append(f"| {key} | {d:.4f} | {b:.4f} | {b - d:+.4f} |")
    lines.append(
        f"| fallback_rate | {float(default_row.get('fallback_rate', 0.0)) if default_row else 0.0:.4f} | "
        f"{float(best.get('fallback_rate', 0.0)):.4f} | "
        f"{float(best.get('fallback_rate', 0.0)) - (float(default_row.get('fallback_rate', 0.0)) if default_row else 0.0):+.4f} |"
    )
    lines.append("")

    lines.append("## Per Query Type (Best, Full Variant)")
    lines.append("")
    lines.append("| query_type | hit@k_strict | hit@1_strict | fp_rate |")
    lines.append("|---|---:|---:|---:|")
    best_query = json.loads(str(best["by_query_type_json"]))
    for qtype in sorted(best_query.keys()):
        item = best_query[qtype]
        lines.append(
            f"| {qtype} | {float(item.get('hit_at_k_strict', 0.0)):.4f} | "
            f"{float(item.get('hit_at_1_strict', 0.0)):.4f} | {float(item.get('fp_rate', 0.0)):.4f} |"
        )
    lines.append("")

    lines.append("## Per Duration Bucket (Best, Full Variant)")
    lines.append("")
    lines.append("| duration_bucket | hit@k_strict | hit@1_strict | fp_rate |")
    lines.append("|---|---:|---:|---:|")
    best_bucket = json.loads(str(best["by_duration_bucket_json"]))
    for bucket in sorted(best_bucket.keys()):
        item = best_bucket[bucket]
        lines.append(
            f"| {bucket} | {float(item.get('hit_at_k_strict', 0.0)):.4f} | "
            f"{float(item.get('hit_at_1_strict', 0.0)):.4f} | {float(item.get('fp_rate', 0.0)):.4f} |"
        )
    lines.append("")

    best_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"results_saved={results_csv}")
    print(f"best_config_saved={best_cfg_path}")
    print(f"best_report_saved={best_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
