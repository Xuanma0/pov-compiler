from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import NLQSample, load_hard_pseudo_nlq
from pov_compiler.bench.nlq.evaluator import evaluate_nlq_samples
from pov_compiler.bench.nlq.sweep_utils import summarize_variant_metrics
from pov_compiler.retrieval.constraints import HardConstraintConfig
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.schemas import Output


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


def _discover_inputs(json_dir: Path, index_dir: Path, pattern: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for jpath in sorted(json_dir.glob(pattern)):
        uid = jpath.stem.replace("_v03_decisions", "")
        prefix = index_dir / uid
        if not (Path(str(prefix) + ".index.npz").exists() and Path(str(prefix) + ".index_meta.json").exists()):
            continue
        out.append({"video_uid": uid, "json_path": jpath, "index_prefix": prefix})
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate reranker terms on hard_pseudo_nlq")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--pattern", default="*_v03_decisions.json")
    parser.add_argument("--cfg", default=str(ROOT / "configs" / "rerank_default.yaml"))
    parser.add_argument("--hard-constraints-cfg", default=str(ROOT / "configs" / "hard_constraints_default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=6)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _budget() -> dict[str, list[int | float]]:
    return {"max_total_s": [60.0], "max_tokens": [200], "max_decisions": [12]}


def _make_variant_cfg(base: WeightConfig, term: str) -> tuple[WeightConfig, HardConstraintConfig]:
    cfg = WeightConfig.from_dict(base.to_dict())
    cfg.name = f"ablate_{term}"
    hard = HardConstraintConfig()

    if term == "intent_bonus":
        for key in list(cfg.to_dict().keys()):
            if key.startswith("bonus_intent_"):
                setattr(cfg, key, 0.0)
    elif term == "distractor_penalty":
        cfg.penalty_distractor_near = 0.0
    elif term == "match_mismatch":
        cfg.bonus_anchor_highlight_match = 0.0
        cfg.bonus_anchor_decision_match = 0.0
        cfg.penalty_anchor_highlight_mismatch = 0.0
        cfg.penalty_anchor_decision_mismatch = 0.0
        cfg.bonus_token_match = 0.0
        cfg.bonus_token_highlight_overlap = 0.0
        cfg.penalty_token_mismatch = 0.0
        cfg.bonus_decision_match = 0.0
        cfg.penalty_decision_mismatch = 0.0
    elif term == "constraint_hard_filter":
        hard = HardConstraintConfig(
            enable_after_scene_change=False,
            enable_first_last=False,
            enable_type_match=False,
            relax_on_empty=True,
            relax_order=["after_scene_change", "first_last", "type_match"],
        )
    return cfg, hard


def main() -> int:
    args = _parse_args()
    json_dir = Path(args.json_dir)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = resolve_weight_config(Path(args.cfg))
    base_cfg.name = "base"
    base_hard = HardConstraintConfig.from_yaml(Path(args.hard_constraints_cfg))

    inputs = _discover_inputs(json_dir, index_dir, str(args.pattern))
    if not inputs:
        print("error=no_inputs")
        return 1

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
        prepared.append({**item, "output": output, "samples": samples})
    if not prepared:
        print("error=no_samples")
        return 1

    terms = ["baseline", "intent_bonus", "distractor_penalty", "match_mismatch", "constraint_hard_filter"]
    rows: list[dict[str, Any]] = []
    for term in terms:
        if term == "baseline":
            cfg = base_cfg
            hard_cfg = base_hard
        else:
            cfg, hard_cfg = _make_variant_cfg(base_cfg, term)

        overall_all: list[dict[str, Any]] = []
        for entry in prepared:
            result = evaluate_nlq_samples(
                output=entry["output"],
                samples=entry["samples"],
                budgets=_budget(),
                sweep=False,
                retriever_config={},
                index_prefix=entry["index_prefix"],
                rerank_cfg=cfg,
                hard_constraints_cfg=hard_cfg,
                allow_gt_fallback=False,
            )
            overall_all.extend(result["overall_rows"])

        full = summarize_variant_metrics(overall_all, variant="full")
        rows.append(
            {
                "term": term,
                "cfg_name": cfg.name,
                "cfg_hash": cfg.short_hash(),
                "hard_constraints_enabled": bool(
                    hard_cfg.enable_after_scene_change or hard_cfg.enable_first_last or hard_cfg.enable_type_match
                ),
                "hit_at_k_strict": float(full["hit_at_k_strict"]),
                "hit_at_1_strict": float(full["hit_at_1_strict"]),
                "fp_rate": float(full["fp_rate"]),
                "hit_at_k": float(full["hit_at_k"]),
                "mrr": float(full["mrr"]),
            }
        )
        print(
            f"term={term} strict={full['hit_at_k_strict']:.4f} hit1_strict={full['hit_at_1_strict']:.4f} fp={full['fp_rate']:.4f}"
        )

    baseline = next((r for r in rows if r["term"] == "baseline"), None)
    for row in rows:
        if baseline is None:
            row["delta_hit_at_k_strict"] = 0.0
            row["delta_hit_at_1_strict"] = 0.0
            row["delta_fp_rate"] = 0.0
        else:
            row["delta_hit_at_k_strict"] = float(row["hit_at_k_strict"]) - float(baseline["hit_at_k_strict"])
            row["delta_hit_at_1_strict"] = float(row["hit_at_1_strict"]) - float(baseline["hit_at_1_strict"])
            row["delta_fp_rate"] = float(row["fp_rate"]) - float(baseline["fp_rate"])

    csv_path = out_dir / "table_ablation_terms.csv"
    _write_csv(csv_path, rows)

    report = out_dir / "report.md"
    lines: list[str] = []
    lines.append("# Reranker Terms Ablation")
    lines.append("")
    lines.append("| term | hit@k_strict | hit@1_strict | fp_rate | delta_hit@k_strict | delta_fp_rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['term']} | {float(row['hit_at_k_strict']):.4f} | {float(row['hit_at_1_strict']):.4f} | "
            f"{float(row['fp_rate']):.4f} | {float(row['delta_hit_at_k_strict']):+.4f} | {float(row['delta_fp_rate']):+.4f} |"
        )
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved_table={csv_path}")
    print(f"saved_report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
