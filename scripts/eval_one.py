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

from pov_compiler.eval.ablation import ALL_VARIANTS, apply_variant
from pov_compiler.eval.budget_sweep import apply_budget, run_budget_sweep
from pov_compiler.eval.metrics import evaluate_output
from pov_compiler.schemas import Output


REQUIRED_COLUMNS = [
    "video_id",
    "variant",
    "budget_max_total_s",
    "budget_max_tokens",
    "budget_max_decisions",
    "compression_ratio",
    "coverage_ratio",
    "hit_at_k",
    "mrr",
    "tokens_total",
    "decisions_total",
    "highlights_total",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        return data
    return {}


def _as_output(path: Path) -> Output:
    data = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(data)  # type: ignore[attr-defined]
    return Output.parse_obj(data)


def _append_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(REQUIRED_COLUMNS)
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one output JSON")
    parser.add_argument("--json", required=True, help="Input output json")
    parser.add_argument("--variant", choices=ALL_VARIANTS, default="full")
    parser.add_argument("--out", required=True, help="Results csv path")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--index", default=None, help="Optional index prefix for efficiency metadata")
    parser.add_argument("--sweep", action="store_true", help="Run budget sweep")
    parser.add_argument("--max-total-s", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    return parser.parse_args()


def _single_budget(eval_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    budget_cfg = dict(eval_cfg.get("budgets", {}))
    default_total_s = max([float(x) for x in budget_cfg.get("max_total_s", [60])])
    default_tokens = max([int(x) for x in budget_cfg.get("max_tokens", [200])])
    default_decisions = max([int(x) for x in budget_cfg.get("max_decisions", [12])])
    return {
        "max_total_s": float(args.max_total_s if args.max_total_s is not None else default_total_s),
        "max_tokens": int(args.max_tokens if args.max_tokens is not None else default_tokens),
        "max_decisions": int(args.max_decisions if args.max_decisions is not None else default_decisions),
    }


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    eval_cfg = dict(cfg.get("eval", {}))
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    output = _as_output(Path(args.json))

    rows: list[dict[str, Any]] = []
    if args.sweep:
        rows = run_budget_sweep(
            output=output,
            variant=args.variant,
            budgets=dict(eval_cfg.get("budgets", {})),
            eval_config=eval_cfg,
            retriever_config=retrieval_cfg,
            index_prefix=args.index,
        )
    else:
        budget = _single_budget(eval_cfg, args)
        variant_output = apply_variant(output, variant=args.variant)
        budgeted_output = apply_budget(variant_output, budget=budget)
        metrics = evaluate_output(
            output=budgeted_output,
            eval_config=eval_cfg,
            retriever_config=retrieval_cfg,
            index_prefix=args.index,
        )
        rows = [
            {
                "video_id": output.video_id,
                "variant": args.variant,
                "budget_max_total_s": float(budget["max_total_s"]),
                "budget_max_tokens": int(budget["max_tokens"]),
                "budget_max_decisions": int(budget["max_decisions"]),
                **metrics,
            }
        ]

    _append_rows_csv(Path(args.out), rows)
    if rows:
        sample = rows[0]
        print(f"rows_written={len(rows)}")
        print(f"variant={sample.get('variant')}")
        print(f"compression_ratio={float(sample.get('compression_ratio', 0.0)):.4f}")
        print(f"coverage_ratio={float(sample.get('coverage_ratio', 0.0)):.4f}")
        print(f"hit_at_k={float(sample.get('hit_at_k', 0.0)):.4f}")
        print(f"mrr={float(sample.get('mrr', 0.0)):.4f}")
    print(f"saved={Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
