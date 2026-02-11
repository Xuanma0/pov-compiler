from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.eval_cross_variant import (
    build_budget_grid,
    evaluate_cross_variant,
    make_cross_report,
)
from pov_compiler.eval.fixed_queries import load_queries_jsonl
from pov_compiler.schemas import Output


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-variant evaluation with fixed query set")
    parser.add_argument("--json", required=True, help="Full output json path")
    parser.add_argument("--queries", required=True, help="Fixed queries jsonl path")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--sweep", action="store_true", help="Run full budget sweep")
    parser.add_argument("--max-total-s", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--no-per-query", action="store_true", help="Do not write results_per_query.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    eval_cfg = dict(cfg.get("eval", {}))
    retrieval_cfg = dict(cfg.get("retrieval", {}))

    output = _as_output(Path(args.json))
    queries = load_queries_jsonl(args.queries)
    if not queries:
        print("queries_total=0")
        print("error=queries file is empty")
        return 1

    budgets_cfg = dict(eval_cfg.get("budgets", {}))
    if not args.sweep:
        max_total_s = float(args.max_total_s if args.max_total_s is not None else max(budgets_cfg.get("max_total_s", [60])))
        max_tokens = int(args.max_tokens if args.max_tokens is not None else max(budgets_cfg.get("max_tokens", [200])))
        max_decisions = int(
            args.max_decisions if args.max_decisions is not None else max(budgets_cfg.get("max_decisions", [12]))
        )
        budgets_cfg = {
            "max_total_s": [max_total_s],
            "max_tokens": [max_tokens],
            "max_decisions": [max_decisions],
        }

    result = evaluate_cross_variant(
        full_output=output,
        queries=queries,
        budgets=budgets_cfg,
        sweep=args.sweep,
        retriever_config=retrieval_cfg,
    )
    overall_rows = result["overall_rows"]
    by_type_rows = result["by_query_type_rows"]
    per_query_rows = result["per_query_rows"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_csv = out_dir / "results_overall.csv"
    by_type_csv = out_dir / "results_by_query_type.csv"
    per_query_csv = out_dir / "results_per_query.csv"
    report_md = out_dir / "report.md"

    _write_csv(overall_csv, overall_rows)
    _write_csv(by_type_csv, by_type_rows)
    if not args.no_per_query:
        _write_csv(per_query_csv, per_query_rows)
    make_cross_report(
        overall_rows=overall_rows,
        by_query_type_rows=by_type_rows,
        report_path=report_md,
        overall_csv_path=overall_csv,
        by_type_csv_path=by_type_csv,
    )

    counts = Counter(q.type for q in queries)
    budget_grid = build_budget_grid(budgets_cfg, sweep=args.sweep)
    print(f"video_id={output.video_id}")
    print(f"queries_total={len(queries)}")
    print(f"queries_by_type={dict(sorted(counts.items()))}")
    print(f"budget_points={len(budget_grid)}")
    print(f"rows_overall={len(overall_rows)}")
    print(f"rows_by_query_type={len(by_type_rows)}")
    if not args.no_per_query:
        print(f"rows_per_query={len(per_query_rows)}")
    print(f"saved_overall={overall_csv}")
    print(f"saved_by_query_type={by_type_csv}")
    if not args.no_per_query:
        print(f"saved_per_query={per_query_csv}")
    print(f"saved_report={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
