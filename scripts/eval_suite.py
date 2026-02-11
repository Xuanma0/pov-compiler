from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.eval.eval_cross_variant import evaluate_cross_variant, make_cross_report
from pov_compiler.eval.fixed_queries import generate_fixed_queries, save_queries_jsonl
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


def _sanitize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch cross-variant eval with fixed query sets")
    parser.add_argument("--json_dir", required=True, help="Directory containing output json files")
    parser.add_argument("--pattern", default="*_v03_decisions.json", help="Glob pattern under json_dir")
    parser.add_argument("--out", required=True, help="Output results_overall.csv path")
    parser.add_argument("--report", default=None, help="Output report.md path")
    parser.add_argument("--queries_dir", default=None, help="Directory to save per-video queries jsonl")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--sweep", action="store_true", help="Run full budget sweep")
    parser.add_argument("--no-per-query", action="store_true", help="Do not write results_per_query.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    eval_cfg = dict(cfg.get("eval", {}))
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    fixed_cfg = dict(eval_cfg.get("fixed_queries", {}))
    budgets = dict(eval_cfg.get("budgets", {}))

    json_dir = Path(args.json_dir)
    files = sorted(json_dir.glob(args.pattern))
    if not files:
        print("matched_files=0")
        return 1

    overall_rows: list[dict[str, Any]] = []
    by_type_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    query_type_counter: Counter[str] = Counter()

    out_path = Path(args.out)
    out_dir = out_path.parent
    queries_dir = Path(args.queries_dir) if args.queries_dir else (out_dir / "queries")
    queries_dir.mkdir(parents=True, exist_ok=True)

    for json_path in files:
        output = _as_output(json_path)
        queries = generate_fixed_queries(
            output=output,
            seed=int(fixed_cfg.get("seed", 0)),
            n_time=int(fixed_cfg.get("n_time", 10)),
            n_anchor=int(fixed_cfg.get("n_anchor", 6)),
            n_token=int(fixed_cfg.get("n_token", 10)),
            n_decision=int(fixed_cfg.get("n_decision", 10)),
            n_hard_time=int(fixed_cfg.get("n_hard_time", 10)),
            time_window_s=float(eval_cfg.get("time_window_s", 8.0)),
            default_top_k=int(eval_cfg.get("default_top_k", 6)),
            hard_overlap_thresh=float(fixed_cfg.get("hard_overlap_thresh", 0.05)),
        )
        queries_path = queries_dir / f"{_sanitize(output.video_id)}.queries.jsonl"
        save_queries_jsonl(queries, queries_path)
        query_type_counter.update(query.type for query in queries)

        result = evaluate_cross_variant(
            full_output=output,
            queries=queries,
            budgets=budgets,
            sweep=args.sweep,
            retriever_config=retrieval_cfg,
        )
        overall_rows.extend(result["overall_rows"])
        by_type_rows.extend(result["by_query_type_rows"])
        per_query_rows.extend(result["per_query_rows"])

    by_type_path = out_path.with_name("results_by_query_type.csv")
    per_query_path = out_path.with_name("results_per_query.csv")
    report_path = Path(args.report) if args.report else out_path.with_name("report.md")

    _write_csv(out_path, overall_rows)
    _write_csv(by_type_path, by_type_rows)
    if not args.no_per_query:
        _write_csv(per_query_path, per_query_rows)
    make_cross_report(
        overall_rows=overall_rows,
        by_query_type_rows=by_type_rows,
        report_path=report_path,
        overall_csv_path=out_path,
        by_type_csv_path=by_type_path,
    )

    print(f"matched_files={len(files)}")
    print(f"queries_generated={sum(query_type_counter.values())}")
    print(f"queries_by_type={dict(sorted(query_type_counter.items()))}")
    print(f"rows_overall={len(overall_rows)}")
    print(f"rows_by_query_type={len(by_type_rows)}")
    if not args.no_per_query:
        print(f"rows_per_query={len(per_query_rows)}")
    print(f"saved_overall={out_path}")
    print(f"saved_by_query_type={by_type_path}")
    if not args.no_per_query:
        print(f"saved_per_query={per_query_path}")
    print(f"saved_report={report_path}")
    print(f"saved_queries_dir={queries_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
