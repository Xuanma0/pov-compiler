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
from pov_compiler.retrieval.constraints import HardConstraintConfig, apply_constraints_detailed
from pov_compiler.retrieval.query_planner import plan as plan_query
from pov_compiler.retrieval.rerank_debug import explain_scores
from pov_compiler.retrieval.reranker import rerank
from pov_compiler.retrieval.reranker_config import WeightConfig, resolve_weight_config
from pov_compiler.retrieval.retriever import Retriever
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _force_top_k(query: str, top_k: int) -> str:
    text = str(query).strip()
    if not text:
        return ""
    if "top_k=" in text:
        parts = [x for x in text.split() if not x.startswith("top_k=")]
        return f"{' '.join(parts)} top_k={int(max(1, top_k))}".strip()
    return f"{text} top_k={int(max(1, top_k))}"


def _build_candidate_queries(sample: NLQSample) -> list[str]:
    planned = plan_query(sample.query)
    out: list[str] = []
    seen: set[str] = set()
    for cand in sorted(planned.candidates, key=lambda x: (int(x.get("priority", 100)), str(x.get("query", "")))):
        q = _force_top_k(str(cand.get("query", "")), int(sample.top_k))
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reranker score decomposition debug")
    parser.add_argument("--json", required=True, help="Pipeline output json")
    parser.add_argument("--index", default=None, help="Vector index prefix")
    parser.add_argument("--mode", choices=["hard_pseudo_nlq"], default="hard_pseudo_nlq")
    parser.add_argument("--n", type=int, default=20, help="Sample count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=6)
    parser.add_argument("--rerank-cfg", default=None)
    parser.add_argument("--hard-constraints-cfg", default=str(ROOT / "configs" / "hard_constraints_default.yaml"))
    parser.add_argument("--out", default=str(ROOT / "data" / "outputs" / "rerank_debug" / "debug.csv"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = _as_output(Path(args.json))
    cfg_yaml = _load_yaml(Path(args.config))
    retrieval_cfg = dict(cfg_yaml.get("retrieval", {}))
    rerank_cfg_yaml = cfg_yaml.get("reranker", {})
    if args.rerank_cfg:
        weights = resolve_weight_config(Path(args.rerank_cfg))
    elif isinstance(rerank_cfg_yaml, dict) and rerank_cfg_yaml:
        weights = resolve_weight_config(rerank_cfg_yaml)
    else:
        weights = WeightConfig()
    hard_cfg = (
        HardConstraintConfig.from_yaml(Path(args.hard_constraints_cfg))
        if args.hard_constraints_cfg and Path(args.hard_constraints_cfg).exists()
        else HardConstraintConfig()
    )

    samples = load_hard_pseudo_nlq(
        output,
        seed=int(args.seed),
        n_highlight=max(1, int(args.n)),
        n_token=max(1, int(args.n)),
        n_decision=max(1, int(args.n)),
        top_k=max(1, int(args.top_k)),
    )
    if not samples:
        print("error=no_samples")
        return 1

    retriever = Retriever(output_json=output, index=args.index, config=retrieval_cfg)
    rows: list[dict[str, Any]] = []
    for sample in samples[: max(1, int(args.n))]:
        planned = plan_query(sample.query)
        candidates = _build_candidate_queries(sample)
        hits = retriever.retrieve_multi(candidates)
        cresult = apply_constraints_detailed(hits, query_plan=planned, cfg=hard_cfg, output=output)
        explained = explain_scores(
            cresult.hits,
            plan=planned,
            cfg=weights,
            context=output,
            distractors=sample.distractors,
        )
        reranked = rerank(cresult.hits, plan=planned, context=output, cfg=weights, distractors=sample.distractors)
        for rank, item in enumerate(explained, start=1):
            rows.append(
                {
                    "qid": sample.qid,
                    "query": sample.query,
                    "query_type": sample.query_type,
                    "intent": planned.intent,
                    "constraints": json.dumps(planned.constraints, ensure_ascii=False, sort_keys=True),
                    "candidate_count": len(candidates),
                    "merged_hits": len(hits),
                    "filtered_hits": len(cresult.hits),
                    "used_fallback": bool(cresult.used_fallback),
                    "rank_explain": rank,
                    "rank_rerank_top1_id": str(reranked[0]["id"]) if reranked else "",
                    **item,
                }
            )

    if not rows:
        print("error=no_debug_rows")
        return 1

    out_path = Path(args.out)
    _write_csv(out_path, rows)

    nonzero = [row for row in rows if abs(float(row.get("intent_bonus", 0.0))) > 1e-9]
    top1_rows = [row for row in rows if int(row.get("rank_explain", 9999)) == 1]
    top1_nonzero = [row for row in top1_rows if abs(float(row.get("intent_bonus", 0.0))) > 1e-9]
    ratio = float(len(nonzero) / max(1, len(rows)))
    top1_ratio = float(len(top1_nonzero) / max(1, len(top1_rows)))
    mean_bonus = float(sum(float(row.get("intent_bonus", 0.0)) for row in rows) / max(1, len(rows)))
    print(f"rows={len(rows)}")
    print(f"samples={min(len(samples), max(1, int(args.n)))}")
    print(f"intent_bonus_nonzero_ratio={ratio:.4f}")
    print(f"intent_bonus_nonzero_ratio_top1={top1_ratio:.4f}")
    print(f"intent_bonus_mean={mean_bonus:.4f}")
    if ratio <= 1e-9:
        print("warning=intent_bonus_all_zero check planner intent or cfg bonus values")
    print(f"saved_debug={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

