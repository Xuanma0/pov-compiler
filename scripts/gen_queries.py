from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed queries from full output json")
    parser.add_argument("--json", required=True, help="Full output json path")
    parser.add_argument("--out", required=True, help="queries.jsonl path")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-time", "--n_time", dest="n_time", type=int, default=None)
    parser.add_argument("--n-anchor", "--n_anchor", dest="n_anchor", type=int, default=None)
    parser.add_argument("--n-token", "--n_token", dest="n_token", type=int, default=None)
    parser.add_argument("--n-decision", "--n_decision", dest="n_decision", type=int, default=None)
    parser.add_argument("--n-hard-time", "--n_hard_time", dest="n_hard_time", type=int, default=None)
    parser.add_argument("--time-window-s", "--time_window_s", dest="time_window_s", type=float, default=None)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
    parser.add_argument(
        "--hard-overlap-thresh",
        "--hard_overlap_thresh",
        dest="hard_overlap_thresh",
        type=float,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    eval_cfg = dict(cfg.get("eval", {}))
    fixed_cfg = dict(eval_cfg.get("fixed_queries", {}))

    output = _as_output(Path(args.json))
    queries = generate_fixed_queries(
        output=output,
        seed=int(args.seed if args.seed is not None else fixed_cfg.get("seed", 0)),
        n_time=int(args.n_time if args.n_time is not None else fixed_cfg.get("n_time", 10)),
        n_anchor=int(args.n_anchor if args.n_anchor is not None else fixed_cfg.get("n_anchor", 6)),
        n_token=int(args.n_token if args.n_token is not None else fixed_cfg.get("n_token", 10)),
        n_decision=int(args.n_decision if args.n_decision is not None else fixed_cfg.get("n_decision", 10)),
        n_hard_time=int(args.n_hard_time if args.n_hard_time is not None else fixed_cfg.get("n_hard_time", 10)),
        time_window_s=float(args.time_window_s if args.time_window_s is not None else eval_cfg.get("time_window_s", 8.0)),
        default_top_k=int(args.top_k if args.top_k is not None else eval_cfg.get("default_top_k", 6)),
        hard_overlap_thresh=float(
            args.hard_overlap_thresh
            if args.hard_overlap_thresh is not None
            else fixed_cfg.get("hard_overlap_thresh", 0.05)
        ),
    )
    save_queries_jsonl(queries, args.out)

    counts = Counter(query.type for query in queries)
    print(f"video_id={output.video_id}")
    print(f"queries_total={len(queries)}")
    print(f"counts_by_type={dict(sorted(counts.items()))}")
    print(f"saved={Path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
