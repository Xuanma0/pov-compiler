from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.context.context_builder import build_context
from pov_compiler.retrieval.retriever import Retriever


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal context JSON from pipeline output")
    parser.add_argument("--json", required=True, help="Path to pipeline output JSON")
    parser.add_argument("--out", required=True, help="Path to context output JSON")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Config YAML path")
    parser.add_argument("--mode", choices=["timeline", "highlights", "decisions", "full", "repo_only", "events_plus_repo"], default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-highlights", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--query", default=None, help='Optional retrieval query, e.g. "anchor=turn_head top_k=6"')
    parser.add_argument("--index", default=None, help="Optional index prefix for vector retrieval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    context_cfg = dict(cfg.get("context_default", {}))
    retrieval_cfg = dict(cfg.get("retrieval", {}))

    budget: dict[str, Any] = {}
    for key in ("max_events", "max_highlights", "max_tokens", "max_seconds"):
        if key in context_cfg:
            budget[key] = context_cfg[key]
    selected_events: list[str] | None = None
    selected_highlights: list[str] | None = None
    selected_tokens: list[str] | None = None
    selected_decisions: list[str] | None = None

    mode = args.mode or context_cfg.get("mode", "highlights")

    if args.query:
        retriever = Retriever(
            output_json=Path(args.json),
            index=Path(args.index) if args.index else None,
            config=retrieval_cfg,
        )
        result = retriever.retrieve(args.query)
        mode = args.mode or result.get("mode") or mode
        budget.update(result.get("budget_overrides", {}))
        selected_events = list(result.get("selected_events", []))
        selected_highlights = list(result.get("selected_highlights", []))
        selected_tokens = list(result.get("selected_tokens", []))
        selected_decisions = list(result.get("selected_decisions", []))

    if args.max_events is not None:
        budget["max_events"] = int(args.max_events)
    if args.max_highlights is not None:
        budget["max_highlights"] = int(args.max_highlights)
    if args.max_decisions is not None:
        budget["max_decisions"] = int(args.max_decisions)
    if args.max_tokens is not None:
        budget["max_tokens"] = int(args.max_tokens)
    if args.max_seconds is not None:
        budget["max_seconds"] = float(args.max_seconds)

    context = build_context(
        output_json=Path(args.json),
        mode=str(mode),
        budget=budget,
        selected_events=selected_events,
        selected_highlights=selected_highlights,
        selected_tokens=selected_tokens,
        selected_decisions=selected_decisions,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)

    token_stats = context.get("token_stats", {})
    print(f"tokens_before={token_stats.get('before', 0)}")
    print(f"tokens_after={token_stats.get('after', 0)}")
    print(f"highlights_kept={len(context.get('highlights', []))}")
    print(f"decisions_kept={len(context.get('decision_points', []))}")
    if args.query:
        print(f"selected_events={len(selected_events or [])}")
        print(f"selected_highlights={len(selected_highlights or [])}")
        print(f"selected_decisions={len(selected_decisions or [])}")
    print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
