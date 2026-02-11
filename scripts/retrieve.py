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
    parser = argparse.ArgumentParser(description="Retrieve events/highlights and optionally build context")
    parser.add_argument("--json", required=True, help="Pipeline output json path")
    parser.add_argument("--query", required=True, help='Query string, e.g. "anchor=turn_head top_k=6"')
    parser.add_argument("--index", default=None, help="Index prefix path")
    parser.add_argument("--out", default=None, help="Output context path (optional)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="Config YAML path")
    parser.add_argument(
        "--mode",
        choices=["timeline", "highlights", "decisions", "full"],
        default=None,
        help="Context mode override",
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-highlights", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    context_cfg = dict(cfg.get("context_default", {}))

    retriever = Retriever(
        output_json=Path(args.json),
        index=Path(args.index) if args.index else None,
        config=retrieval_cfg,
    )
    result = retriever.retrieve(args.query)

    selected_events = result.get("selected_events", [])
    selected_highlights = result.get("selected_highlights", [])
    selected_decisions = result.get("selected_decisions", [])
    selected_tokens = result.get("selected_tokens", [])
    reason = result.get("debug", {}).get("reason", "")

    print(f"selected_events={len(selected_events)}")
    print(f"selected_highlights={len(selected_highlights)}")
    print(f"selected_decisions={len(selected_decisions)}")
    print(f"selected_tokens={len(selected_tokens)}")
    print(f"reason={reason}")

    if args.out:
        mode = args.mode or result.get("mode") or context_cfg.get("mode", "highlights")
        budget: dict[str, Any] = {}
        for key in ("max_events", "max_highlights", "max_tokens", "max_seconds"):
            if key in context_cfg:
                budget[key] = context_cfg[key]
        budget.update(result.get("budget_overrides", {}))
        if args.max_events is not None:
            budget["max_events"] = int(args.max_events)
        if args.max_highlights is not None:
            budget["max_highlights"] = int(args.max_highlights)
        if args.max_decisions is not None:
            budget["max_decisions"] = int(args.max_decisions)
        if args.max_tokens is not None:
            budget["max_tokens"] = int(args.max_tokens)

        context = build_context(
            output_json=Path(args.json),
            mode=str(mode),
            budget=budget,
            selected_events=list(selected_events),
            selected_highlights=list(selected_highlights),
            selected_decisions=list(selected_decisions),
            selected_tokens=list(selected_tokens),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"tokens_after={len(context.get('tokens', []))}")
        print(f"decisions_after={len(context.get('decision_points', []))}")
        print(f"saved={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
