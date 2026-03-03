from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.context.context_builder import build_context
from pov_compiler.repository import build_repo_chunks, deduplicate_chunks
from pov_compiler.schemas import Output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repo summary smoke (summary_v0 + repo selection)")
    parser.add_argument("--json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--repo-write-policy", default="multiscale+summary_v0")
    parser.add_argument("--provider", default="fake", choices=["fake", "openai", "openai_compat", "gemini", "qwen", "deepseek", "glm"])
    parser.add_argument("--model", default="fake-summary-v0")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget", default="20/50/4")
    parser.add_argument("--query", default="anchor=turn_head top_k=6")
    return parser.parse_args()


def _read_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _parse_budget(raw: str) -> tuple[float, int, int]:
    parts = [x.strip() for x in str(raw).split("/") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"invalid budget: {raw}")
    return float(parts[0]), int(parts[1]), int(parts[2])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    in_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = _read_output(in_path)
    b_s, b_tok, b_dec = _parse_budget(args.budget)

    repo_cfg = {
        "write_policy": {"name": str(args.repo_write_policy), "chunk_step_s": 0.0, "summary_window_s": 60.0},
        "summary": {
            "enabled": True,
            "window_s": 60.0,
            "model": {
                "enabled": not bool(args.dry_run),
                "provider": str(args.provider),
                "model": str(args.model),
                "base_url": args.base_url,
                "api_key_env": str(args.api_key_env or ""),
                "timeout_s": 60,
                "max_retries": 1,
                "max_tokens": 400,
                "temperature": 0.2,
                "model_cache_enabled": True,
                "model_cache_dir": "data/outputs/model_cache",
            },
        },
        "scales": {"event": True, "decision": True, "place": True, "window": True, "segment": True},
        "window_s": 30.0,
        "min_segment_s": 5.0,
        "dedup": {"iou_thresh": 0.6, "sim_thresh": 0.9, "cross_scale": True, "keep_best_importance": True},
    }

    chunks_raw = build_repo_chunks(output, cfg=repo_cfg)
    chunks = deduplicate_chunks(chunks_raw, cfg=repo_cfg.get("dedup", {}))
    rows = [_dump(c) for c in chunks]
    summary_rows = [r for r in rows if str(r.get("level", r.get("scale", ""))).lower() == "summary"]
    output.repository = {
        "chunks": rows,
        "summary": {
            "chunks_before_dedup": len(chunks_raw),
            "chunks_after_dedup": len(chunks),
            "summary_chunks": len(summary_rows),
        },
        "cfg": repo_cfg,
    }
    context = build_context(
        output,
        mode="repo_only",
        budget={
            "use_repo": True,
            "max_total_s": float(b_s),
            "max_tokens": int(b_tok),
            "max_decisions": int(b_dec),
            "max_seconds": float(b_s),
            "max_repo_chunks": max(4, min(64, int(b_tok // 8) if b_tok > 0 else 16)),
            "max_repo_chars": int(max(1200, b_tok * 36)),
            "max_repo_tokens": int(b_tok),
            "repo_read_policy": "query_aware",
            "repo_strategy": "importance_greedy",
            "repo_query": str(args.query),
        },
        query_info={"query": str(args.query), "top_k": 6},
    )
    selected_rows = list(context.get("repo_chunks", []))
    context_text = "\n".join(f"- {str(r.get('text', ''))}" for r in selected_rows)

    repo_chunks_path = out_dir / "repo_chunks.jsonl"
    repo_selected_path = out_dir / "repo_selected.jsonl"
    repo_summaries_path = out_dir / "repo_summaries.jsonl"
    context_path = out_dir / "context.txt"
    report_path = out_dir / "report.md"
    snapshot_path = out_dir / "snapshot.json"
    _write_jsonl(repo_chunks_path, rows)
    _write_jsonl(repo_selected_path, selected_rows)
    _write_jsonl(repo_summaries_path, summary_rows)
    context_path.write_text(context_text, encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json": str(in_path),
            "video_id": str(output.video_id),
            "repo_write_policy": str(args.repo_write_policy),
            "provider": str(args.provider),
            "model": str(args.model),
            "dry_run": bool(args.dry_run),
            "api_key_env": str(args.api_key_env or ""),
            "base_url": str(args.base_url or ""),
            "budget": str(args.budget),
            "query": str(args.query),
        },
        "stats": {
            "chunks_total": len(rows),
            "summary_chunks": len(summary_rows),
            "selected_chunks": len(selected_rows),
            "selected_summary_chunks": sum(
                1 for r in selected_rows if str(r.get("level", r.get("scale", ""))).lower() == "summary"
            ),
            "repo_trace": dict(context.get("repo_trace", {})),
        },
        "outputs": {
            "repo_chunks_jsonl": str(repo_chunks_path),
            "repo_selected_jsonl": str(repo_selected_path),
            "repo_summaries_jsonl": str(repo_summaries_path),
            "context_txt": str(context_path),
            "report_md": str(report_path),
            "snapshot_json": str(snapshot_path),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Repo Summary Smoke",
        "",
        f"- video_id: `{output.video_id}`",
        f"- policy: `{args.repo_write_policy}`",
        f"- provider/model: `{args.provider}/{args.model}`",
        f"- dry_run: `{str(bool(args.dry_run)).lower()}`",
        f"- chunks_total: `{len(rows)}`",
        f"- summary_chunks: `{len(summary_rows)}`",
        f"- selected_chunks: `{len(selected_rows)}`",
        "",
        "## Outputs",
        "",
        f"- `{repo_chunks_path}`",
        f"- `{repo_selected_path}`",
        f"- `{repo_summaries_path}`",
        f"- `{context_path}`",
        f"- `{snapshot_path}`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"video_id={output.video_id}")
    print(f"chunks_total={len(rows)}")
    print(f"summary_chunks={len(summary_rows)}")
    print(f"saved_repo_summaries={repo_summaries_path}")
    print(f"saved_context={context_path}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

