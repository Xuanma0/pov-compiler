from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository import build_repo_chunks, deduplicate_chunks, select_chunks_for_query
from pov_compiler.repository.schema import RepoChunk
from pov_compiler.schemas import Output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoV0 smoke runner (write/read/dedup)")
    parser.add_argument("--json", required=True, help="Path to *_v03_decisions.json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--query", default="", help="Optional query for read-stage selection")
    parser.add_argument("--window-s", type=float, default=30.0, help="Window chunk size in seconds")
    parser.add_argument("--min-segment-s", type=float, default=5.0, help="Minimum segment chunk size in seconds")
    parser.add_argument("--strategy", choices=["importance_greedy", "recency_greedy"], default="importance_greedy")
    parser.add_argument("--max-repo-chunks", type=int, default=16)
    parser.add_argument("--max-repo-chars", type=int, default=6000)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--dedup-iou", type=float, default=0.6)
    return parser.parse_args()


def _model_validate_output(payload: dict[str, Any]) -> Output:
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _hash_cfg(cfg: dict[str, Any]) -> str:
    raw = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _by_scale(chunks: list[RepoChunk]) -> dict[str, int]:
    out: dict[str, int] = {}
    for chunk in chunks:
        key = str(chunk.scale)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    args = parse_args()
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    output = _model_validate_output(payload)

    repo_cfg = {
        "window_s": float(args.window_s),
        "min_segment_s": float(args.min_segment_s),
        "scales": {"event": True, "window": True, "segment": True},
    }
    dedup_cfg = {
        "iou_thresh": float(args.dedup_iou),
        "keep_best_importance": True,
    }
    read_budget = {
        "max_repo_chunks": int(args.max_repo_chunks),
        "max_repo_chars": int(args.max_repo_chars),
        "max_seconds": float(args.max_seconds) if args.max_seconds is not None else None,
        "repo_strategy": str(args.strategy),
    }

    raw_chunks = build_repo_chunks(output, cfg=repo_cfg)
    dedup_chunks = deduplicate_chunks(raw_chunks, cfg=dedup_cfg)
    selected = select_chunks_for_query(
        dedup_chunks,
        query=str(args.query or ""),
        budget=read_budget,
        cfg={"strategy": str(args.strategy)},
    )

    raw_rows = [_dump(c) for c in sorted(raw_chunks, key=lambda x: (float(x.t0), float(x.t1), str(x.id)))]
    dedup_rows = [_dump(c) for c in sorted(dedup_chunks, key=lambda x: (float(x.t0), float(x.t1), str(x.id)))]
    selected_rows = [_dump(c) for c in sorted(selected, key=lambda x: (float(x.t0), float(x.t1), str(x.id)))]

    repo_jsonl = out_dir / "repo_chunks.jsonl"
    selected_jsonl = out_dir / "repo_selected.jsonl"
    report_md = out_dir / "report.md"
    snapshot_json = out_dir / "snapshot.json"
    _write_jsonl(repo_jsonl, dedup_rows)
    _write_jsonl(selected_jsonl, selected_rows)

    before_n = len(raw_rows)
    after_n = len(dedup_rows)
    dedup_rate = 0.0 if before_n <= 0 else float(1.0 - (after_n / max(1, before_n)))
    selected_chars = int(sum(len(str(r.get("text", ""))) for r in selected_rows))
    selected_seconds = float(sum(max(0.0, float(r.get("t1", 0.0)) - float(r.get("t0", 0.0))) for r in selected_rows))
    scale_after = _by_scale(dedup_chunks)
    scale_before = _by_scale(raw_chunks)

    cfg_hash = _hash_cfg(
        {
            "repo_cfg": repo_cfg,
            "dedup_cfg": dedup_cfg,
            "read_budget": read_budget,
            "query": str(args.query or ""),
        }
    )
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "json": str(json_path),
            "video_id": str(output.video_id),
            "duration_s": float(output.meta.get("duration_s", 0.0) or 0.0),
        },
        "cfg": {
            "repo": repo_cfg,
            "dedup": dedup_cfg,
            "read_budget": read_budget,
            "query": str(args.query or ""),
            "cfg_hash": cfg_hash,
        },
        "stats": {
            "chunks_before_dedup": before_n,
            "chunks_after_dedup": after_n,
            "dedup_rate": dedup_rate,
            "by_scale_before": scale_before,
            "by_scale_after": scale_after,
            "selected_chunks": len(selected_rows),
            "selected_chars": selected_chars,
            "selected_seconds": selected_seconds,
        },
        "outputs": {
            "repo_chunks_jsonl": str(repo_jsonl),
            "repo_selected_jsonl": str(selected_jsonl),
            "report_md": str(report_md),
            "snapshot_json": str(snapshot_json),
        },
    }
    snapshot_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RepoV0 Smoke Report",
        "",
        f"- video_id: `{output.video_id}`",
        f"- cfg_hash: `{cfg_hash}`",
        f"- chunks_before_dedup: `{before_n}`",
        f"- chunks_after_dedup: `{after_n}`",
        f"- dedup_rate: `{dedup_rate:.4f}`",
        f"- by_scale_before: `{json.dumps(scale_before, ensure_ascii=False, sort_keys=True)}`",
        f"- by_scale_after: `{json.dumps(scale_after, ensure_ascii=False, sort_keys=True)}`",
        f"- selected_chunks: `{len(selected_rows)}`",
        f"- selected_chars: `{selected_chars}`",
        f"- selected_seconds: `{selected_seconds:.2f}`",
        "",
        "## Outputs",
        "",
        f"- `{repo_jsonl}`",
        f"- `{selected_jsonl}`",
        f"- `{snapshot_json}`",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"video_id={output.video_id}")
    print(f"chunks_before_dedup={before_n}")
    print(f"chunks_after_dedup={after_n}")
    print(f"selected_chunks={len(selected_rows)}")
    print(f"saved_repo_chunks={repo_jsonl}")
    print(f"saved_report={report_md}")
    print(f"saved_snapshot={snapshot_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
