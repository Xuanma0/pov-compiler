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

from pov_compiler.repository import build_repo_chunks, deduplicate_chunks, policy_cfg_hash, select_chunks_for_query
from pov_compiler.repository.policy import load_policy_yaml
from pov_compiler.repository.schema import RepoChunk
from pov_compiler.schemas import Output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoV1 policy smoke (write/read/dedup)")
    parser.add_argument("--json", required=True, help="Path to *_v03_decisions.json")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--repo-cfg", default="", help="Optional repo config yaml/json (default: configs/repo_default.yaml)")
    parser.add_argument("--query", action="append", default=[], help="Query for read-stage selection (repeatable)")
    parser.add_argument("--max-repo-chunks", type=int, default=None)
    parser.add_argument("--max-repo-tokens", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
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


def _load_repo_cfg(path_hint: str) -> dict[str, Any]:
    candidates = []
    if path_hint:
        candidates.append(Path(path_hint))
    candidates.append(ROOT / "configs" / "repo_default.yaml")
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = load_policy_yaml(path)
        except Exception:
            continue
        repo_cfg = dict(payload.get("repo", payload))
        if isinstance(repo_cfg, dict):
            return repo_cfg
    # Safe built-in fallback.
    return {
        "enable": True,
        "write_policy": {"name": "fixed_interval", "chunk_step_s": 8.0, "keep_levels": ["decision", "place"]},
        "read_policy": {"name": "budgeted_topk", "max_chunks": 16, "max_tokens": 200},
        "scales": {"event": True, "decision": True, "place": True, "window": True, "segment": True},
        "window_s": 30.0,
        "min_segment_s": 5.0,
        "dedup": {"iou_thresh": 0.6, "sim_thresh": 0.9, "cross_scale": True, "keep_best_importance": True},
    }


def _token_est(text: str) -> int:
    return max(1, int(round(len(str(text)) / 4.0)))


def _interval_union_seconds(rows: list[dict[str, Any]]) -> float:
    intervals: list[tuple[float, float]] = []
    for row in rows:
        try:
            t0 = float(row.get("t0", 0.0))
            t1 = float(row.get("t1", 0.0))
        except Exception:
            continue
        if t1 > t0:
            intervals.append((t0, t1))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float]] = []
    s0, s1 = intervals[0]
    for a0, a1 in intervals[1:]:
        if a0 <= s1:
            s1 = max(s1, a1)
            continue
        merged.append((s0, s1))
        s0, s1 = a0, a1
    merged.append((s0, s1))
    return float(sum(max(0.0, b - a) for a, b in merged))


def main() -> int:
    args = parse_args()
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    output = _model_validate_output(payload)

    repo_cfg = _load_repo_cfg(str(args.repo_cfg or ""))
    dedup_cfg = dict(repo_cfg.get("dedup", {}))
    read_cfg = dict(repo_cfg.get("read_policy", {"name": "budgeted_topk"}))
    if args.max_repo_chunks is not None:
        read_cfg["max_chunks"] = int(args.max_repo_chunks)
    if args.max_repo_tokens is not None:
        read_cfg["max_tokens"] = int(args.max_repo_tokens)
    if args.max_seconds is not None:
        read_cfg["max_seconds"] = float(args.max_seconds)

    raw_chunks = build_repo_chunks(output, cfg=repo_cfg)
    dedup_chunks = deduplicate_chunks(raw_chunks, cfg=dedup_cfg)
    queries = [str(q) for q in (args.query or []) if str(q).strip()]
    if not queries:
        queries = [""]

    selected_union: list[RepoChunk] = []
    selected_seen: set[str] = set()
    query_summaries: list[dict[str, Any]] = []
    for query in queries:
        selected = select_chunks_for_query(
            dedup_chunks,
            query=query,
            budget={
                "max_repo_chunks": int(read_cfg.get("max_chunks", 16)),
                "max_tokens": int(read_cfg.get("max_tokens", 200)),
                "max_seconds": read_cfg.get("max_seconds", None),
                "repo_read_policy": str(read_cfg.get("name", "budgeted_topk")),
            },
            cfg={"read_policy": read_cfg},
        )
        query_summaries.append({"query": query, "selected": len(selected)})
        for item in selected:
            if item.id in selected_seen:
                continue
            selected_seen.add(item.id)
            selected_union.append(item)

    dedup_rows = [_dump(c) for c in sorted(dedup_chunks, key=lambda x: (float(x.t0), float(x.t1), str(x.level), str(x.id)))]
    selected_rows = [_dump(c) for c in sorted(selected_union, key=lambda x: (float(x.t0), float(x.t1), str(x.level), str(x.id)))]
    context_text = "\n".join(f"- {row.get('text', '')}" for row in selected_rows)

    repo_jsonl = out_dir / "repo_chunks.jsonl"
    selected_jsonl = out_dir / "repo_selected.jsonl"
    context_txt = out_dir / "context.txt"
    report_md = out_dir / "report.md"
    snapshot_json = out_dir / "snapshot.json"
    _write_jsonl(repo_jsonl, dedup_rows)
    _write_jsonl(selected_jsonl, selected_rows)
    context_txt.write_text(context_text, encoding="utf-8")

    before_n = len(raw_chunks)
    after_n = len(dedup_chunks)
    dedup_rate = 0.0 if before_n <= 0 else float(1.0 - (after_n / max(1, before_n)))
    selected_tokens_est = int(sum(_token_est(str(r.get("text", ""))) for r in selected_rows))
    selected_seconds = _interval_union_seconds(selected_rows)
    level_counts: dict[str, int] = {}
    for row in dedup_rows:
        level = str(row.get("level", row.get("scale", "unknown")))
        level_counts[level] = level_counts.get(level, 0) + 1

    cfg_hash = _hash_cfg({"repo": repo_cfg, "read": read_cfg, "queries": queries})
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "json": str(json_path),
            "video_id": str(output.video_id),
            "duration_s": float(output.meta.get("duration_s", 0.0) or 0.0),
            "queries": queries,
        },
        "cfg": {
            "repo_cfg": repo_cfg,
            "read_cfg": read_cfg,
            "cfg_hash": cfg_hash,
            "repo_cfg_hash": policy_cfg_hash(repo_cfg),
        },
        "stats": {
            "chunks_before_dedup": before_n,
            "chunks_after_dedup": after_n,
            "dedup_rate": dedup_rate,
            "selected_chunks": len(selected_rows),
            "selected_tokens_est": selected_tokens_est,
            "selected_coverage_s": selected_seconds,
            "by_level_after_dedup": dict(sorted(level_counts.items())),
            "query_summaries": query_summaries,
        },
        "outputs": {
            "repo_chunks_jsonl": str(repo_jsonl),
            "repo_selected_jsonl": str(selected_jsonl),
            "context_txt": str(context_txt),
            "report_md": str(report_md),
            "snapshot_json": str(snapshot_json),
        },
    }
    snapshot_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RepoV1 Policy Smoke Report",
        "",
        f"- video_id: `{output.video_id}`",
        f"- cfg_hash: `{cfg_hash}`",
        f"- repo_cfg_hash: `{snapshot['cfg']['repo_cfg_hash']}`",
        f"- write_policy: `{repo_cfg.get('write_policy', {}).get('name', 'fixed_interval')}`",
        f"- read_policy: `{read_cfg.get('name', 'budgeted_topk')}`",
        f"- chunks_before_dedup: `{before_n}`",
        f"- chunks_after_dedup: `{after_n}`",
        f"- selected_chunks: `{len(selected_rows)}`",
        f"- selected_tokens_est: `{selected_tokens_est}`",
        f"- selected_coverage_s: `{selected_seconds:.2f}`",
        f"- by_level_after_dedup: `{json.dumps(level_counts, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Outputs",
        "",
        f"- `{repo_jsonl}`",
        f"- `{selected_jsonl}`",
        f"- `{context_txt}`",
        f"- `{snapshot_json}`",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"video_id={output.video_id}")
    print(f"chunks_before_dedup={before_n}")
    print(f"chunks_after_dedup={after_n}")
    print(f"selected_chunks={len(selected_rows)}")
    print(f"saved_repo_chunks={repo_jsonl}")
    print(f"saved_selected={selected_jsonl}")
    print(f"saved_context={context_txt}")
    print(f"saved_report={report_md}")
    print(f"saved_snapshot={snapshot_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
