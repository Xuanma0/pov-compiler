from __future__ import annotations

import hashlib
import re
from typing import Any

from pov_compiler.repository.schema import RepoChunk


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def _normalize_text(text: str) -> str:
    base = str(text).lower()
    base = _PUNCT_RE.sub(" ", base)
    base = _WS_RE.sub(" ", base).strip()
    return base


def _hash_text(text: str) -> str:
    norm = _normalize_text(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(float(a0), float(b0))
    right = min(float(a1), float(b1))
    inter = max(0.0, right - left)
    union = max(1e-9, (float(a1) - float(a0)) + (float(b1) - float(b0)) - inter)
    return float(inter / union)


def deduplicate_chunks(chunks: list[RepoChunk], cfg: dict[str, Any] | None = None) -> list[RepoChunk]:
    cfg = dict(cfg or {})
    iou_thresh = float(cfg.get("iou_thresh", 0.6))
    keep_best = bool(cfg.get("keep_best_importance", True))

    ordered = sorted(chunks, key=lambda c: (float(c.t0), float(c.t1), str(c.scale), str(c.id)))
    kept: list[RepoChunk] = []
    groups: dict[str, list[int]] = {}

    for chunk in ordered:
        text_hash = _hash_text(chunk.text)
        bucket = groups.get(text_hash, [])
        duplicate_idx: int | None = None
        for idx in bucket:
            ref = kept[idx]
            if _iou(chunk.t0, chunk.t1, ref.t0, ref.t1) >= iou_thresh:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            chunk.meta = dict(chunk.meta or {})
            chunk.meta["norm_hash"] = text_hash
            groups.setdefault(text_hash, []).append(len(kept))
            kept.append(chunk)
            continue

        if not keep_best:
            continue

        prev = kept[duplicate_idx]
        cand = chunk
        prev_key = (float(prev.importance), float(prev.t1 - prev.t0), -len(prev.text), str(prev.id))
        cand_key = (float(cand.importance), float(cand.t1 - cand.t0), -len(cand.text), str(cand.id))
        if cand_key > prev_key:
            cand.meta = dict(cand.meta or {})
            cand.meta["norm_hash"] = text_hash
            kept[duplicate_idx] = cand

    kept.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.scale), str(c.id)))
    return kept
