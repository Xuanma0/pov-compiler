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


def _token_set(text: str) -> set[str]:
    return {x for x in _normalize_text(text).split(" ") if x}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return float(len(a & b) / max(1, len(a | b)))


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(float(a0), float(b0))
    right = min(float(a1), float(b1))
    inter = max(0.0, right - left)
    union = max(1e-9, (float(a1) - float(a0)) + (float(b1) - float(b0)) - inter)
    return float(inter / union)


def _level_weight(level: str) -> float:
    lv = str(level).lower()
    if lv == "decision":
        return 0.35
    if lv == "place":
        return 0.3
    if lv == "event":
        return 0.2
    if lv == "segment":
        return 0.15
    if lv == "window":
        return 0.1
    return 0.05


def _info_density(chunk: RepoChunk) -> float:
    return float(
        float(chunk.importance)
        + _level_weight(chunk.level or chunk.scale)
        + 0.02 * len(chunk.tags or [])
        + 0.01 * len(chunk.source_ids or [])
    )


def deduplicate_chunks(chunks: list[RepoChunk], cfg: dict[str, Any] | None = None) -> list[RepoChunk]:
    cfg = dict(cfg or {})
    iou_thresh = float(cfg.get("iou_thresh", 0.6))
    sim_thresh = float(cfg.get("sim_thresh", cfg.get("text_sim_thresh", 0.9)))
    keep_best = bool(cfg.get("keep_best_importance", True))
    cross_scale = bool(cfg.get("cross_scale", True))

    ordered = sorted(chunks, key=lambda c: (float(c.t0), float(c.t1), str(c.level or c.scale), str(c.id)))
    kept: list[RepoChunk] = []

    for chunk in ordered:
        chunk.meta = dict(chunk.meta or {})
        chunk.meta["norm_hash"] = _hash_text(chunk.text)
        words = _token_set(chunk.text)

        duplicate_idx: int | None = None
        duplicate_reason = ""
        for idx, ref in enumerate(kept):
            if not cross_scale and str(ref.level or ref.scale) != str(chunk.level or chunk.scale):
                continue
            iou = _iou(chunk.t0, chunk.t1, ref.t0, ref.t1)
            if iou < iou_thresh:
                continue
            sim = _jaccard(words, _token_set(ref.text))
            same_hash = bool(chunk.meta.get("norm_hash") == ref.meta.get("norm_hash"))
            if sim < sim_thresh and not same_hash:
                continue
            duplicate_idx = idx
            duplicate_reason = (
                f"sim={sim:.2f} iou={iou:.2f} keep={str(ref.level or ref.scale)} "
                f"drop={str(chunk.level or chunk.scale)}"
            )
            break

        if duplicate_idx is None:
            chunk.meta.setdefault("dedup_dropped_ids", [])
            chunk.meta.setdefault("dedup_kept_reasons", [])
            kept.append(chunk)
            continue

        if not keep_best:
            ref = kept[duplicate_idx]
            ref.meta = dict(ref.meta or {})
            dropped = list(ref.meta.get("dedup_dropped_ids", []))
            dropped.append(str(chunk.id))
            reasons = list(ref.meta.get("dedup_kept_reasons", []))
            reasons.append(str(duplicate_reason))
            ref.meta["dedup_dropped_ids"] = dropped
            ref.meta["dedup_kept_reasons"] = reasons
            continue

        ref = kept[duplicate_idx]
        ref_key = (_info_density(ref), float(ref.t1 - ref.t0), -len(ref.text), str(ref.id))
        chunk_key = (_info_density(chunk), float(chunk.t1 - chunk.t0), -len(chunk.text), str(chunk.id))
        if chunk_key > ref_key:
            chunk.meta = dict(chunk.meta or {})
            dropped = list(chunk.meta.get("dedup_dropped_ids", []))
            dropped.append(str(ref.id))
            reasons = list(chunk.meta.get("dedup_kept_reasons", []))
            reasons.append(
                f"sim_replaced keep={str(chunk.level or chunk.scale)} drop={str(ref.level or ref.scale)}; {duplicate_reason}"
            )
            chunk.meta["dedup_dropped_ids"] = dropped
            chunk.meta["dedup_kept_reasons"] = reasons
            kept[duplicate_idx] = chunk
        else:
            ref.meta = dict(ref.meta or {})
            dropped = list(ref.meta.get("dedup_dropped_ids", []))
            dropped.append(str(chunk.id))
            reasons = list(ref.meta.get("dedup_kept_reasons", []))
            reasons.append(str(duplicate_reason))
            ref.meta["dedup_dropped_ids"] = dropped
            ref.meta["dedup_kept_reasons"] = reasons

    kept.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.level or c.scale), str(c.id)))
    return kept

