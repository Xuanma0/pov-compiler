from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pov_compiler.io.video_reader import VideoReader
from pov_compiler.perception.backends import PerceptionBackend, create_backend
from pov_compiler.perception.contact import select_active_contact


def _histogram_signature(frame_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [int(bins)], [0, 256]).astype(np.float32).reshape(-1)
    s = float(np.sum(hist))
    if s > 0:
        hist = hist / s
    return hist


def _histogram_change(prev_hist: np.ndarray | None, curr_hist: np.ndarray) -> float:
    if prev_hist is None:
        return 0.0
    return float(0.5 * np.sum(np.abs(curr_hist - prev_hist)))


def _topk_labels(label_counts: dict[str, int], k: int) -> list[dict[str, Any]]:
    ranked = sorted(label_counts.items(), key=lambda x: (-int(x[1]), str(x[0])))
    return [{"label": str(label), "count": int(count)} for label, count in ranked[: max(1, int(k))]]


def _cache_key(
    *,
    video_path: Path,
    sample_fps: float,
    max_frames: int,
    backend_name: str,
    contact_min_score: float,
) -> str:
    stat = video_path.stat()
    payload = {
        "video": str(video_path.resolve()),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sample_fps": float(sample_fps),
        "max_frames": int(max_frames),
        "backend": str(backend_name),
        "contact_min_score": float(contact_min_score),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def run_perception(
    *,
    video_path: str | Path,
    sample_fps: float = 10.0,
    max_frames: int = 300,
    backend_name: str = "real",
    backend: PerceptionBackend | None = None,
    backend_kwargs: dict[str, Any] | None = None,
    fallback_to_stub: bool = True,
    strict: bool = False,
    cache_dir: str | Path | None = None,
    contact_min_score: float = 0.25,
    objects_topk: int = 10,
) -> dict[str, Any]:
    path = Path(video_path)
    reader = VideoReader(path)
    effective_fallback = bool(fallback_to_stub) and not bool(strict)
    requested_backend = str(backend_name).strip().lower()
    cache_hit = False
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        key = _cache_key(
            video_path=path,
            sample_fps=float(sample_fps),
            max_frames=int(max_frames),
            backend_name=requested_backend,
            contact_min_score=float(contact_min_score),
        )
        cache_file = cache_root / f"{path.stem}_{key}.perception.json"
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("frames", None), list):
                    cache_hit = True
                    summary = payload.get("summary", {})
                    if isinstance(summary, dict):
                        summary["cache_hit"] = True
                        payload["summary"] = summary
                    return payload
            except Exception:
                pass

    deps_ok = True
    fallback_used = False
    fallback_reason = ""
    if backend is None:
        try:
            backend = create_backend(requested_backend, **(backend_kwargs or {}))
        except Exception as exc:
            deps_ok = False
            if requested_backend == "real" and effective_fallback:
                backend = create_backend("stub")
                fallback_used = True
                fallback_reason = f"backend_init_failed:{type(exc).__name__}"
            else:
                raise RuntimeError(
                    f"Failed to initialize perception backend={requested_backend}. "
                    f"Set --perception-fallback-stub to allow fallback."
                ) from exc
    started = time.perf_counter()

    frames_out: list[dict[str, Any]] = []
    signal_time: list[float] = []
    signal_visual: list[float] = []
    signal_contact: list[float] = []
    label_counts: dict[str, int] = {}
    hand_frames = 0
    contact_events = 0
    frames_processed = 0
    frames_failed = 0

    prev_hist: np.ndarray | None = None
    for idx, (t, frame_bgr) in enumerate(reader.iter_samples(sample_fps=float(sample_fps))):
        if int(max_frames) > 0 and idx >= int(max_frames):
            break

        frames_processed += 1
        try:
            det = backend.detect(frame_bgr, frame_index=idx, t=float(t))
        except Exception as exc:
            frames_failed += 1
            if strict:
                raise RuntimeError(f"Perception frame inference failed at frame={idx}, t={float(t):.3f}") from exc
            if requested_backend == "real" and effective_fallback and str(getattr(backend, "name", "")) != "stub":
                try:
                    backend = create_backend("stub")
                    fallback_used = True
                    fallback_reason = f"frame_infer_failed:{type(exc).__name__}"
                    det = backend.detect(frame_bgr, frame_index=idx, t=float(t))
                except Exception:
                    det = {"objects": [], "hands": []}
            else:
                det = {"objects": [], "hands": []}
        objects = det.get("objects", [])
        hands = det.get("hands", [])
        if not isinstance(objects, list):
            objects = []
        if not isinstance(hands, list):
            hands = []

        for obj in objects:
            label = str(obj.get("label", "unknown"))
            label_counts[label] = label_counts.get(label, 0) + 1
        if hands:
            hand_frames += 1

        contact = select_active_contact(
            hands=[dict(x) for x in hands if isinstance(x, dict)],
            objects=[dict(x) for x in objects if isinstance(x, dict)],
            frame_shape=frame_bgr.shape,
            t=float(t),
            min_score=float(contact_min_score),
        )
        if contact.get("active") is not None:
            contact_events += 1

        hist = _histogram_signature(frame_bgr)
        visual_change = _histogram_change(prev_hist, hist)
        prev_hist = hist

        signal_time.append(float(t))
        signal_visual.append(float(visual_change))
        signal_contact.append(float(contact.get("active_score", 0.0)))
        frames_out.append(
            {
                "frame_index": int(idx),
                "t": float(t),
                "objects": objects,
                "hands": hands,
                "contact": contact,
            }
        )

    elapsed_s = float(max(1e-9, time.perf_counter() - started))
    processed = len(frames_out)
    out = {
        "video_id": path.stem,
        "meta": {
            "video_path": str(path),
            "source_fps": float(reader.fps),
            "duration_s": float(reader.duration_s),
            "sample_fps": float(sample_fps),
            "max_frames": int(max_frames),
            "backend": str(getattr(backend, "name", backend_name)),
            "processed_frames": int(processed),
            "elapsed_s": float(elapsed_s),
            "throughput_fps": float(processed / elapsed_s) if elapsed_s > 0 else 0.0,
        },
        "frames": frames_out,
        "signals": {
            "time": signal_time,
            "visual_change": signal_visual,
            "contact_score": signal_contact,
        },
        "summary": {
            "backend": str(getattr(backend, "name", requested_backend)),
            "requested_backend": requested_backend,
            "deps_ok": bool(deps_ok),
            "frames_processed": int(frames_processed),
            "frames_failed": int(frames_failed),
            "fallback_used": bool(fallback_used),
            "fallback_reason": str(fallback_reason),
            "cache_hit": bool(cache_hit),
            "objects_topk": _topk_labels(label_counts, k=int(objects_topk)),
            "hand_presence_rate": float(hand_frames / processed) if processed > 0 else 0.0,
            "contact_events_count": int(contact_events),
            "objects_total": int(sum(label_counts.values())),
            "frames_total": int(processed),
        },
    }
    if cache_file is not None:
        try:
            cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return out
