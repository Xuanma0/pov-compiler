from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from pov_compiler.utils.subprocesses import run_command


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number:  # NaN
        return None
    if number < 0:
        return None
    return number


def _parse_frame_rate(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if "/" in text:
        lhs, rhs = text.split("/", 1)
        num = _safe_float(lhs)
        den = _safe_float(rhs)
        if num is None or den is None or den <= 0:
            return None
        return float(num / den)
    return _safe_float(text)


def probe_video_ffprobe(
    path: Path,
    timeout_s: float = 20.0,
    run_fn: Callable[..., Any] = run_command,
) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = run_fn(cmd, timeout_s=timeout_s, check=False)
    if int(getattr(result, "returncode", 1)) != 0:
        return {"ok": False, "error": str(getattr(result, "stderr", "")).strip() or "ffprobe_failed"}
    try:
        payload = json.loads(str(getattr(result, "stdout", "")))
    except Exception:
        return {"ok": False, "error": "ffprobe_invalid_json"}

    stream = {}
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    if isinstance(streams, list) and streams:
        first = streams[0]
        if isinstance(first, dict):
            stream = first
    fmt = payload.get("format", {}) if isinstance(payload, dict) else {}
    if not isinstance(fmt, dict):
        fmt = {}

    duration_s = _safe_float(fmt.get("duration"))
    fps = _parse_frame_rate(stream.get("r_frame_rate"))
    width = int(stream.get("width")) if str(stream.get("width", "")).isdigit() else None
    height = int(stream.get("height")) if str(stream.get("height", "")).isdigit() else None
    return {
        "ok": True,
        "duration_s": duration_s,
        "fps": fps,
        "width": width,
        "height": height,
        "probe_backend": "ffprobe",
    }


def probe_video_cv2(path: Path, cv2_module: Any | None = None) -> dict[str, Any]:
    cv2 = cv2_module
    if cv2 is None:
        try:
            import cv2 as cv2_imported  # type: ignore
        except Exception:
            return {"ok": False, "error": "cv2_unavailable"}
        cv2 = cv2_imported

    cap = cv2.VideoCapture(str(path))
    if not cap or not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return {"ok": False, "error": "cv2_open_failed"}
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    duration_s: float | None = None
    fps_out: float | None = None
    if fps and fps > 0 and frame_count >= 0:
        duration_s = float(frame_count / fps)
        fps_out = float(fps)
    width_out = int(width) if width > 0 else None
    height_out = int(height) if height > 0 else None
    return {
        "ok": True,
        "duration_s": duration_s,
        "fps": fps_out,
        "width": width_out,
        "height": height_out,
        "probe_backend": "cv2",
    }


def probe_video_metadata(
    path: Path,
    *,
    prefer_ffprobe: bool = True,
    timeout_s: float = 20.0,
    which_fn: Callable[[str], str | None] = shutil.which,
    run_fn: Callable[..., Any] = run_command,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    path = Path(path)
    base = {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "duration_s": None,
        "fps": None,
        "width": None,
        "height": None,
        "probe_backend": None,
        "probed": True,
        "ok": False,
        "error": None,
    }

    ffprobe_available = bool(which_fn("ffprobe"))
    if prefer_ffprobe and ffprobe_available:
        ffprobe_result = probe_video_ffprobe(path=path, timeout_s=timeout_s, run_fn=run_fn)
        if ffprobe_result.get("ok"):
            base.update(ffprobe_result)
            base["ok"] = True
            return base
        base["error"] = ffprobe_result.get("error")

    cv2_result = probe_video_cv2(path=path, cv2_module=cv2_module)
    if cv2_result.get("ok"):
        base.update(cv2_result)
        base["ok"] = True
        return base

    if base.get("error") is None:
        base["error"] = cv2_result.get("error")
    return base


def get_duration_bucket(duration_s: float | int | None, bins: list[float] | tuple[float, ...] = (120.0, 600.0, 1800.0)) -> str:
    """Map a duration (seconds) to a coarse bucket.

    Default buckets:
    - short: <= 120s
    - medium: <= 600s
    - long: <= 1800s
    - very_long: > 1800s
    """

    value = _safe_float(duration_s)
    if value is None:
        return "unknown"
    try:
        edges = sorted(float(x) for x in bins)
    except Exception:
        edges = [120.0, 600.0, 1800.0]
    if len(edges) < 3:
        edges = [120.0, 600.0, 1800.0]
    if value <= edges[0]:
        return "short"
    if value <= edges[1]:
        return "medium"
    if value <= edges[2]:
        return "long"
    return "very_long"
