from __future__ import annotations

from typing import Any

import numpy as np

from pov_compiler.schemas import Event


def normalize_signal(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / span


def fuse_boundary_signal_v0(
    visual_change: list[float] | np.ndarray,
    contact_score: list[float] | np.ndarray,
    *,
    visual_weight: float = 0.7,
    contact_weight: float = 0.3,
) -> np.ndarray:
    v = normalize_signal(visual_change)
    c = normalize_signal(contact_score)
    if v.size == 0 and c.size == 0:
        return np.asarray([], dtype=np.float32)
    if v.size == 0:
        v = np.zeros_like(c)
    if c.size == 0:
        c = np.zeros_like(v)
    n = min(v.size, c.size)
    if n <= 0:
        return np.asarray([], dtype=np.float32)
    vv = v[:n]
    cc = c[:n]
    score = float(visual_weight) * vv + float(contact_weight) * cc
    return np.asarray(score, dtype=np.float32)


def _find_peaks(signal: np.ndarray, thresh: float) -> list[int]:
    n = int(signal.size)
    if n == 0:
        return []
    if n == 1:
        return [0] if float(signal[0]) >= float(thresh) else []
    peaks: list[int] = []
    if float(signal[0]) >= float(thresh) and signal[0] > signal[1]:
        peaks.append(0)
    for i in range(1, n - 1):
        curr = float(signal[i])
        if curr >= float(thresh) and curr >= float(signal[i - 1]) and curr >= float(signal[i + 1]):
            peaks.append(i)
    if float(signal[n - 1]) >= float(thresh) and signal[n - 1] > signal[n - 2]:
        peaks.append(n - 1)
    return peaks


def _enforce_min_gap(peaks: list[int], times: np.ndarray, duration_s: float, min_event_s: float) -> list[int]:
    kept: list[int] = []
    last_t = 0.0
    for idx in sorted(set(peaks)):
        t = float(times[idx])
        if t - last_t < float(min_event_s):
            continue
        if float(duration_s) - t < float(min_event_s):
            continue
        kept.append(idx)
        last_t = t
    return kept


def _segment_label(*, visual_mean: float, contact_mean: float) -> str:
    if contact_mean >= 0.35:
        return "interaction-heavy"
    if visual_mean >= 0.45:
        return "navigation"
    return "idle"


def segment_events_v0(
    *,
    times: list[float] | np.ndarray,
    visual_change: list[float] | np.ndarray,
    contact_score: list[float] | np.ndarray,
    duration_s: float,
    thresh: float = 0.45,
    min_event_s: float = 3.0,
    visual_weight: float = 0.7,
    contact_weight: float = 0.3,
) -> tuple[list[Event], np.ndarray]:
    times_arr = np.asarray(times, dtype=np.float32)
    if times_arr.size == 0:
        return [], np.asarray([], dtype=np.float32)
    if float(duration_s) <= 0:
        duration_s = float(times_arr[-1])
    duration_s = max(float(duration_s), float(times_arr[-1]))

    score = fuse_boundary_signal_v0(
        visual_change=visual_change,
        contact_score=contact_score,
        visual_weight=float(visual_weight),
        contact_weight=float(contact_weight),
    )
    n = min(times_arr.size, score.size)
    if n <= 0:
        return [], score
    times_arr = times_arr[:n]
    score = score[:n]
    v_norm = normalize_signal(visual_change)[:n]
    c_norm = normalize_signal(contact_score)[:n]

    peaks = _find_peaks(score, thresh=float(thresh))
    peaks = _enforce_min_gap(peaks, times_arr, duration_s=float(duration_s), min_event_s=float(min_event_s))
    boundaries = [0.0] + [float(times_arr[i]) for i in peaks] + [float(duration_s)]

    events: list[Event] = []
    for i in range(len(boundaries) - 1):
        t0 = float(boundaries[i])
        t1 = float(boundaries[i + 1])
        if t1 <= t0:
            continue
        mask = (times_arr >= t0) & (times_arr <= t1)
        if not np.any(mask):
            v_mean = 0.0
            c_mean = 0.0
            b_conf = 0.0
        else:
            v_mean = float(np.mean(v_norm[mask]))
            c_mean = float(np.mean(c_norm[mask]))
            b_conf = float(np.max(score[mask]))
        label = _segment_label(visual_mean=v_mean, contact_mean=c_mean)
        events.append(
            Event(
                id=f"v0_event_{i + 1:04d}",
                t0=t0,
                t1=t1,
                scores={
                    "boundary_conf": float(b_conf),
                    "visual_mean": float(v_mean),
                    "contact_mean": float(c_mean),
                },
                anchors=[],
                meta={"label": label, "layer": "event_v0"},
            )
        )
    if not events:
        events = [
            Event(
                id="v0_event_0001",
                t0=0.0,
                t1=float(duration_s),
                scores={"boundary_conf": 0.0, "visual_mean": 0.0, "contact_mean": 0.0},
                anchors=[],
                meta={"label": "idle", "layer": "event_v0"},
            )
        ]
    return events, score


def events_v0_to_dict(events: list[Event]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        out.append(
            {
                "id": str(event.id),
                "t0": float(event.t0),
                "t1": float(event.t1),
                "label": str(event.meta.get("label", "")),
                "scores": dict(event.scores),
            }
        )
    return out

