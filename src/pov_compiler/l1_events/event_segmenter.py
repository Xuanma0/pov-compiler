from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pov_compiler.schemas import Event


@dataclass
class SegmenterConfig:
    thresh: float = 0.65
    min_event_s: float = 3.0


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


def compute_motion_change(motion_energy: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(motion_energy, dtype=np.float32)
    if arr.size == 0:
        return arr
    diff = np.diff(arr, prepend=arr[0])
    return np.abs(diff)


def fuse_boundary_score(
    embed_dist: list[float] | np.ndarray,
    motion_energy: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embed_norm = normalize_signal(embed_dist)
    motion_change = compute_motion_change(motion_energy)
    motion_change_norm = normalize_signal(motion_change)
    boundary_score = 0.6 * embed_norm + 0.4 * motion_change_norm
    return motion_change, motion_change_norm, boundary_score


def _find_local_peaks(boundary_score: np.ndarray, thresh: float) -> list[int]:
    peaks: list[int] = []
    n = int(boundary_score.size)
    if n == 0:
        return peaks
    if n == 1:
        if float(boundary_score[0]) > thresh:
            peaks.append(0)
        return peaks

    if float(boundary_score[0]) > thresh and boundary_score[0] > boundary_score[1]:
        peaks.append(0)
    for i in range(1, n - 1):
        curr = float(boundary_score[i])
        if curr > thresh and curr >= float(boundary_score[i - 1]) and curr >= float(boundary_score[i + 1]):
            peaks.append(i)
    if float(boundary_score[n - 1]) > thresh and boundary_score[n - 1] > boundary_score[n - 2]:
        peaks.append(n - 1)
    return peaks


def _enforce_min_event(
    peak_indices: list[int],
    times: np.ndarray,
    duration_s: float,
    min_event_s: float,
) -> list[int]:
    if not peak_indices:
        return []

    kept: list[int] = []
    last_boundary_t = 0.0
    for idx in sorted(set(peak_indices)):
        t = float(times[idx])
        if t - last_boundary_t < min_event_s:
            continue
        if duration_s - t < min_event_s:
            continue
        kept.append(idx)
        last_boundary_t = t
    return kept


def segment_events(
    times: list[float] | np.ndarray,
    boundary_score: list[float] | np.ndarray,
    duration_s: float,
    thresh: float = 0.65,
    min_event_s: float = 3.0,
) -> list[Event]:
    times_arr = np.asarray(times, dtype=np.float32)
    score_arr = np.asarray(boundary_score, dtype=np.float32)

    if times_arr.size == 0:
        return []

    if duration_s <= 0:
        duration_s = float(times_arr[-1])
    duration_s = max(duration_s, float(times_arr[-1]))

    raw_peaks = _find_local_peaks(score_arr, thresh=thresh)
    peaks = _enforce_min_event(raw_peaks, times_arr, duration_s, min_event_s)

    boundary_times = [0.0] + [float(times_arr[idx]) for idx in peaks] + [float(duration_s)]
    events: list[Event] = []
    for i in range(len(boundary_times) - 1):
        t0 = float(boundary_times[i])
        t1 = float(boundary_times[i + 1])
        if t1 <= t0:
            continue
        mask = (times_arr >= t0) & (times_arr <= t1)
        conf = float(np.max(score_arr[mask])) if np.any(mask) else 0.0
        events.append(
            Event(
                id=f"event_{i + 1:04d}",
                t0=t0,
                t1=t1,
                scores={"boundary_conf": conf},
                anchors=[],
            )
        )

    if not events and duration_s > 0:
        events.append(
            Event(
                id="event_0001",
                t0=0.0,
                t1=float(duration_s),
                scores={"boundary_conf": 0.0},
                anchors=[],
            )
        )
    return events
