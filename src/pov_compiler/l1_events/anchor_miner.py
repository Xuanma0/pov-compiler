from __future__ import annotations

import numpy as np

from pov_compiler.l1_events.event_segmenter import normalize_signal
from pov_compiler.schemas import Anchor, Event


def _event_index_range(times: np.ndarray, event: Event) -> np.ndarray:
    return np.where((times >= event.t0) & (times <= event.t1))[0]


def _median_dt(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    return float(np.median(np.diff(times)))


def _find_local_peaks(values: np.ndarray, idxs: np.ndarray) -> list[int]:
    if idxs.size == 0:
        return []
    peaks: list[int] = []
    for k in range(1, idxs.size - 1):
        left = idxs[k - 1]
        i = idxs[k]
        right = idxs[k + 1]
        if values[i] >= values[left] and values[i] >= values[right]:
            peaks.append(int(i))
    if idxs.size == 1:
        peaks.append(int(idxs[0]))
    return peaks


def mine_stop_look(
    event: Event,
    times: np.ndarray,
    motion_energy: np.ndarray,
    quantile: float = 0.2,
    min_duration_s: float = 0.5,
) -> list[Anchor]:
    idxs = _event_index_range(times, event)
    if idxs.size == 0:
        return []

    event_motion = motion_energy[idxs]
    threshold = float(np.quantile(event_motion, quantile))
    mask = event_motion <= threshold
    if not np.any(mask):
        return []

    anchors: list[Anchor] = []
    dt = _median_dt(times[idxs]) if idxs.size > 1 else 0.0
    values_norm = normalize_signal(event_motion)

    run_start = None
    for pos, flag in enumerate(mask):
        if flag and run_start is None:
            run_start = pos
        if not flag and run_start is not None:
            run_end = pos - 1
            start_idx = idxs[run_start]
            end_idx = idxs[run_end]
            duration = float(times[end_idx] - times[start_idx] + dt)
            if duration >= min_duration_s:
                midpoint_t = float((times[start_idx] + times[end_idx]) * 0.5)
                mean_norm = float(np.mean(values_norm[run_start : run_end + 1]))
                anchors.append(
                    Anchor(
                        type="stop_look",
                        t=midpoint_t,
                        conf=float(max(0.0, min(1.0, 1.0 - mean_norm))),
                        meta={"duration_s": duration, "threshold": threshold},
                    )
                )
            run_start = None

    if run_start is not None:
        start_idx = idxs[run_start]
        end_idx = idxs[-1]
        duration = float(times[end_idx] - times[start_idx] + dt)
        if duration >= min_duration_s:
            midpoint_t = float((times[start_idx] + times[end_idx]) * 0.5)
            mean_norm = float(np.mean(values_norm[run_start:]))
            anchors.append(
                Anchor(
                    type="stop_look",
                    t=midpoint_t,
                    conf=float(max(0.0, min(1.0, 1.0 - mean_norm))),
                    meta={"duration_s": duration, "threshold": threshold},
                )
            )

    return anchors


def mine_turn_head(
    event: Event,
    times: np.ndarray,
    embed_dist: np.ndarray,
    motion_change: np.ndarray,
    top_k: int = 2,
) -> list[Anchor]:
    idxs = _event_index_range(times, event)
    if idxs.size == 0:
        return []

    embed_norm = normalize_signal(embed_dist)
    motion_norm = normalize_signal(motion_change)
    combined = 0.5 * embed_norm + 0.5 * motion_norm

    peaks = _find_local_peaks(combined, idxs)
    if not peaks:
        peaks = [int(i) for i in idxs]

    peaks = sorted(peaks, key=lambda i: float(combined[i]), reverse=True)[: max(1, top_k)]
    anchors: list[Anchor] = []
    for i in peaks:
        anchors.append(
            Anchor(
                type="turn_head",
                t=float(times[i]),
                conf=float(max(0.0, min(1.0, combined[i]))),
                meta={
                    "embed_dist": float(embed_dist[i]),
                    "motion_change": float(motion_change[i]),
                },
            )
        )
    return anchors


def mine_interaction_stub(event: Event) -> list[Anchor]:
    _ = event
    return []


def _merge_stop_look_cluster(cluster: list[Anchor]) -> Anchor:
    if not cluster:
        raise ValueError("cluster must not be empty")

    weights = [max(1e-6, float(a.conf)) for a in cluster]
    sum_weights = sum(weights)
    merged_t = float(sum(a.t * w for a, w in zip(cluster, weights)) / sum_weights)
    merged_conf = float(max(a.conf for a in cluster))
    duration_sum = float(sum(float(a.meta.get("duration_s", 0.0)) for a in cluster))
    threshold_max = float(max(float(a.meta.get("threshold", 0.0)) for a in cluster))
    return Anchor(
        type="stop_look",
        t=merged_t,
        conf=merged_conf,
        meta={
            "duration_s": duration_sum,
            "threshold": threshold_max,
            "merged_count": len(cluster),
        },
    )


def _suppress_stop_look(
    anchors: list[Anchor],
    max_stop_look_per_event: int,
    min_gap_s: float,
    stop_look_min_conf: float,
) -> list[Anchor]:
    if not anchors:
        return []
    ordered = sorted(anchors, key=lambda a: a.t)
    clusters: list[list[Anchor]] = []
    current: list[Anchor] = []

    for anchor in ordered:
        if not current:
            current = [anchor]
            continue
        if float(anchor.t - current[-1].t) < float(min_gap_s):
            current.append(anchor)
        else:
            clusters.append(current)
            current = [anchor]
    if current:
        clusters.append(current)

    merged = [_merge_stop_look_cluster(cluster) for cluster in clusters]
    merged = [a for a in merged if float(a.conf) >= float(stop_look_min_conf)]
    if max_stop_look_per_event >= 0:
        merged = sorted(
            merged,
            key=lambda a: (float(a.conf), float(a.meta.get("duration_s", 0.0))),
            reverse=True,
        )[:max_stop_look_per_event]
    merged.sort(key=lambda a: a.t)
    return merged


def suppress_event_anchors(
    anchors: list[Anchor],
    max_stop_look_per_event: int = 3,
    min_gap_s: float = 2.0,
    stop_look_min_conf: float = 0.6,
) -> list[Anchor]:
    by_type: dict[str, list[Anchor]] = {}
    for anchor in anchors:
        by_type.setdefault(anchor.type, []).append(anchor)

    output: list[Anchor] = []
    stop_look = by_type.pop("stop_look", [])
    output.extend(
        _suppress_stop_look(
            anchors=stop_look,
            max_stop_look_per_event=max_stop_look_per_event,
            min_gap_s=min_gap_s,
            stop_look_min_conf=stop_look_min_conf,
        )
    )
    for anchors_for_type in by_type.values():
        output.extend(anchors_for_type)

    output.sort(key=lambda a: (a.t, a.type))
    return output


def mine_event_anchors(
    event: Event,
    times: list[float] | np.ndarray,
    motion_energy: list[float] | np.ndarray,
    embed_dist: list[float] | np.ndarray,
    motion_change: list[float] | np.ndarray,
    stop_look_quantile: float = 0.2,
    stop_look_min_duration_s: float = 0.5,
    turn_head_top_k: int = 2,
    max_stop_look_per_event: int = 3,
    min_gap_s: float = 2.0,
    stop_look_min_conf: float = 0.6,
) -> list[Anchor]:
    times_arr = np.asarray(times, dtype=np.float32)
    motion_arr = np.asarray(motion_energy, dtype=np.float32)
    embed_arr = np.asarray(embed_dist, dtype=np.float32)
    motion_change_arr = np.asarray(motion_change, dtype=np.float32)

    anchors: list[Anchor] = []
    anchors.extend(
        mine_stop_look(
            event=event,
            times=times_arr,
            motion_energy=motion_arr,
            quantile=stop_look_quantile,
            min_duration_s=stop_look_min_duration_s,
        )
    )
    anchors.extend(
        mine_turn_head(
            event=event,
            times=times_arr,
            embed_dist=embed_arr,
            motion_change=motion_change_arr,
            top_k=turn_head_top_k,
        )
    )
    anchors.extend(mine_interaction_stub(event))
    return suppress_event_anchors(
        anchors=anchors,
        max_stop_look_per_event=max_stop_look_per_event,
        min_gap_s=min_gap_s,
        stop_look_min_conf=stop_look_min_conf,
    )
