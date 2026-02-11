from __future__ import annotations

from dataclasses import dataclass, field

from pov_compiler.schemas import Event, KeyClip


@dataclass
class _Window:
    t0: float
    t1: float
    source_event: str
    anchor_type: str
    anchor_t: float
    conf: float
    priority: int
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, float(self.t1 - self.t0))


def _anchor_priority(anchor_type: str, priority_map: dict[str, int]) -> int:
    if anchor_type in priority_map:
        return int(priority_map[anchor_type])
    if anchor_type.startswith("interaction"):
        return int(priority_map.get("interaction", 0))
    return int(priority_map.get(anchor_type, 0))


def _merge_windows(windows: list[_Window], merge_gap_s: float) -> list[list[_Window]]:
    if not windows:
        return []
    sorted_windows = sorted(windows, key=lambda w: (w.t0, w.t1))
    groups: list[list[_Window]] = []
    for win in sorted_windows:
        if not groups:
            groups.append([win])
            continue
        prev_group = groups[-1]
        prev_end = max(w.t1 for w in prev_group)
        if win.t0 <= prev_end + merge_gap_s:
            prev_group.append(win)
        else:
            groups.append([win])
    return groups


def _group_to_keyclip(group: list[_Window], idx: int, priority_map: dict[str, int]) -> KeyClip:
    t0 = float(min(w.t0 for w in group))
    t1 = float(max(w.t1 for w in group))

    anchor_types = sorted(set(w.anchor_type for w in group))
    source_events = sorted(set(w.source_event for w in group))

    best = sorted(group, key=lambda w: (w.priority, w.conf, w.duration), reverse=True)[0]
    if len(anchor_types) == 1:
        anchor_type = anchor_types[0]
    else:
        anchor_type = best.anchor_type

    source_event = source_events[0] if len(source_events) == 1 else "mixed"

    weights = [max(1e-6, float(w.conf)) for w in group]
    weight_sum = sum(weights)
    anchor_t = float(sum(w.anchor_t * ww for w, ww in zip(group, weights)) / weight_sum)
    conf = float(max(w.conf for w in group))

    return KeyClip(
        id=f"hl_{idx:04d}",
        t0=t0,
        t1=t1,
        source_event=source_event,
        anchor_type=anchor_type,
        anchor_t=anchor_t,
        conf=conf,
        meta={
            "anchor_types": anchor_types,
            "source_events": source_events,
            "merged_count": len(group),
            "priority_max": max(_anchor_priority(t, priority_map) for t in anchor_types),
        },
    )


def _windows_union_duration(windows: list[_Window], merge_gap_s: float) -> float:
    groups = _merge_windows(windows, merge_gap_s=merge_gap_s)
    duration = 0.0
    for group in groups:
        duration += max(0.0, max(w.t1 for w in group) - min(w.t0 for w in group))
    return float(duration)


def _to_keyclips(windows: list[_Window], merge_gap_s: float, priority_map: dict[str, int]) -> list[KeyClip]:
    groups = _merge_windows(windows, merge_gap_s=merge_gap_s)
    keyclips = [_group_to_keyclip(group, idx + 1, priority_map) for idx, group in enumerate(groups)]
    return keyclips


def _stats(original_duration_s: float, keyclips: list[KeyClip]) -> dict[str, float | int]:
    kept = float(sum(max(0.0, clip.t1 - clip.t0) for clip in keyclips))
    if kept > 1e-9:
        compression = float(original_duration_s / kept)
    else:
        compression = 0.0
    return {
        "original_duration_s": float(original_duration_s),
        "kept_duration_s": kept,
        "compression_ratio": compression,
        "num_highlights": int(len(keyclips)),
    }


def build_highlights(
    events: list[Event],
    duration_s: float,
    pre_s: float,
    post_s: float,
    max_total_s: float,
    priority_map: dict[str, int] | None = None,
    merge_gap_s: float = 0.2,
) -> tuple[list[KeyClip], dict[str, float | int]]:
    priority_map = priority_map or {"interaction": 3, "turn_head": 2, "stop_look": 1}
    original_duration_s = float(max(0.0, duration_s))
    if original_duration_s <= 0:
        return [], _stats(original_duration_s, [])
    if max_total_s <= 0:
        return [], _stats(original_duration_s, [])

    windows: list[_Window] = []
    for event in events:
        for anchor in event.anchors:
            priority = _anchor_priority(anchor.type, priority_map)
            if priority <= 0:
                continue
            t0 = max(0.0, float(anchor.t) - float(pre_s))
            t1 = min(original_duration_s, float(anchor.t) + float(post_s))
            if t1 - t0 < 0.2:
                continue
            windows.append(
                _Window(
                    t0=t0,
                    t1=t1,
                    source_event=event.id,
                    anchor_type=anchor.type,
                    anchor_t=float(anchor.t),
                    conf=float(anchor.conf),
                    priority=priority,
                    meta=dict(anchor.meta),
                )
            )

    if not windows:
        return [], _stats(original_duration_s, [])

    merged_all = _to_keyclips(windows, merge_gap_s=merge_gap_s, priority_map=priority_map)
    merged_duration = float(sum(max(0.0, clip.t1 - clip.t0) for clip in merged_all))
    if merged_duration <= max_total_s + 1e-9:
        return merged_all, _stats(original_duration_s, merged_all)

    ranked = sorted(windows, key=lambda w: (w.priority, w.conf, w.duration), reverse=True)
    selected: list[_Window] = []
    current = 0.0

    for candidate in ranked:
        next_total = _windows_union_duration(selected + [candidate], merge_gap_s=merge_gap_s)
        add = next_total - current
        if add <= 1e-9:
            continue
        if next_total > max_total_s + 1e-9:
            continue
        selected.append(candidate)
        current = next_total
        if current >= max_total_s - 1e-9:
            break

    keyclips = _to_keyclips(selected, merge_gap_s=merge_gap_s, priority_map=priority_map)
    return keyclips, _stats(original_duration_s, keyclips)
