from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pov_compiler.schemas import Event, Output, Token, TokenCodec


VOCAB_V02 = [
    "EVENT_START",
    "EVENT_END",
    "MOTION_MOVING",
    "MOTION_STILL",
    "ATTENTION_TURN_HEAD",
    "ATTENTION_STOP_LOOK",
    "HIGHLIGHT",
    "SCENE_CHANGE",
    "INTERACTION",
]


@dataclass
class TokenCompilerConfig:
    enabled: bool = True
    attention_pre_s: float = 0.5
    attention_post_s: float = 0.5
    motion_min_run_s: float = 0.8
    motion_max_runs_per_event: int = 4
    motion_quantile: float = 0.6
    scene_change_top_k: int = 3
    merge_gap_s: float = 0.2


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _event_indices(times: np.ndarray, event: Event) -> np.ndarray:
    return np.where((times >= event.t0) & (times <= event.t1))[0]


def _run_duration(times: np.ndarray, start_idx: int, end_idx: int) -> float:
    if start_idx > end_idx:
        return 0.0
    if start_idx == end_idx:
        return 0.0
    dt = float(np.median(np.diff(times[start_idx : end_idx + 1])))
    return float(times[end_idx] - times[start_idx] + dt)


def _collapse_same_state_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not runs:
        return []
    merged = [dict(runs[0])]
    for run in runs[1:]:
        prev = merged[-1]
        if bool(prev["moving"]) == bool(run["moving"]):
            prev["end"] = int(run["end"])
        else:
            merged.append(dict(run))
    return merged


def _merge_short_runs(
    runs: list[dict[str, Any]],
    times: np.ndarray,
    min_run_s: float,
) -> list[dict[str, Any]]:
    runs = _collapse_same_state_runs(runs)
    if len(runs) <= 1:
        return runs

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(list(runs)):
            dur = _run_duration(times, int(run["start"]), int(run["end"]))
            if dur >= min_run_s:
                continue
            if i == 0:
                runs[i + 1]["start"] = int(run["start"])
                del runs[i]
            elif i == len(runs) - 1:
                runs[i - 1]["end"] = int(run["end"])
                del runs[i]
            else:
                left = runs[i - 1]
                right = runs[i + 1]
                left_dur = _run_duration(times, int(left["start"]), int(left["end"]))
                right_dur = _run_duration(times, int(right["start"]), int(right["end"]))
                if left_dur >= right_dur:
                    left["end"] = int(run["end"])
                else:
                    right["start"] = int(run["start"])
                del runs[i]
            runs = _collapse_same_state_runs(runs)
            changed = True
            break
    return runs


def _time_to_event(t: float, events: list[Event]) -> str:
    for event in events:
        if event.t0 - 1e-6 <= t <= event.t1 + 1e-6:
            return event.id
    if not events:
        return "global"
    return events[-1].id


def _find_top_peaks(signal: np.ndarray, top_k: int) -> list[int]:
    n = int(signal.size)
    if n == 0 or top_k <= 0:
        return []
    peaks: list[int] = []
    if n == 1:
        return [0]
    if signal[0] >= signal[1]:
        peaks.append(0)
    for i in range(1, n - 1):
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            peaks.append(i)
    if signal[n - 1] >= signal[n - 2]:
        peaks.append(n - 1)
    peaks = sorted(peaks, key=lambda idx: float(signal[idx]), reverse=True)[:top_k]
    peaks.sort()
    return peaks


def _token_priority(token: Token) -> int:
    if token.type == "HIGHLIGHT":
        return 90
    if token.type.startswith("ATTENTION_"):
        return 80
    if token.type in ("EVENT_START", "EVENT_END"):
        return 70
    if token.type.startswith("MOTION_"):
        return 40
    if token.type == "SCENE_CHANGE":
        return 20
    return 10


class TokenCodecCompiler:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.cfg = TokenCompilerConfig(
            enabled=bool(cfg.get("enabled", True)),
            attention_pre_s=float(cfg.get("attention_pre_s", 0.5)),
            attention_post_s=float(cfg.get("attention_post_s", 0.5)),
            motion_min_run_s=float(cfg.get("motion_min_run_s", 0.8)),
            motion_max_runs_per_event=int(cfg.get("motion_max_runs_per_event", 4)),
            motion_quantile=float(cfg.get("motion_quantile", 0.6)),
            scene_change_top_k=int(cfg.get("scene_change_top_k", 3)),
            merge_gap_s=float(cfg.get("merge_gap_s", 0.2)),
        )

    def compile(self, output: Output) -> TokenCodec:
        if not self.cfg.enabled:
            return TokenCodec(version="0.2", vocab=VOCAB_V02, tokens=[])

        duration_s = float(output.meta.get("duration_s", 0.0))
        events = sorted(output.events, key=lambda e: (e.t0, e.t1))

        signals = output.debug.get("signals", {}) if isinstance(output.debug, dict) else {}
        times = np.asarray(signals.get("time", []), dtype=np.float32)
        motion = np.asarray(signals.get("motion_energy", []), dtype=np.float32)
        embed = np.asarray(signals.get("embed_dist", []), dtype=np.float32)
        boundary = np.asarray(signals.get("boundary_score", []), dtype=np.float32)

        raw_tokens: list[Token] = []
        raw_tokens.extend(self._compile_event_tokens(events))
        raw_tokens.extend(self._compile_motion_tokens(events, times, motion, duration_s))
        raw_tokens.extend(self._compile_attention_tokens(events, duration_s))
        raw_tokens.extend(self._compile_highlight_tokens(output))
        raw_tokens.extend(self._compile_scene_change_tokens(events, times, boundary, embed, duration_s))

        merged = self._merge_same_type(raw_tokens, self.cfg.merge_gap_s)
        merged.sort(key=lambda t: (t.t0, t.t1, -_token_priority(t), -t.conf, t.type))
        for i, token in enumerate(merged, start=1):
            token.id = f"tok_{i:06d}"

        return TokenCodec(version="0.2", vocab=VOCAB_V02, tokens=merged)

    def _compile_event_tokens(self, events: list[Event]) -> list[Token]:
        tokens: list[Token] = []
        for event in events:
            conf = float(event.scores.get("boundary_conf", 1.0))
            tokens.append(
                Token(
                    id="",
                    t0=float(event.t0),
                    t1=float(event.t0),
                    type="EVENT_START",
                    conf=_clamp(conf, 0.0, 1.0),
                    source_event=event.id,
                    source={"event_id": event.id},
                    meta={},
                )
            )
            tokens.append(
                Token(
                    id="",
                    t0=float(event.t1),
                    t1=float(event.t1),
                    type="EVENT_END",
                    conf=_clamp(conf, 0.0, 1.0),
                    source_event=event.id,
                    source={"event_id": event.id},
                    meta={},
                )
            )
        return tokens

    def _compile_motion_tokens(
        self,
        events: list[Event],
        times: np.ndarray,
        motion: np.ndarray,
        duration_s: float,
    ) -> list[Token]:
        tokens: list[Token] = []
        if times.size == 0 or motion.size == 0 or times.size != motion.size:
            return tokens

        for event in events:
            idxs = _event_indices(times, event)
            if idxs.size == 0:
                continue
            event_motion = motion[idxs]
            event_times = times[idxs]
            norm = _normalize(event_motion)
            threshold_raw = float(np.quantile(event_motion, self.cfg.motion_quantile))
            threshold_norm = float(np.quantile(norm, self.cfg.motion_quantile))

            labels = event_motion >= threshold_raw
            runs: list[dict[str, Any]] = []
            run_start = 0
            for i in range(1, idxs.size):
                if bool(labels[i]) != bool(labels[i - 1]):
                    runs.append({"start": run_start, "end": i - 1, "moving": bool(labels[i - 1])})
                    run_start = i
            runs.append({"start": run_start, "end": idxs.size - 1, "moving": bool(labels[-1])})

            runs = _merge_short_runs(runs, event_times, self.cfg.motion_min_run_s)

            run_entries: list[dict[str, Any]] = []
            for run in runs:
                s = int(run["start"])
                e = int(run["end"])
                mean_norm = float(np.mean(norm[s : e + 1]))
                conf = _clamp(abs(mean_norm - threshold_norm), 0.0, 1.0)
                run_t0 = float(event_times[s])
                run_t1 = float(event_times[e])
                if e > s:
                    run_t1 = min(float(event.t1), run_t1 + 0.001)
                run_entries.append(
                    {
                        "moving": bool(run["moving"]),
                        "t0": run_t0,
                        "t1": run_t1,
                        "conf": conf,
                        "score": conf + max(0.0, run_t1 - run_t0),
                        "mean_norm": mean_norm,
                        "threshold_raw": threshold_raw,
                    }
                )

            if self.cfg.motion_max_runs_per_event > 0 and len(run_entries) > self.cfg.motion_max_runs_per_event:
                run_entries = sorted(run_entries, key=lambda r: float(r["score"]), reverse=True)[
                    : self.cfg.motion_max_runs_per_event
                ]
            run_entries.sort(key=lambda r: float(r["t0"]))

            for run in run_entries:
                token_type = "MOTION_MOVING" if run["moving"] else "MOTION_STILL"
                tokens.append(
                    Token(
                        id="",
                        t0=_clamp(float(run["t0"]), 0.0, duration_s),
                        t1=_clamp(float(run["t1"]), 0.0, duration_s),
                        type=token_type,
                        conf=_clamp(float(run["conf"]), 0.0, 1.0),
                        source_event=event.id,
                        source={"signal": "motion_energy"},
                        meta={
                            "mean_motion_norm": float(run["mean_norm"]),
                            "threshold": float(run["threshold_raw"]),
                        },
                    )
                )
        return tokens

    def _compile_attention_tokens(self, events: list[Event], duration_s: float) -> list[Token]:
        tokens: list[Token] = []
        for event in events:
            for idx, anchor in enumerate(event.anchors):
                if anchor.type == "turn_head":
                    t0 = max(0.0, float(anchor.t) - self.cfg.attention_pre_s)
                    t1 = min(duration_s, float(anchor.t) + self.cfg.attention_post_s)
                    token_type = "ATTENTION_TURN_HEAD"
                elif anchor.type == "stop_look":
                    duration = float(anchor.meta.get("duration_s", 0.8))
                    if duration <= 0:
                        duration = 0.8
                    t0 = max(0.0, float(anchor.t) - duration * 0.5)
                    t1 = min(duration_s, float(anchor.t) + duration * 0.5)
                    token_type = "ATTENTION_STOP_LOOK"
                elif anchor.type.startswith("interaction"):
                    t0 = max(0.0, float(anchor.t) - self.cfg.attention_pre_s)
                    t1 = min(duration_s, float(anchor.t) + self.cfg.attention_post_s)
                    token_type = "INTERACTION"
                else:
                    continue

                if t1 < t0:
                    t1 = t0
                tokens.append(
                    Token(
                        id="",
                        t0=float(t0),
                        t1=float(t1),
                        type=token_type,
                        conf=_clamp(float(anchor.conf), 0.0, 1.0),
                        source_event=event.id,
                        source={
                            "anchor_index": idx,
                            "anchor_type": anchor.type,
                            "anchor_t": float(anchor.t),
                        },
                        meta=dict(anchor.meta),
                    )
                )
        return tokens

    def _compile_highlight_tokens(self, output: Output) -> list[Token]:
        tokens: list[Token] = []
        duration_s = float(output.meta.get("duration_s", 0.0))
        for hl in output.highlights:
            tokens.append(
                Token(
                    id="",
                    t0=_clamp(float(hl.t0), 0.0, duration_s),
                    t1=_clamp(float(hl.t1), 0.0, duration_s),
                    type="HIGHLIGHT",
                    conf=_clamp(float(hl.conf), 0.0, 1.0),
                    source_event=hl.source_event,
                    source={
                        "highlight_id": hl.id,
                        "anchor_type": hl.anchor_type,
                        "anchor_t": float(hl.anchor_t),
                    },
                    meta=dict(hl.meta),
                )
            )
        return tokens

    def _compile_scene_change_tokens(
        self,
        events: list[Event],
        times: np.ndarray,
        boundary: np.ndarray,
        embed: np.ndarray,
        duration_s: float,
    ) -> list[Token]:
        tokens: list[Token] = []
        signal = boundary
        source_name = "boundary_score"
        if signal.size == 0:
            signal = embed
            source_name = "embed_dist"
        if signal.size == 0:
            return tokens

        norm = _normalize(signal)
        peaks = _find_top_peaks(norm, self.cfg.scene_change_top_k)
        for idx in peaks:
            t = float(times[idx]) if idx < times.size else float(idx)
            tokens.append(
                Token(
                    id="",
                    t0=_clamp(t - 0.5, 0.0, duration_s),
                    t1=_clamp(t + 0.5, 0.0, duration_s),
                    type="SCENE_CHANGE",
                    conf=_clamp(float(norm[idx]), 0.0, 1.0),
                    source_event=_time_to_event(t, events),
                    source={"signal": source_name, "index": int(idx)},
                    meta={},
                )
            )
        return tokens

    def _merge_same_type(self, tokens: list[Token], gap_s: float) -> list[Token]:
        if not tokens:
            return []
        by_type: dict[str, list[Token]] = {}
        for token in tokens:
            by_type.setdefault(token.type, []).append(token)

        merged_all: list[Token] = []
        for token_type, items in by_type.items():
            items = sorted(items, key=lambda t: (t.t0, t.t1, -t.conf))
            current: Token | None = None
            for token in items:
                if current is None:
                    current = token.model_copy(deep=True)
                    current.meta = dict(current.meta)
                    current.meta["merged_count"] = int(current.meta.get("merged_count", 1))
                    continue

                if token.t0 <= current.t1 + gap_s:
                    current.t1 = float(max(current.t1, token.t1))
                    current.conf = float(max(current.conf, token.conf))
                    if current.source_event != token.source_event:
                        current.source_event = "mixed"
                    current.meta["merged_count"] = int(current.meta.get("merged_count", 1)) + int(
                        token.meta.get("merged_count", 1)
                    )
                    sources = current.meta.get("sources")
                    if sources is None:
                        sources = [current.source]
                    sources.append(token.source)
                    current.meta["sources"] = sources
                else:
                    merged_all.append(current)
                    current = token.model_copy(deep=True)
                    current.meta = dict(current.meta)
                    current.meta["merged_count"] = int(current.meta.get("merged_count", 1))
            if current is not None:
                merged_all.append(current)
        return merged_all
