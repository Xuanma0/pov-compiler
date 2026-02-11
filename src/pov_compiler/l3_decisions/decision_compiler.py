from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pov_compiler.schemas import Alternative, DecisionPoint, Output, Token


@dataclass
class DecisionConfig:
    enabled: bool = True
    pre_s: float = 2.0
    post_s: float = 2.0
    merge_iou: float = 0.7
    min_gap_s: float = 0.0
    boundary_thresh: float = 0.65


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    if union <= 1e-9:
        return 0.0
    return float(inter / union)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _mean_window(times: np.ndarray, signal: np.ndarray, t0: float, t1: float) -> float:
    if times.size == 0 or signal.size == 0:
        return 0.0
    idx = np.where((times >= t0) & (times <= t1))[0]
    if idx.size == 0:
        nearest = int(np.argmin(np.abs(times - ((t0 + t1) * 0.5))))
        return float(signal[nearest])
    return float(np.mean(signal[idx]))


def _max_window(times: np.ndarray, signal: np.ndarray, t0: float, t1: float) -> float:
    if times.size == 0 or signal.size == 0:
        return 0.0
    idx = np.where((times >= t0) & (times <= t1))[0]
    if idx.size == 0:
        nearest = int(np.argmin(np.abs(times - ((t0 + t1) * 0.5))))
        return float(signal[nearest])
    return float(np.max(signal[idx]))


def _token_ids_in_window(tokens: list[Token], t0: float, t1: float) -> list[str]:
    ids = [token.id for token in tokens if _overlap(token.t0, token.t1, t0, t1)]
    return sorted(set(ids))


def _token_types_in_window(tokens: list[Token], t0: float, t1: float) -> list[str]:
    types = [token.type for token in tokens if _overlap(token.t0, token.t1, t0, t1)]
    return sorted(set(types))


def _scene_change_nearby(tokens: list[Token], t0: float, t1: float) -> bool:
    return any(token.type == "SCENE_CHANGE" and _overlap(token.t0, token.t1, t0, t1) for token in tokens)


def _motion_state(value: float) -> str:
    return "MOVING" if value >= 0.5 else "STILL"


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _build_constraints(
    motion_change_peak: float,
    embed_mean: float,
    motion_mean: float,
    event_duration: float,
    event_highlight_coverage: float,
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    if motion_change_peak > 0.7:
        constraints.append(
            {
                "type": "STABILITY_CONSTRAINT",
                "score": _clamp01(motion_change_peak),
                "explanation": "Motion perturbation is high around decision window.",
                "evidence": {"motion_change_peak": float(motion_change_peak)},
            }
        )
    visibility_score = 0.5 * embed_mean + 0.5 * (1.0 - motion_mean)
    if embed_mean > 0.6 and motion_mean < 0.4:
        constraints.append(
            {
                "type": "VISIBILITY_CONSTRAINT",
                "score": _clamp01(visibility_score),
                "explanation": "Viewpoint likely changed while movement stayed low.",
                "evidence": {"embed_mean": float(embed_mean), "motion_mean": float(motion_mean)},
            }
        )
    if event_duration > 25.0 and event_highlight_coverage < 0.25:
        score = _clamp01((event_duration / 120.0) + (0.25 - event_highlight_coverage))
        constraints.append(
            {
                "type": "BUDGET_CONSTRAINT",
                "score": score,
                "explanation": "Long event with sparse preserved highlights under memory budget.",
                "evidence": {
                    "event_duration_s": float(event_duration),
                    "highlight_coverage": float(event_highlight_coverage),
                },
            }
        )
    return constraints[:3]


def _build_alternatives(action_type: str, stability_score: float) -> list[Alternative]:
    moving_risk_boost = 0.2 * stability_score

    if action_type == "ATTENTION_STOP_LOOK":
        return [
            Alternative(
                action_type="CONTINUE_MOVING",
                rationale="Keep moving without pausing.",
                expected_outcome="Faster progress but may miss key details.",
                risk_delta=0.2 + moving_risk_boost,
                conf=0.68,
            ),
            Alternative(
                action_type="SHORTER_PAUSE",
                rationale="Pause for less time and resume early.",
                expected_outcome="Lower information gain but better time efficiency.",
                risk_delta=0.1 + 0.5 * moving_risk_boost,
                conf=0.72,
            ),
        ]
    if action_type == "ATTENTION_TURN_HEAD":
        return [
            Alternative(
                action_type="LOOK_FORWARD_ONLY",
                rationale="Maintain forward gaze to reduce viewpoint noise.",
                expected_outcome="More stable trajectory but weaker peripheral checking.",
                risk_delta=0.15 + 0.5 * moving_risk_boost,
                conf=0.66,
            ),
            Alternative(
                action_type="TURN_HEAD_OPPOSITE",
                rationale="Inspect opposite side for alternative cues.",
                expected_outcome="May reveal different cues with moderate overhead.",
                risk_delta=0.05 + 0.3 * moving_risk_boost,
                conf=0.63,
            ),
        ]
    if action_type == "REORIENT_AND_SCAN":
        return [
            Alternative(
                action_type="STOP_WITHOUT_TURN",
                rationale="Pause but keep orientation unchanged.",
                expected_outcome="Improves stability with narrower visual coverage.",
                risk_delta=0.08,
                conf=0.67,
            ),
            Alternative(
                action_type="TURN_WITHOUT_STOP",
                rationale="Scan surroundings while continuing motion.",
                expected_outcome="Broader exploration but higher motion blur risk.",
                risk_delta=0.18 + moving_risk_boost,
                conf=0.61,
            ),
        ]
    if action_type == "TRANSITION":
        return [
            Alternative(
                action_type="DELAY_TRANSITION",
                rationale="Delay entering the next segment to gather more context.",
                expected_outcome="Potentially safer transition but slower progress.",
                risk_delta=-0.05,
                conf=0.62,
            ),
            Alternative(
                action_type="EARLY_TRANSITION",
                rationale="Commit to transition earlier to reduce dwell time.",
                expected_outcome="Faster continuation with potentially missing context.",
                risk_delta=0.12,
                conf=0.60,
            ),
        ]
    return [
        Alternative(
            action_type="KEEP_CURRENT",
            rationale="Hold current strategy.",
            expected_outcome="Stable behavior with limited exploration.",
            risk_delta=0.0,
            conf=0.55,
        ),
        Alternative(
            action_type="SWITCH_STRATEGY",
            rationale="Adopt a different exploratory behavior.",
            expected_outcome="Potential extra cues with uncertainty.",
            risk_delta=0.1,
            conf=0.5,
        ),
    ]


class DecisionCompiler:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.cfg = DecisionConfig(
            enabled=bool(cfg.get("enabled", True)),
            pre_s=float(cfg.get("pre_s", 2.0)),
            post_s=float(cfg.get("post_s", 2.0)),
            merge_iou=float(cfg.get("merge_iou", 0.7)),
            min_gap_s=float(cfg.get("min_gap_s", 0.0)),
            boundary_thresh=float(cfg.get("boundary_thresh", 0.65)),
        )

    def compile(self, output: Output) -> list[DecisionPoint]:
        if not self.cfg.enabled:
            return []

        duration = float(output.meta.get("duration_s", 0.0))
        events = sorted(output.events, key=lambda e: (e.t0, e.t1))
        tokens = list(output.token_codec.tokens)

        signals = output.debug.get("signals", {}) if isinstance(output.debug, dict) else {}
        times = np.asarray(signals.get("time", []), dtype=np.float32)
        motion = _normalize(np.asarray(signals.get("motion_energy", []), dtype=np.float32))
        embed = _normalize(np.asarray(signals.get("embed_dist", []), dtype=np.float32))
        boundary = _normalize(np.asarray(signals.get("boundary_score", []), dtype=np.float32))
        motion_change = _normalize(np.abs(np.diff(motion, prepend=motion[0])) if motion.size > 0 else np.asarray([], dtype=np.float32))

        event_map = {event.id: event for event in events}
        event_highlight_coverage: dict[str, float] = {}
        for event in events:
            event_dur = max(1e-6, float(event.t1 - event.t0))
            total_hl = sum(
                max(0.0, min(event.t1, hl.t1) - max(event.t0, hl.t0))
                for hl in output.highlights
                if hl.source_event == event.id and _overlap(event.t0, event.t1, hl.t0, hl.t1)
            )
            event_highlight_coverage[event.id] = float(total_hl / event_dur)

        trigger_specs: list[dict[str, Any]] = []
        for hl in output.highlights:
            anchor_types = hl.meta.get("anchor_types")
            if not isinstance(anchor_types, list):
                anchor_types = [hl.anchor_type]
            t = float(hl.anchor_t if hl.t0 <= hl.anchor_t <= hl.t1 else (hl.t0 + hl.t1) * 0.5)
            trigger_specs.append(
                {
                    "t": t,
                    "source_event": hl.source_event,
                    "source_highlight": hl.id,
                    "anchor_types": [str(x).lower() for x in anchor_types],
                    "anchor_t": float(hl.anchor_t),
                    "conf": float(hl.conf),
                    "trigger_t0": float(hl.t0),
                    "trigger_t1": float(hl.t1),
                }
            )

        for event in events:
            for anchor in event.anchors:
                covered = any(
                    hl.source_event == event.id and _overlap(anchor.t, anchor.t, hl.t0, hl.t1)
                    for hl in output.highlights
                )
                if covered:
                    continue
                trigger_specs.append(
                    {
                        "t": float(anchor.t),
                        "source_event": event.id,
                        "source_highlight": None,
                        "anchor_types": [str(anchor.type).lower()],
                        "anchor_t": float(anchor.t),
                        "conf": float(anchor.conf),
                        "trigger_t0": float(anchor.t),
                        "trigger_t1": float(anchor.t),
                    }
                )

        trigger_specs.sort(key=lambda x: float(x["t"]))
        decisions: list[DecisionPoint] = []
        for spec in trigger_specs:
            t = float(spec["t"])
            source_event = str(spec["source_event"])
            source_highlight = spec["source_highlight"]

            t0 = max(0.0, t - self.cfg.pre_s)
            t1 = min(duration, t + self.cfg.post_s)

            nearby_token_ids = _token_ids_in_window(tokens, t0, t1)
            nearby_token_types = _token_types_in_window(tokens, t0, t1)
            trigger_token_ids = _token_ids_in_window(tokens, float(spec["trigger_t0"]), float(spec["trigger_t1"]))

            motion_before = _mean_window(times, motion, max(0.0, t - self.cfg.pre_s), t)
            motion_after = _mean_window(times, motion, t, min(duration, t + self.cfg.post_s))
            embed_mean = _mean_window(times, embed, t0, t1)
            boundary_near = _max_window(times, boundary, t - 0.5, t + 0.5) >= self.cfg.boundary_thresh
            scene_change_near = _scene_change_nearby(tokens, t - 0.6, t + 0.6)
            motion_change_peak = _max_window(times, motion_change, t0, t1)

            state = {
                "event_id": source_event,
                "motion_state_before": _motion_state(motion_before),
                "motion_state_after": _motion_state(motion_after),
                "boundary_nearby": bool(boundary_near),
                "scene_change_nearby": bool(scene_change_near),
                "nearby_tokens": nearby_token_types,
                "evidence": {
                    "token_ids": nearby_token_ids,
                    "signal_summary": {
                        "motion_mean_before": float(motion_before),
                        "motion_mean_after": float(motion_after),
                        "embed_mean": float(embed_mean),
                    },
                },
            }

            anchor_types = set(spec["anchor_types"])
            if "turn_head" in anchor_types and "stop_look" in anchor_types:
                action_type = "REORIENT_AND_SCAN"
            elif "turn_head" in anchor_types:
                action_type = "ATTENTION_TURN_HEAD"
            elif "stop_look" in anchor_types:
                action_type = "ATTENTION_STOP_LOOK"
            else:
                action_type = "MONITOR"
            if boundary_near and scene_change_near:
                action_type = "TRANSITION"

            consistency = 1.0 if state["motion_state_before"] != state["motion_state_after"] else 0.5
            action_conf = _clamp01(0.6 * float(spec["conf"]) + 0.25 * consistency + 0.15 * float(scene_change_near))
            action = {
                "type": action_type,
                "conf": float(action_conf),
                "anchor_types": sorted(anchor_types),
            }

            event = event_map.get(source_event)
            event_duration = float(event.t1 - event.t0) if event is not None else 0.0
            constraints = _build_constraints(
                motion_change_peak=float(motion_change_peak),
                embed_mean=float(embed_mean),
                motion_mean=float((motion_before + motion_after) * 0.5),
                event_duration=event_duration,
                event_highlight_coverage=float(event_highlight_coverage.get(source_event, 0.0)),
            )
            stability_score = max(
                [float(c["score"]) for c in constraints if c.get("type") == "STABILITY_CONSTRAINT"] or [0.0]
            )

            motion_delta = motion_after - motion_before
            post_scene_change = _scene_change_nearby(tokens, t, min(duration, t + self.cfg.post_s))
            if post_scene_change or embed_mean > 0.72:
                outcome_type = "SCENE_CHANGED"
                outcome_conf = _clamp01(max(embed_mean, motion_change_peak))
            elif state["motion_state_before"] == "STILL" and state["motion_state_after"] == "MOVING":
                outcome_type = "RESUME_MOVING"
                outcome_conf = _clamp01(0.5 + abs(motion_delta))
            elif state["motion_state_before"] == "MOVING" and state["motion_state_after"] == "STILL":
                outcome_type = "STOPPED"
                outcome_conf = _clamp01(0.5 + abs(motion_delta))
            elif state["motion_state_before"] == "STILL" and state["motion_state_after"] == "STILL":
                outcome_type = "STILL_CONTINUE"
                outcome_conf = _clamp01(0.55 + 0.2 * (1.0 - motion_after))
            elif motion_delta > 0.15:
                outcome_type = "MOTION_INCREASE"
                outcome_conf = _clamp01(0.5 + motion_delta)
            else:
                outcome_type = "STILL_CONTINUE"
                outcome_conf = _clamp01(0.45 + (1.0 - abs(motion_delta)))

            outcome_token_ids = _token_ids_in_window(tokens, t, min(duration, t + self.cfg.post_s))
            outcome = {
                "type": outcome_type,
                "conf": float(outcome_conf),
                "evidence": {
                    "token_ids": outcome_token_ids,
                    "signal_summary": {
                        "motion_mean_before": float(motion_before),
                        "motion_mean_after": float(motion_after),
                        "motion_delta": float(motion_delta),
                    },
                },
            }

            alternatives = _build_alternatives(action_type, stability_score)

            trigger = {
                "anchor_type": sorted(anchor_types)[0] if anchor_types else "unknown",
                "anchor_types": sorted(anchor_types),
                "anchor_t": float(spec["anchor_t"]),
                "conf": float(spec["conf"]),
                "token_ids": trigger_token_ids,
            }

            overall_conf = _clamp01(0.4 * float(spec["conf"]) + 0.35 * action_conf + 0.25 * outcome_conf)
            decisions.append(
                DecisionPoint(
                    id="",
                    t=float(t),
                    t0=float(t0),
                    t1=float(t1),
                    source_event=source_event,
                    source_highlight=source_highlight,
                    trigger=trigger,
                    state=state,
                    action=action,
                    constraints=constraints,
                    outcome=outcome,
                    alternatives=alternatives,
                    conf=float(overall_conf),
                    meta={"merged_count": 1},
                )
            )

        decisions = self._merge_overlaps(decisions)
        for i, decision in enumerate(decisions, start=1):
            decision.id = f"dp_{i:06d}"
        return decisions

    def _merge_overlaps(self, decisions: list[DecisionPoint]) -> list[DecisionPoint]:
        if not decisions:
            return []
        decisions = sorted(decisions, key=lambda d: (d.t, d.t0, d.t1))
        merged: list[DecisionPoint] = []
        for decision in decisions:
            if not merged:
                merged.append(decision)
                continue
            prev = merged[-1]
            iou = _interval_iou(prev.t0, prev.t1, decision.t0, decision.t1)
            if iou < self.cfg.merge_iou:
                if self.cfg.min_gap_s > 0 and (decision.t - prev.t) < self.cfg.min_gap_s:
                    if decision.conf > prev.conf:
                        merged[-1] = decision
                    continue
                merged.append(decision)
                continue

            keep, other = (decision, prev) if decision.conf >= prev.conf else (prev, decision)
            weight_keep = max(1e-6, float(keep.conf))
            weight_other = max(1e-6, float(other.conf))

            keep.t0 = float(min(keep.t0, other.t0))
            keep.t1 = float(max(keep.t1, other.t1))
            keep.t = float((keep.t * weight_keep + other.t * weight_other) / (weight_keep + weight_other))
            keep.conf = float(max(keep.conf, other.conf))

            token_ids = set(keep.trigger.get("token_ids", [])) | set(other.trigger.get("token_ids", []))
            keep.trigger["token_ids"] = sorted(token_ids)

            anchor_types = set(keep.trigger.get("anchor_types", [])) | set(other.trigger.get("anchor_types", []))
            if anchor_types:
                keep.trigger["anchor_types"] = sorted(anchor_types)
                keep.trigger["anchor_type"] = sorted(anchor_types)[0]

            nearby_tokens = set(keep.state.get("nearby_tokens", [])) | set(other.state.get("nearby_tokens", []))
            keep.state["nearby_tokens"] = sorted(nearby_tokens)

            constraints_map: dict[str, dict[str, Any]] = {}
            for c in list(keep.constraints) + list(other.constraints):
                c_type = str(c.get("type", "UNKNOWN"))
                prev_c = constraints_map.get(c_type)
                if prev_c is None or float(c.get("score", 0.0)) > float(prev_c.get("score", 0.0)):
                    constraints_map[c_type] = dict(c)
            keep.constraints = list(constraints_map.values())

            alternatives = list(keep.alternatives) if len(keep.alternatives) >= len(other.alternatives) else list(other.alternatives)
            keep.alternatives = alternatives[:]

            keep.meta["merged_count"] = int(keep.meta.get("merged_count", 1)) + int(other.meta.get("merged_count", 1))
            if keep.source_highlight is None and other.source_highlight is not None:
                keep.source_highlight = other.source_highlight

            merged[-1] = keep

        return merged
