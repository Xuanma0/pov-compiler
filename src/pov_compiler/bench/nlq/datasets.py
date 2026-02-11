from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.schemas import Output


@dataclass
class NLQSample:
    qid: str
    query: str
    query_type: str
    gt_span: tuple[float, float]
    top_k: int = 6
    distractors: list[tuple[float, float]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": str(self.qid),
            "query": str(self.query),
            "query_type": str(self.query_type),
            "gt_span": {"t0": float(self.gt_span[0]), "t1": float(self.gt_span[1])},
            "top_k": int(self.top_k),
            "distractors": [{"t0": float(t0), "t1": float(t1)} for t0, t1 in self.distractors],
            "meta": dict(self.meta),
        }


def _as_output(output_json_path: str | Path | dict[str, Any] | Output) -> Output:
    if isinstance(output_json_path, Output):
        return output_json_path
    if isinstance(output_json_path, (str, Path)):
        payload = json.loads(Path(output_json_path).read_text(encoding="utf-8"))
    elif isinstance(output_json_path, dict):
        payload = output_json_path
    else:
        raise TypeError("output_json_path must be Output, dict, or path")
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _center(span: tuple[float, float]) -> float:
    return 0.5 * (float(span[0]) + float(span[1]))


def _span_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = float(min(a[0], a[1])), float(max(a[0], a[1]))
    b0, b1 = float(min(b[0], b[1])), float(max(b[0], b[1]))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-9, (a1 - a0) + (b1 - b0) - inter)
    return float(inter / union)


def _pick_distractors(
    *,
    gt_span: tuple[float, float],
    pool: list[tuple[float, float]],
    rng: np.random.Generator,
    min_gap_s: float,
    max_count: int,
) -> list[tuple[float, float]]:
    if int(max_count) <= 0:
        return []
    gt_c = _center(gt_span)
    candidates: list[tuple[float, tuple[float, float]]] = []
    for span in pool:
        if _span_iou(gt_span, span) >= 0.1:
            continue
        if abs(_center(span) - gt_c) <= float(min_gap_s):
            continue
        dist = abs(_center(span) - gt_c)
        candidates.append((dist, span))
    if not candidates:
        return []

    # Prefer near-miss negatives so the query needs real disambiguation.
    candidates.sort(key=lambda x: x[0])
    span_only = [span for _, span in candidates]
    want = int(rng.integers(1, int(max_count) + 1))
    return span_only[: min(want, len(span_only), int(max_count))]


def _sample_indices(total: int, n: int, rng: np.random.Generator) -> list[int]:
    if total <= 0 or n <= 0:
        return []
    if n >= total:
        return list(range(total))
    picks = rng.choice(total, size=n, replace=False)
    return sorted(int(x) for x in picks.tolist())


def _anchor_types_from_highlight(hl: Any) -> set[str]:
    types = hl.meta.get("anchor_types")
    if isinstance(types, list):
        out = {str(x).lower() for x in types if str(x)}
    else:
        out = {str(hl.anchor_type).lower()}
    return {x for x in out if x}


def _highlight_span(hl: Any) -> tuple[float, float]:
    return float(hl.t0), float(hl.t1)


def _scene_change_times(output: Output) -> list[float]:
    times: list[float] = []
    for token in output.token_codec.tokens:
        if str(token.type).upper() != "SCENE_CHANGE":
            continue
        times.append(_center((float(token.t0), float(token.t1))))
    return sorted(times)


def _append_anchor_sample(
    *,
    samples: list[NLQSample],
    rng: np.random.Generator,
    top_k: int,
    gt_highlight: Any,
    pool_highlights: list[Any],
    query: str,
    anchor_type: str,
    disambiguation: str,
    distractor_min_gap_s: float,
    max_distractors: int,
    extra_meta: dict[str, Any] | None = None,
) -> bool:
    gt = _highlight_span(gt_highlight)
    pool = [_highlight_span(x) for x in pool_highlights]
    distractors = _pick_distractors(
        gt_span=gt,
        pool=pool,
        rng=rng,
        min_gap_s=float(distractor_min_gap_s),
        max_count=int(max_distractors),
    )
    if not distractors:
        return False

    meta = {
        "source_kind": "highlight",
        "anchor_type": str(anchor_type),
        "highlight_id": str(gt_highlight.id),
        "disambiguation": str(disambiguation),
    }
    if extra_meta:
        meta.update(extra_meta)

    samples.append(
        NLQSample(
            qid=f"hnlq_{len(samples) + 1:06d}",
            query=str(query),
            query_type="hard_pseudo_anchor",
            gt_span=gt,
            top_k=int(top_k),
            distractors=distractors,
            meta=meta,
        )
    )
    return True


def load_hard_pseudo_nlq(
    output_json_path: str | Path | dict[str, Any] | Output,
    *,
    seed: int = 0,
    n_highlight: int = 10,
    n_token: int = 10,
    n_decision: int = 10,
    top_k: int = 6,
    max_distractors: int = 3,
    distractor_min_gap_s: float = 10.0,
) -> list[NLQSample]:
    """Generate hard pseudo-NLQ samples without label leakage.

    Rules:
    - Natural language templates only (query text does not expose token/decision label names).
    - Add 1-3 same-type distractor spans (IoU < 0.1 and center-gap > min_gap_s).
    - hard_pseudo_anchor v2:
      if same-type anchors >= 3, generate disambiguation queries
      (first/last/after-scene-change).
    """

    output = _as_output(output_json_path)
    rng = np.random.default_rng(int(seed))
    samples: list[NLQSample] = []

    # Anchor-trigger templates intentionally include planner keywords.
    turn_head_templates = [
        "什么时候我开始左右张望？",
        "我突然回头看一眼是在什么时候？",
        "什么时候我在周围扫视了一下？",
        "When did I start looking around?",
    ]
    stop_look_templates = [
        "我停下来观察周围是在什么时候？",
        "什么时候我停住看了一下？",
        "When did I pause to look around?",
    ]
    turn_head_first_templates = [
        "我第一次左右张望是在什么时候？",
        "When was the first time I looked around?",
    ]
    turn_head_last_templates = [
        "我最后一次左右张望是在什么时候？",
        "When was the last time I looked around?",
    ]
    turn_head_after_scene_templates = [
        "场景变化之后，我左右张望是在什么时候？",
        "After the scene changed, when did I look around?",
    ]
    stop_look_first_templates = [
        "我第一次停下来观察是在什么时候？",
        "When was the first time I paused to look?",
    ]
    stop_look_last_templates = [
        "我最后一次停下来观察是在什么时候？",
        "When was the last time I paused to look?",
    ]
    stop_look_after_scene_templates = [
        "场景变化之后，我停下来观察是在什么时候？",
        "After the scene changed, when did I pause to look?",
    ]

    token_templates = {
        "SCENE_CHANGE": [
            "什么时候场景发生了明显变化？",
            "什么时候我进入了一个新环境或离开了一个区域？",
        ],
        "MOTION_MOVING": [
            "什么时候我开始持续移动？",
        ],
        "MOTION_STILL": [
            "什么时候我基本保持不动？",
        ],
    }

    decision_turn_templates = [
        "什么时候我做了一个观察周围的动作？",
        "什么时候我有一次明显的左右看动作？",
    ]
    decision_stop_templates = [
        "什么时候我做了一个短暂停下来的动作？",
        "什么时候我停住并观察了一下？",
    ]

    # 1) highlight-based queries (turn_head / stop_look), with v2 disambiguation.
    hl_turn: list[Any] = []
    hl_stop: list[Any] = []
    for hl in output.highlights:
        types = _anchor_types_from_highlight(hl)
        if "turn_head" in types:
            hl_turn.append(hl)
        if "stop_look" in types:
            hl_stop.append(hl)

    scene_times = _scene_change_times(output)

    def _emit_anchor_queries(
        *,
        anchor_type: str,
        group: list[Any],
        budget: int,
        normal_templates: list[str],
        first_templates: list[str],
        last_templates: list[str],
        after_scene_templates: list[str],
    ) -> None:
        if budget <= 0 or not group:
            return

        ordered = sorted(group, key=lambda x: (float(x.t0), float(x.t1), str(x.id)))
        emitted = 0
        used_ids: set[str] = set()

        # If enough same-type occurrences, generate 1-2 disambiguation queries.
        disambiguation_items: list[tuple[str, Any, str, dict[str, Any]]] = []
        if len(ordered) >= 3:
            first_hl = ordered[0]
            last_hl = ordered[-1]
            disambiguation_items.append(
                (
                    str(rng.choice(first_templates)),
                    first_hl,
                    "first_occurrence",
                    {"ordinal": "first"},
                )
            )
            if str(last_hl.id) != str(first_hl.id):
                disambiguation_items.append(
                    (
                        str(rng.choice(last_templates)),
                        last_hl,
                        "last_occurrence",
                        {"ordinal": "last"},
                    )
                )

            # First same-type anchor after the first scene-change that has a following anchor.
            if scene_times:
                centers = [_center(_highlight_span(hl)) for hl in ordered]
                for scene_t in scene_times:
                    after_idx = next((idx for idx, c in enumerate(centers) if c > float(scene_t)), None)
                    if after_idx is None:
                        continue
                    after_hl = ordered[after_idx]
                    disambiguation_items.append(
                        (
                            str(rng.choice(after_scene_templates)),
                            after_hl,
                            "after_scene_change",
                            {"scene_change_t": float(scene_t)},
                        )
                    )
                    break

            if disambiguation_items:
                max_disamb = min(int(budget), len(disambiguation_items), 2)
                want_disamb = int(rng.integers(1, max_disamb + 1))
                for idx in _sample_indices(len(disambiguation_items), want_disamb, rng):
                    query, target_hl, disambiguation, extra_meta = disambiguation_items[idx]
                    ok = _append_anchor_sample(
                        samples=samples,
                        rng=rng,
                        top_k=int(top_k),
                        gt_highlight=target_hl,
                        pool_highlights=ordered,
                        query=query,
                        anchor_type=anchor_type,
                        disambiguation=disambiguation,
                        distractor_min_gap_s=float(distractor_min_gap_s),
                        max_distractors=int(max_distractors),
                        extra_meta=extra_meta,
                    )
                    if ok:
                        emitted += 1
                        used_ids.add(str(target_hl.id))
                    if emitted >= int(budget):
                        return

        # Fill remaining budget with regular hard_pseudo_anchor queries.
        remaining = int(budget) - emitted
        if remaining <= 0:
            return
        candidate_idx = list(range(len(ordered)))
        rng.shuffle(candidate_idx)
        for idx in candidate_idx:
            if remaining <= 0:
                break
            hl = ordered[idx]
            if str(hl.id) in used_ids and len(ordered) > 1:
                continue
            ok = _append_anchor_sample(
                samples=samples,
                rng=rng,
                top_k=int(top_k),
                gt_highlight=hl,
                pool_highlights=ordered,
                query=str(rng.choice(normal_templates)),
                anchor_type=anchor_type,
                disambiguation="none",
                distractor_min_gap_s=float(distractor_min_gap_s),
                max_distractors=int(max_distractors),
                extra_meta=None,
            )
            if ok:
                emitted += 1
                remaining -= 1
                used_ids.add(str(hl.id))

    turn_budget = max(0, int(n_highlight // 2))
    stop_budget = max(0, int(n_highlight - turn_budget))
    _emit_anchor_queries(
        anchor_type="turn_head",
        group=hl_turn,
        budget=turn_budget,
        normal_templates=turn_head_templates,
        first_templates=turn_head_first_templates,
        last_templates=turn_head_last_templates,
        after_scene_templates=turn_head_after_scene_templates,
    )
    _emit_anchor_queries(
        anchor_type="stop_look",
        group=hl_stop,
        budget=stop_budget,
        normal_templates=stop_look_templates,
        first_templates=stop_look_first_templates,
        last_templates=stop_look_last_templates,
        after_scene_templates=stop_look_after_scene_templates,
    )

    # 2) token-based queries (no label names in query text).
    token_groups: dict[str, list[Any]] = {"SCENE_CHANGE": [], "MOTION_MOVING": [], "MOTION_STILL": []}
    for token in output.token_codec.tokens:
        token_type = str(token.type).upper()
        if token_type in token_groups:
            token_groups[token_type].append(token)

    token_per_type = max(1, int(max(1, n_token) // max(1, len(token_groups))))
    for token_type, group in token_groups.items():
        picks = _sample_indices(len(group), token_per_type, rng)
        for idx in picks:
            token = group[idx]
            gt = (float(token.t0), float(token.t1))
            pool = [(float(x.t0), float(x.t1)) for x in group]
            distractors = _pick_distractors(
                gt_span=gt,
                pool=pool,
                rng=rng,
                min_gap_s=float(distractor_min_gap_s),
                max_count=int(max_distractors),
            )
            if not distractors:
                continue
            q = str(rng.choice(token_templates[token_type]))
            samples.append(
                NLQSample(
                    qid=f"hnlq_{len(samples) + 1:06d}",
                    query=q,
                    query_type="hard_pseudo_token",
                    gt_span=gt,
                    top_k=int(top_k),
                    distractors=distractors,
                    meta={
                        "source_kind": "token",
                        "token_type": token_type,
                        "token_id": token.id,
                    },
                )
            )

    # 3) decision-based queries.
    decision_turn: list[Any] = []
    decision_stop: list[Any] = []
    for dp in output.decision_points:
        action_type = str(dp.action.get("type", "")).upper()
        if "TURN_HEAD" in action_type or "REORIENT_AND_SCAN" in action_type:
            decision_turn.append(dp)
        if "STOP_LOOK" in action_type or "REORIENT_AND_SCAN" in action_type:
            decision_stop.append(dp)

    for idx in _sample_indices(len(decision_turn), max(0, int(n_decision // 2)), rng):
        dp = decision_turn[idx]
        gt = (float(dp.t0), float(dp.t1))
        pool = [(float(x.t0), float(x.t1)) for x in decision_turn]
        distractors = _pick_distractors(
            gt_span=gt,
            pool=pool,
            rng=rng,
            min_gap_s=float(distractor_min_gap_s),
            max_count=int(max_distractors),
        )
        if not distractors:
            continue
        q = str(rng.choice(decision_turn_templates))
        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=q,
                query_type="hard_pseudo_decision",
                gt_span=gt,
                top_k=int(top_k),
                distractors=distractors,
                meta={
                    "source_kind": "decision",
                    "decision_type": "TURN_HEAD_LIKE",
                    "decision_id": dp.id,
                },
            )
        )

    for idx in _sample_indices(len(decision_stop), max(0, int(n_decision - n_decision // 2)), rng):
        dp = decision_stop[idx]
        gt = (float(dp.t0), float(dp.t1))
        pool = [(float(x.t0), float(x.t1)) for x in decision_stop]
        distractors = _pick_distractors(
            gt_span=gt,
            pool=pool,
            rng=rng,
            min_gap_s=float(distractor_min_gap_s),
            max_count=int(max_distractors),
        )
        if not distractors:
            continue
        q = str(rng.choice(decision_stop_templates))
        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=q,
                query_type="hard_pseudo_decision",
                gt_span=gt,
                top_k=int(top_k),
                distractors=distractors,
                meta={
                    "source_kind": "decision",
                    "decision_type": "STOP_LOOK_LIKE",
                    "decision_id": dp.id,
                },
            )
        )

    samples.sort(key=lambda x: (float(x.gt_span[0]), float(x.gt_span[1]), x.qid))
    for i, sample in enumerate(samples, start=1):
        sample.qid = f"hnlq_{i:06d}"
    return samples
