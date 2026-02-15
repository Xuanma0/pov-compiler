from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pov_compiler.ir.events_v1 import ensure_events_v1
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


def _sample_indices(total: int, n: int, rng: np.random.Generator) -> list[int]:
    if total <= 0 or n <= 0:
        return []
    if n >= total:
        return list(range(total))
    picks = rng.choice(total, size=n, replace=False)
    return sorted(int(x) for x in picks.tolist())


def _highlight_span(hl: Any) -> tuple[float, float]:
    return float(hl.t0), float(hl.t1)


def _anchor_types_from_highlight(hl: Any) -> set[str]:
    types = hl.meta.get("anchor_types")
    if isinstance(types, list):
        out = {str(x).lower() for x in types if str(x)}
    else:
        out = {str(hl.anchor_type).lower()}
    return {x for x in out if x}


def _scene_change_times(output: Output) -> list[float]:
    times: list[float] = []
    for token in output.token_codec.tokens:
        if str(token.type).upper() != "SCENE_CHANGE":
            continue
        times.append(_center((float(token.t0), float(token.t1))))
    return sorted(times)


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
    candidates.sort(key=lambda x: x[0])
    span_only = [span for _, span in candidates]
    want = int(rng.integers(1, int(max_count) + 1))
    return span_only[: min(want, len(span_only), int(max_count))]


def _pick_after_scene_distractors(
    *,
    gt_span: tuple[float, float],
    pool: list[tuple[float, float]],
    scene_t: float,
    rng: np.random.Generator,
    max_count: int,
) -> list[tuple[float, float]]:
    """Prefer near-miss negatives around the scene-change boundary."""

    if int(max_count) <= 0:
        return []
    gt_c = _center(gt_span)
    candidates: list[tuple[float, tuple[float, float]]] = []
    for span in pool:
        if _span_iou(gt_span, span) >= 0.1:
            continue
        c = _center(span)
        if abs(c - gt_c) <= 1.0:
            continue
        # near misses around scene boundary and around GT, but not overlapping
        around_scene = abs(c - float(scene_t)) <= 30.0
        around_gt = abs(c - gt_c) <= 20.0
        if not (around_scene or around_gt):
            continue
        candidates.append((abs(c - gt_c), span))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    want = int(rng.integers(1, int(max_count) + 1))
    return [span for _, span in candidates[: min(want, len(candidates), int(max_count))]]


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
    scene_t_for_distractors: float | None = None,
) -> bool:
    gt = _highlight_span(gt_highlight)
    pool = [_highlight_span(x) for x in pool_highlights]
    if scene_t_for_distractors is not None:
        distractors = _pick_after_scene_distractors(
            gt_span=gt,
            pool=pool,
            scene_t=float(scene_t_for_distractors),
            rng=rng,
            max_count=int(max_distractors),
        )
    else:
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
    """Generate hard pseudo-NLQ samples without label leakage."""

    output = ensure_events_v1(_as_output(output_json_path))
    rng = np.random.default_rng(int(seed))
    samples: list[NLQSample] = []

    turn_head_templates = [
        "When did I start looking around?",
        "When did I start looking around to the side?",
        "When did I begin looking around quickly?",
    ]
    stop_look_templates = [
        "When did I pause to look around?",
        "When did I pause briefly to observe?",
        "When did I pause and look at something?",
    ]
    turn_head_first_templates = [
        "When was the first time I started looking around?",
    ]
    turn_head_last_templates = [
        "When was the last time I was looking around?",
    ]
    turn_head_after_scene_first_templates = [
        "After the scene changed, when was the first time I started looking around?",
    ]
    turn_head_after_scene_last_templates = [
        "After the scene changed, when was the last time I was looking around?",
    ]
    stop_look_first_templates = [
        "When was the first time I paused to look around?",
    ]
    stop_look_last_templates = [
        "When was the last time I paused to look around?",
    ]
    stop_look_after_scene_first_templates = [
        "After the scene changed, when was the first time I paused to look around?",
    ]
    stop_look_after_scene_last_templates = [
        "After the scene changed, when was the last time I paused to look around?",
    ]

    token_templates: dict[str, list[str]] = {
        "SCENE_CHANGE": [
            "When did the scene change noticeably?",
            "When did I enter a new area or leave one?",
        ],
        "MOTION_MOVING": [
            "When did I start moving continuously?",
        ],
        "MOTION_STILL": [
            "When did I stay mostly still?",
        ],
    }

    decision_turn_templates = [
        "When did I make a quick look-around action?",
        "When did I do a side-check movement?",
    ]
    decision_stop_templates = [
        "When did I make a brief pause decision?",
        "When did I stop shortly to inspect?",
    ]
    contact_templates = [
        "When was I actively handling something with my hands?",
        "When did I interact directly with an object nearby?",
        "When was I manipulating an object up close?",
    ]
    place_first_templates = [
        "In this place, when was the first relevant moment?",
        "After entering this area, when was the first key moment?",
    ]
    place_last_templates = [
        "In this place, when was the last relevant moment?",
        "Near the end of this area, when was the final key moment?",
    ]
    interaction_object_templates = [
        "When did I interact with the {obj} directly?",
        "When was I handling the {obj} with my hands?",
    ]
    interaction_generic_templates = [
        "When was I actively interacting with an object?",
        "When did I handle something with my hands?",
    ]
    lost_object_contact_templates = [
        "When did I last interact with the {obj}?",
        "Find the moment I last touched the {obj}.",
    ]
    lost_object_seen_templates = [
        "When did I last see the {obj}?",
        "Find the last moment the {obj} appeared.",
    ]

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
        after_scene_first_templates: list[str],
        after_scene_last_templates: list[str],
    ) -> None:
        if budget <= 0 or not group:
            return

        ordered = sorted(group, key=lambda x: (float(x.t0), float(x.t1), str(x.id)))
        emitted = 0
        used_ids: set[str] = set()

        disambiguation_items: list[dict[str, Any]] = []
        if len(ordered) >= 3:
            disambiguation_items.append(
                {
                    "query": str(rng.choice(first_templates)),
                    "target": ordered[0],
                    "disambiguation": "first_occurrence",
                    "extra_meta": {"ordinal": "first"},
                    "scene_t_for_distractors": None,
                }
            )
            if str(ordered[-1].id) != str(ordered[0].id):
                disambiguation_items.append(
                    {
                        "query": str(rng.choice(last_templates)),
                        "target": ordered[-1],
                        "disambiguation": "last_occurrence",
                        "extra_meta": {"ordinal": "last"},
                        "scene_t_for_distractors": None,
                    }
                )

            # Make after_scene_change queries satisfiable: pick GT from a bounded post-scene window.
            for scene_t in scene_times:
                window_after = [
                    hl
                    for hl in ordered
                    if _center(_highlight_span(hl)) > float(scene_t)
                    and _center(_highlight_span(hl)) <= float(scene_t) + 90.0
                ]
                if len(window_after) < 1:
                    continue
                use_first = bool(rng.integers(0, 2) == 0)
                target = window_after[0] if use_first else window_after[-1]
                query = (
                    str(rng.choice(after_scene_first_templates))
                    if use_first
                    else str(rng.choice(after_scene_last_templates))
                )
                disambiguation_items.append(
                    {
                        "query": query,
                        "target": target,
                        "disambiguation": "after_scene_change",
                        "extra_meta": {
                            "scene_change_t": float(scene_t),
                            "ordinal": "first" if use_first else "last",
                        },
                        "scene_t_for_distractors": float(scene_t),
                    }
                )
                break

        if disambiguation_items:
            # Encourage at least one disambiguation query.
            max_disamb = min(int(budget), len(disambiguation_items), 2)
            want_disamb = int(rng.integers(1, max_disamb + 1))
            for idx in _sample_indices(len(disambiguation_items), want_disamb, rng):
                item = disambiguation_items[idx]
                ok = _append_anchor_sample(
                    samples=samples,
                    rng=rng,
                    top_k=int(top_k),
                    gt_highlight=item["target"],
                    pool_highlights=ordered,
                    query=item["query"],
                    anchor_type=anchor_type,
                    disambiguation=item["disambiguation"],
                    distractor_min_gap_s=float(distractor_min_gap_s),
                    max_distractors=int(max_distractors),
                    extra_meta=item["extra_meta"],
                    scene_t_for_distractors=item["scene_t_for_distractors"],
                )
                if ok:
                    emitted += 1
                    used_ids.add(str(item["target"].id))
                if emitted >= int(budget):
                    return

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
                scene_t_for_distractors=None,
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
        after_scene_first_templates=turn_head_after_scene_first_templates,
        after_scene_last_templates=turn_head_after_scene_last_templates,
    )
    _emit_anchor_queries(
        anchor_type="stop_look",
        group=hl_stop,
        budget=stop_budget,
        normal_templates=stop_look_templates,
        first_templates=stop_look_first_templates,
        last_templates=stop_look_last_templates,
        after_scene_first_templates=stop_look_after_scene_first_templates,
        after_scene_last_templates=stop_look_after_scene_last_templates,
    )

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
                min_gap_s=max(2.0, float(distractor_min_gap_s) * 0.5),
                max_count=int(max_distractors),
            )
            if not distractors:
                continue
            samples.append(
                NLQSample(
                    qid=f"hnlq_{len(samples) + 1:06d}",
                    query=str(rng.choice(token_templates[token_type])),
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
            min_gap_s=max(2.0, float(distractor_min_gap_s) * 0.5),
            max_count=int(max_distractors),
        )
        if not distractors:
            continue
        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=str(rng.choice(decision_turn_templates)),
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
            min_gap_s=max(2.0, float(distractor_min_gap_s) * 0.5),
            max_count=int(max_distractors),
        )
        if not distractors:
            continue
        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=str(rng.choice(decision_stop_templates)),
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

    contact_events: list[Any] = []
    for ev in output.events_v1:
        has_contact = any(str(e.type) == "contact" for e in ev.evidence)
        if has_contact or str(getattr(ev, "label", "")).strip().lower() == "interaction-heavy":
            contact_events.append(ev)
    if not contact_events:
        contact_events = list(sorted(output.events_v1, key=lambda x: (float(x.t0), float(x.t1), str(x.id))))[:3]
    contact_budget = max(0, min(len(contact_events), max(1, int(n_highlight // 3))))
    for idx in _sample_indices(len(contact_events), contact_budget, rng):
        ev = contact_events[idx]
        gt = (float(ev.t0), float(ev.t1))
        pool = [(float(x.t0), float(x.t1)) for x in output.events_v1]
        distractors = _pick_distractors(
            gt_span=gt,
            pool=pool,
            rng=rng,
            min_gap_s=max(4.0, float(distractor_min_gap_s) * 0.6),
            max_count=int(max_distractors),
        )
        if not distractors:
            fallback_pool = [span for span in pool if _span_iou(gt, span) < 0.1]
            distractors = fallback_pool[: max(1, int(max_distractors))]
        if not distractors:
            fallback_pool = [span for span in pool if span != gt]
            distractors = fallback_pool[: max(1, int(max_distractors))]
        if not distractors:
            continue
        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=str(rng.choice(contact_templates)),
                query_type="hard_pseudo_contact",
                gt_span=gt,
                top_k=int(top_k),
                distractors=distractors,
                meta={
                    "source_kind": "event_v1",
                    "event_v1_id": str(ev.id),
                    "event_label": str(getattr(ev, "label", "")),
                },
            )
        )

    # hard_pseudo_place: place-first/place-last style disambiguation.
    place_groups: dict[str, list[Any]] = {}
    for ev in output.events_v1:
        place_id = str(getattr(ev, "place_segment_id", "") or "").strip()
        if not place_id:
            continue
        place_groups.setdefault(place_id, []).append(ev)
    for values in place_groups.values():
        values.sort(key=lambda x: (float(x.t0), float(x.t1), str(x.id)))

    place_budget = max(1, int(n_highlight // 3))
    place_ids = sorted(place_groups.keys())
    place_picks = _sample_indices(len(place_ids), place_budget, rng)
    for idx in place_picks:
        pid = place_ids[idx]
        group = place_groups.get(pid, [])
        if not group:
            continue
        choose_first = bool(rng.integers(0, 2) == 0)
        target = group[0] if choose_first else group[-1]
        gt = (float(target.t0), float(target.t1))
        template = str(rng.choice(place_first_templates if choose_first else place_last_templates))
        query = f"{template} place={'first' if choose_first else 'last'}"
        pool = [(float(ev.t0), float(ev.t1)) for evs in place_groups.values() for ev in evs]
        distractors = _pick_distractors(
            gt_span=gt,
            pool=pool,
            rng=rng,
            min_gap_s=max(2.0, float(distractor_min_gap_s) * 0.5),
            max_count=int(max_distractors),
        )
        if not distractors:
            continue
        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=query,
                query_type="hard_pseudo_place",
                gt_span=gt,
                top_k=int(top_k),
                distractors=distractors,
                meta={
                    "source_kind": "event_v1",
                    "event_v1_id": str(target.id),
                    "gt_place_segment_id": str(pid),
                    "place": "first" if choose_first else "last",
                },
            )
        )

    # hard_pseudo_interaction: object/contact focused.
    interaction_events = sorted(
        list(output.events_v1),
        key=lambda ev: (
            float(getattr(ev, "interaction_score", 0.0) or 0.0),
            float(ev.t1) - float(ev.t0),
        ),
        reverse=True,
    )
    interaction_budget = max(1, int(n_highlight // 3))
    for idx in _sample_indices(len(interaction_events), interaction_budget, rng):
        ev = interaction_events[idx]
        gt = (float(ev.t0), float(ev.t1))
        obj = str(getattr(ev, "interaction_primary_object", "") or "").strip().lower()
        query = (
            str(rng.choice(interaction_object_templates)).format(obj=obj)
            if obj
            else str(rng.choice(interaction_generic_templates))
        )
        if obj:
            query = f"{query} interaction_object={obj}"
        query = f"{query} interaction_min=0.30"
        pool = [(float(x.t0), float(x.t1)) for x in output.events_v1]
        distractors = _pick_distractors(
            gt_span=gt,
            pool=pool,
            rng=rng,
            min_gap_s=max(2.0, float(distractor_min_gap_s) * 0.4),
            max_count=int(max_distractors),
        )
        if not distractors:
            continue
        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=query,
                query_type="hard_pseudo_interaction",
                gt_span=gt,
                top_k=int(top_k),
                distractors=distractors,
                meta={
                    "source_kind": "event_v1",
                    "event_v1_id": str(ev.id),
                    "interaction_object": obj,
                    "interaction_score": float(getattr(ev, "interaction_score", 0.0) or 0.0),
                    "gt_place_segment_id": str(getattr(ev, "place_segment_id", "") or ""),
                },
            )
        )

    # hard_pseudo_lost_object: last seen/contact object-memory queries.
    object_memory_items = sorted(
        list(getattr(output, "object_memory_v0", []) or []),
        key=lambda x: (
            int(getattr(x, "last_contact_t_ms", 0) or 0),
            int(getattr(x, "last_seen_t_ms", 0) or 0),
            str(getattr(x, "object_name", "")),
        ),
        reverse=True,
    )
    duration_s = float(output.meta.get("duration_s", 0.0) or 0.0)
    lost_budget = max(1, min(len(object_memory_items), max(1, int(n_highlight // 3))))
    for idx in _sample_indices(len(object_memory_items), lost_budget, rng):
        item = object_memory_items[idx]
        object_name = str(getattr(item, "object_name", "")).strip().lower()
        if not object_name:
            continue
        anchor_ms_raw = getattr(item, "last_contact_t_ms", None)
        prefer_contact = anchor_ms_raw is not None
        if anchor_ms_raw is None:
            anchor_ms_raw = getattr(item, "last_seen_t_ms", None)
        if anchor_ms_raw is None:
            continue
        anchor_s = float(anchor_ms_raw) / 1000.0
        t0 = max(0.0, anchor_s - 2.0)
        t1 = anchor_s + 2.0
        if duration_s > 0:
            t1 = min(duration_s, t1)
        if t1 <= t0:
            t1 = t0 + 0.5
        gt = (float(t0), float(t1))

        event_pool = []
        for ev in output.events_v1:
            obj = str(getattr(ev, "interaction_primary_object", "") or "").strip().lower()
            if obj and (object_name in obj or obj in object_name):
                event_pool.append((float(ev.t0), float(ev.t1)))
        if not event_pool:
            event_pool = [(float(ev.t0), float(ev.t1)) for ev in output.events_v1]

        distractors = _pick_distractors(
            gt_span=gt,
            pool=event_pool,
            rng=rng,
            min_gap_s=max(3.0, float(distractor_min_gap_s) * 0.5),
            max_count=int(max_distractors),
        )
        if not distractors:
            continue

        if prefer_contact:
            query = str(rng.choice(lost_object_contact_templates)).format(obj=object_name)
            query = f"{query} lost_object={object_name}"
        else:
            query = str(rng.choice(lost_object_seen_templates)).format(obj=object_name)
            query = f"{query} object_last_seen={object_name}"

        samples.append(
            NLQSample(
                qid=f"hnlq_{len(samples) + 1:06d}",
                query=query,
                query_type="hard_pseudo_lost_object",
                gt_span=gt,
                top_k=int(top_k),
                distractors=distractors,
                meta={
                    "source_kind": "object_memory_v0",
                    "object_name": object_name,
                    "last_contact_t_ms": int(getattr(item, "last_contact_t_ms", 0) or 0),
                    "last_seen_t_ms": int(getattr(item, "last_seen_t_ms", 0) or 0),
                    "prefer_contact": bool(prefer_contact),
                    "last_place_id": str(getattr(item, "last_place_id", "") or ""),
                },
            )
        )

    samples.sort(key=lambda x: (float(x.gt_span[0]), float(x.gt_span[1]), x.qid))
    for i, sample in enumerate(samples, start=1):
        sample.qid = f"hnlq_{i:06d}"
    return samples


def load_hard_pseudo_chain(
    output_json_path: str | Path | dict[str, Any] | Output,
    *,
    seed: int = 0,
    n_chain: int = 12,
    n_highlight: int = 10,
    n_token: int = 10,
    n_decision: int = 10,
    top_k: int = 6,
    max_distractors: int = 3,
    distractor_min_gap_s: float = 10.0,
) -> list[NLQSample]:
    base = load_hard_pseudo_nlq(
        output_json_path=output_json_path,
        seed=int(seed),
        n_highlight=int(n_highlight),
        n_token=int(n_token),
        n_decision=int(n_decision),
        top_k=int(top_k),
        max_distractors=int(max_distractors),
        distractor_min_gap_s=float(distractor_min_gap_s),
    )
    if not base:
        return []

    rng = np.random.default_rng(int(seed))
    by_type: dict[str, list[NLQSample]] = {}
    for sample in base:
        by_type.setdefault(str(sample.query_type), []).append(sample)
    for values in by_type.values():
        values.sort(key=lambda x: (float(x.gt_span[0]), float(x.gt_span[1]), str(x.qid)))

    combos: list[dict[str, Any]] = [
        {
            "name": "lost_object_to_scene_change_object_derived",
            "step1": ["hard_pseudo_lost_object"],
            "step2": ["hard_pseudo_token"],
            "derive": "time+object",
            "place_mode": "off",
            "object_mode": "hard",
            "time_mode": "hard",
            "force_step2": "token=SCENE_CHANGE which=last top_k={top_k}",
            "force_step1": "lost_object=door which=last top_k={top_k}",
        },
        {
            "name": "place_or_interaction_to_lost_object",
            "step1": ["hard_pseudo_place", "hard_pseudo_interaction"],
            "step2": ["hard_pseudo_lost_object"],
            "derive": "time+place+object",
            "place_mode": "hard",
            "object_mode": "soft",
            "time_mode": "hard",
        },
        {
            "name": "anchor_to_decision",
            "step1": ["hard_pseudo_anchor"],
            "step2": ["hard_pseudo_decision"],
            "derive": "time_only",
            "place_mode": "off",
            "object_mode": "off",
            "time_mode": "hard",
        },
        {
            "name": "decision_to_token",
            "step1": ["hard_pseudo_decision"],
            "step2": ["hard_pseudo_token"],
            "derive": "time_only",
            "place_mode": "off",
            "object_mode": "off",
            "time_mode": "hard",
        },
        {
            "name": "place_interaction_to_scene_change",
            "step1": ["hard_pseudo_place", "hard_pseudo_interaction"],
            "step2": ["hard_pseudo_token"],
            "derive": "time+place",
            "place_mode": "hard",
            "object_mode": "off",
            "time_mode": "hard",
            "force_step2": "token=SCENE_CHANGE which=first top_k={top_k}",
        },
    ]

    out: list[NLQSample] = []
    budget_per_combo = max(1, int(max(1, n_chain) // max(1, len(combos))))
    rel = "after"
    window_s = 30.0
    for combo_cfg in combos:
        combo_name = str(combo_cfg.get("name", "chain_combo"))
        step1_types = [str(x) for x in combo_cfg.get("step1", [])]
        step2_types = [str(x) for x in combo_cfg.get("step2", [])]
        derive = str(combo_cfg.get("derive", "time_only"))
        place_mode = str(combo_cfg.get("place_mode", "soft"))
        object_mode = str(combo_cfg.get("object_mode", "soft"))
        time_mode = str(combo_cfg.get("time_mode", "hard"))
        force_step2 = str(combo_cfg.get("force_step2", "")).strip()
        force_step1 = str(combo_cfg.get("force_step1", "")).strip()
        step1_pool = [s for t in step1_types for s in by_type.get(t, [])]
        step2_pool = [s for t in step2_types for s in by_type.get(t, [])]
        if not step1_pool or not step2_pool:
            continue
        n_pick = min(int(budget_per_combo), len(step1_pool), len(step2_pool))
        idx1 = _sample_indices(len(step1_pool), n_pick, rng)
        idx2 = _sample_indices(len(step2_pool), n_pick, rng)
        for i in range(min(len(idx1), len(idx2))):
            s1 = step1_pool[idx1[i]]
            s2 = step2_pool[idx2[i]]
            step1_query = force_step1.format(top_k=int(top_k)) if force_step1 else str(s1.query).strip()
            step2_query = force_step2.format(top_k=int(top_k)) if force_step2 else str(s2.query).strip()
            query = (
                f"{step1_query} then {step2_query} "
                f"chain_rel={rel} chain_window_s={window_s:.1f} chain_top1_only=true "
                f"chain_derive={derive} chain_place_mode={place_mode} chain_object_mode={object_mode} chain_time_mode={time_mode}"
            )
            meta = {
                "source_kind": "chain",
                "chain_meta": {
                    "combo": str(combo_name),
                    "step1_type": str(s1.query_type),
                    "step2_type": str(s2.query_type),
                    "rel": str(rel),
                    "window_s": float(window_s),
                    "derive": str(derive),
                    "place_mode": str(place_mode),
                    "object_mode": str(object_mode),
                    "time_mode": str(time_mode),
                },
                "step1_meta": dict(s1.meta),
                "step2_meta": dict(s2.meta),
            }
            out.append(
                NLQSample(
                    qid=f"cnlq_{len(out) + 1:06d}",
                    query=query,
                    query_type="hard_pseudo_chain",
                    gt_span=(float(s2.gt_span[0]), float(s2.gt_span[1])),
                    top_k=int(top_k),
                    distractors=list(s2.distractors),
                    meta=meta,
                )
            )
            if len(out) >= int(n_chain):
                break
        if len(out) >= int(n_chain):
            break

    out.sort(key=lambda x: (float(x.gt_span[0]), float(x.gt_span[1]), str(x.qid)))
    for i, sample in enumerate(out, start=1):
        sample.qid = f"cnlq_{i:06d}"
    return out
