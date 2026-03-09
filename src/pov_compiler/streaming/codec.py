from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class MemoryItem:
    source_kind: str
    source_id: str
    t0: float
    t1: float
    score: float
    score_breakdown: dict[str, float]
    meta: dict[str, Any]


class StreamingCodec(Protocol):
    def encode_step(self, step_events: list[Any], step_meta: dict[str, Any] | None = None) -> list[MemoryItem]:
        ...


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _event_place_segment_id(event: Any) -> str:
    value = str(getattr(event, "place_segment_id", "") or "").strip()
    if value:
        return value
    return str(getattr(event, "meta", {}).get("place_segment_id", "")).strip()


def _event_interaction_score(event: Any) -> float:
    direct = getattr(event, "interaction_score", None)
    if isinstance(direct, (int, float)):
        return float(direct)
    sig = getattr(event, "interaction_signature", {}) if hasattr(event, "interaction_signature") else {}
    if isinstance(sig, dict):
        return _safe_float(sig.get("interaction_score", 0.0))
    return _safe_float(getattr(event, "meta", {}).get("interaction_score", 0.0))


def _event_boundary_conf(event: Any) -> float:
    scores = getattr(event, "scores", {})
    if isinstance(scores, dict):
        return _safe_float(scores.get("boundary_conf", 0.0))
    return 0.0


def _event_conf(event: Any) -> float:
    evidence = getattr(event, "evidence", [])
    if isinstance(evidence, list) and evidence:
        vals = []
        for item in evidence:
            vals.append(_safe_float(getattr(item, "conf", 0.0)))
        return float(sum(vals) / max(1, len(vals)))
    scores = getattr(event, "scores", {})
    if isinstance(scores, dict):
        return _safe_float(scores.get("boundary_conf", 0.0))
    return 0.0


class AllEventsStreamingCodec:
    name = "all_events"

    def encode_step(self, step_events: list[Any], step_meta: dict[str, Any] | None = None) -> list[MemoryItem]:
        out: list[MemoryItem] = []
        for event in step_events:
            out.append(
                MemoryItem(
                    source_kind="event_v1",
                    source_id=str(getattr(event, "id", "")),
                    t0=_safe_float(getattr(event, "t0", 0.0)),
                    t1=_safe_float(getattr(event, "t1", 0.0)),
                    score=1.0,
                    score_breakdown={"base": 1.0},
                    meta={"codec": self.name},
                )
            )
        return out


class FixedKStreamingCodec:
    name = "fixed_k"

    def __init__(
        self,
        *,
        k: int = 8,
        score_weights: dict[str, float] | None = None,
        diversity: bool = False,
    ):
        self.k = max(1, int(k))
        self.weights = {
            "interaction": 0.45,
            "boundary": 0.25,
            "conf": 0.20,
            "novelty": 0.10,
        }
        if isinstance(score_weights, dict):
            for key in ("interaction", "boundary", "conf", "novelty"):
                if key in score_weights:
                    self.weights[key] = _safe_float(score_weights.get(key, self.weights[key]), self.weights[key])
        self.diversity = bool(diversity)

    def _build_item(self, event: Any, step_meta: dict[str, Any] | None = None) -> MemoryItem:
        meta = dict(step_meta or {})
        seen_places = set(str(x) for x in meta.get("seen_place_segments", []) if str(x))
        place_id = _event_place_segment_id(event)
        novelty = 1.0 if place_id and place_id not in seen_places else 0.0
        interaction = max(0.0, min(1.0, _event_interaction_score(event)))
        boundary = max(0.0, min(1.0, _event_boundary_conf(event)))
        conf = max(0.0, min(1.0, _event_conf(event)))
        score_breakdown = {
            "interaction": interaction * self.weights["interaction"],
            "boundary": boundary * self.weights["boundary"],
            "conf": conf * self.weights["conf"],
            "novelty": novelty * self.weights["novelty"],
        }
        score = float(sum(score_breakdown.values()))
        return MemoryItem(
            source_kind="event_v1",
            source_id=str(getattr(event, "id", "")),
            t0=_safe_float(getattr(event, "t0", 0.0)),
            t1=_safe_float(getattr(event, "t1", 0.0)),
            score=score,
            score_breakdown=score_breakdown,
            meta={
                "codec": self.name,
                "place_segment_id": place_id,
                "interaction_score": interaction,
                "boundary_conf": boundary,
                "conf": conf,
            },
        )

    def encode_step(self, step_events: list[Any], step_meta: dict[str, Any] | None = None) -> list[MemoryItem]:
        items = [self._build_item(event, step_meta=step_meta) for event in list(step_events)]
        items.sort(key=lambda x: (-float(x.score), float(x.t0), float(x.t1), str(x.source_id)))
        if not items:
            return []
        if not self.diversity:
            return items[: self.k]

        # Optional diversity: prefer unique place segments first.
        selected: list[MemoryItem] = []
        used_places: set[str] = set()
        for item in items:
            place_id = str(item.meta.get("place_segment_id", "")).strip()
            if place_id and place_id in used_places:
                continue
            selected.append(item)
            if place_id:
                used_places.add(place_id)
            if len(selected) >= self.k:
                return selected
        for item in items:
            if len(selected) >= self.k:
                break
            if item in selected:
                continue
            selected.append(item)
        return selected[: self.k]


def resolve_codec_cfg(codec_cfg: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if codec_cfg is None:
        return {}
    if isinstance(codec_cfg, dict):
        return dict(codec_cfg)
    path = Path(codec_cfg)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text) or {}
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def build_streaming_codec(
    *,
    name: str,
    k: int,
    codec_cfg: dict[str, Any] | str | Path | None = None,
) -> StreamingCodec:
    codec_name = str(name or "all_events").strip().lower()
    cfg = resolve_codec_cfg(codec_cfg)
    if codec_name == "fixed_k":
        weights = cfg.get("score_weights", {}) if isinstance(cfg, dict) else {}
        diversity = bool(cfg.get("diversity", False)) if isinstance(cfg, dict) else False
        return FixedKStreamingCodec(k=max(1, int(k)), score_weights=weights if isinstance(weights, dict) else None, diversity=diversity)
    return AllEventsStreamingCodec()
