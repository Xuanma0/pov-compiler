from __future__ import annotations

import hashlib
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pov_compiler.repository.schema import RepoChunk
from pov_compiler.retrieval.query_parser import parse_query


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _norm_words(text: str) -> set[str]:
    raw = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return {x for x in raw.split() if x}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / max(1, union))


def _query_bonus(chunk: RepoChunk, query: str) -> float:
    parsed = parse_query(query)
    tags = {str(x).lower() for x in (chunk.tags or [])}
    text = str(chunk.text).lower()
    bonus = 0.0

    for anchor in parsed.anchor_types:
        a = str(anchor).lower()
        if a in tags or a in text:
            bonus += 0.25
    for token in parsed.token_types:
        t = str(token).lower()
        if t in tags or t in text:
            bonus += 0.2
    for decision in parsed.decision_types:
        d = str(decision).lower()
        if d in tags or d in text:
            bonus += 0.25
    if parsed.time_range is not None:
        q0, q1 = parsed.time_range
        if max(float(chunk.t0), float(q0)) <= min(float(chunk.t1), float(q1)):
            bonus += 0.15
    if parsed.interaction_object:
        obj = str(parsed.interaction_object).lower()
        if obj in text or obj in tags:
            bonus += 0.2
    return float(bonus)


def _estimate_tokens(text: str) -> int:
    return max(1, int(round(len(str(text)) / 4.0)))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return float(out)


def _time_overlap_ratio(chunk_t0: float, chunk_t1: float, hint_t0: float | None, hint_t1: float | None) -> float:
    t0 = float(chunk_t0)
    t1 = float(chunk_t1)
    if t1 < t0:
        t0, t1 = t1, t0
    if hint_t0 is None and hint_t1 is None:
        return 0.0
    h0 = float(hint_t0) if hint_t0 is not None else float("-inf")
    h1 = float(hint_t1) if hint_t1 is not None else float("inf")
    if h1 < h0:
        h0, h1 = h1, h0
    inter = max(0.0, min(t1, h1) - max(t0, h0))
    duration = max(1e-9, t1 - t0)
    return float(max(0.0, min(1.0, inter / duration)))


def _to_ms(value_s: float | None, *, default: float) -> float:
    if value_s is None:
        return float(default)
    return float(value_s) * 1000.0


def _chunk_overlap_with_hint_ms(
    chunk: RepoChunk,
    hint_t0_s: float | None,
    hint_t1_s: float | None,
) -> tuple[bool, float, float]:
    """Chain time filter (repo side) uses overlap semantics in milliseconds.

    Keep chunk iff:
    - chunk.t1_ms >= t_min_ms
    - chunk.t0_ms <= t_max_ms (if finite)
    """
    chunk_t0_ms = float(int(chunk.t0_ms) if int(chunk.t0_ms) != 0 else int(round(float(chunk.t0) * 1000.0)))
    chunk_t1_ms = float(int(chunk.t1_ms) if int(chunk.t1_ms) != 0 else int(round(float(chunk.t1) * 1000.0)))
    if chunk_t1_ms < chunk_t0_ms:
        chunk_t0_ms, chunk_t1_ms = chunk_t1_ms, chunk_t0_ms
    t_min_ms = _to_ms(hint_t0_s, default=float("-inf"))
    t_max_ms = _to_ms(hint_t1_s, default=float("inf"))
    keep = bool((chunk_t1_ms >= t_min_ms) and ((chunk_t0_ms <= t_max_ms) if t_max_ms != float("inf") else True))
    return keep, t_min_ms, t_max_ms


def _chunk_place_values(chunk: RepoChunk) -> set[str]:
    vals: set[str] = set()
    meta = dict(chunk.meta or {})
    for key in ("place_segment_id", "place_id"):
        value = str(meta.get(key, "")).strip().lower()
        if value:
            vals.add(value)
    for tag in chunk.tags or []:
        text = str(tag).strip().lower()
        if text.startswith("place:"):
            value = text.split(":", 1)[1].strip()
            if value:
                vals.add(value)
        elif text in {"first", "last", "any"}:
            vals.add(text)
    words = _norm_words(chunk.text)
    for token in ("first", "last", "any"):
        if token in words:
            vals.add(token)
    return vals


def _chunk_object_values(chunk: RepoChunk) -> set[str]:
    vals: set[str] = set()
    meta = dict(chunk.meta or {})
    for key in ("primary_object", "interaction_object", "interaction_primary_object", "object_name"):
        value = str(meta.get(key, "")).strip().lower()
        if value:
            vals.add(value)
    for tag in chunk.tags or []:
        text = str(tag).strip().lower()
        if text.startswith("obj:"):
            value = text.split(":", 1)[1].strip()
            if value:
                vals.add(value)
    return vals


@dataclass
class PolicyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def stable_hash(self) -> str:
        return _stable_hash({"name": self.name, "params": self.params})


class WritePolicy(ABC):
    name: str
    params: dict[str, Any]

    def __init__(self, name: str, **params: Any) -> None:
        self.name = str(name)
        self.params = dict(params)

    def to_config(self) -> PolicyConfig:
        return PolicyConfig(name=self.name, params=dict(self.params))

    def stable_hash(self) -> str:
        return self.to_config().stable_hash()

    @abstractmethod
    def write(
        self,
        chunks_in: list[RepoChunk],
        signals: dict[str, Any] | None = None,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
    ) -> list[RepoChunk]:
        raise NotImplementedError


class ReadPolicy(ABC):
    name: str
    params: dict[str, Any]

    def __init__(self, name: str, **params: Any) -> None:
        self.name = str(name)
        self.params = dict(params)

    def to_config(self) -> PolicyConfig:
        return PolicyConfig(name=self.name, params=dict(self.params))

    def stable_hash(self) -> str:
        return self.to_config().stable_hash()

    @abstractmethod
    def select(
        self,
        chunks: list[RepoChunk],
        query: str,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
        query_info: dict[str, Any] | None = None,
        query_hints: dict[str, Any] | None = None,
    ) -> list[RepoChunk]:
        raise NotImplementedError

    def select_with_trace(
        self,
        chunks: list[RepoChunk],
        query: str,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
        query_info: dict[str, Any] | None = None,
        query_hints: dict[str, Any] | None = None,
    ) -> tuple[list[RepoChunk], dict[str, Any]]:
        selected = self.select(
            chunks=chunks,
            query=query,
            budget_cfg=budget_cfg,
            rng=rng,
            query_info=query_info,
            query_hints=query_hints,
        )
        return selected, {
            "policy_name": self.name,
            "policy_hash": self.stable_hash(),
            "selected_chunk_ids": [str(c.id) for c in selected],
            "selected_breakdown_by_level": {},
            "per_chunk_score_fields": {},
            "dropped_topN": [],
        }


class FixedIntervalWritePolicy(WritePolicy):
    def __init__(self, chunk_step_s: float = 8.0, keep_levels: list[str] | None = None) -> None:
        super().__init__("fixed_interval", chunk_step_s=float(chunk_step_s), keep_levels=list(keep_levels or []))
        self.chunk_step_s = float(chunk_step_s)
        self.keep_levels = {str(x).lower() for x in (keep_levels or [])}

    def write(
        self,
        chunks_in: list[RepoChunk],
        signals: dict[str, Any] | None = None,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
    ) -> list[RepoChunk]:
        ordered = sorted(chunks_in, key=lambda c: (float(c.t0), float(c.t1), str(c.level), str(c.id)))
        if self.chunk_step_s <= 0:
            return ordered
        out: list[RepoChunk] = []
        next_t = -1e9
        for chunk in ordered:
            level = str(chunk.level or chunk.scale).lower()
            if level in self.keep_levels:
                out.append(chunk)
                continue
            if float(chunk.t0) >= next_t:
                out.append(chunk)
                next_t = float(chunk.t0) + self.chunk_step_s
        out.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.level), str(c.id)))
        return out


class EventTriggeredWritePolicy(WritePolicy):
    def __init__(self, cooldown_s: float = 4.0, decision_tag: str = "decision", interaction_tag: str = "interaction") -> None:
        super().__init__(
            "event_triggered",
            cooldown_s=float(cooldown_s),
            decision_tag=str(decision_tag),
            interaction_tag=str(interaction_tag),
        )
        self.cooldown_s = float(cooldown_s)
        self.decision_tag = str(decision_tag).lower()
        self.interaction_tag = str(interaction_tag).lower()

    def write(
        self,
        chunks_in: list[RepoChunk],
        signals: dict[str, Any] | None = None,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
    ) -> list[RepoChunk]:
        ordered = sorted(chunks_in, key=lambda c: (float(c.t0), float(c.t1), -float(c.importance), str(c.id)))
        out: list[RepoChunk] = []
        last_t = -1e9
        for chunk in ordered:
            tags = {str(x).lower() for x in (chunk.tags or [])}
            level = str(chunk.level or chunk.scale).lower()
            force = (
                level == "decision"
                or self.decision_tag in tags
                or self.interaction_tag in tags
                or "interaction-heavy" in tags
                or float(chunk.score_fields.get("interaction_score", 0.0)) > 0.35
            )
            if force:
                if float(chunk.t0) - last_t >= self.cooldown_s:
                    out.append(chunk)
                    last_t = float(chunk.t0)
                continue
            # fallback periodic write to avoid sparse memory
            if float(chunk.t0) - last_t >= max(1.0, self.cooldown_s * 1.5):
                out.append(chunk)
                last_t = float(chunk.t0)
        out.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.level), str(c.id)))
        return out


class NoveltyWritePolicy(WritePolicy):
    def __init__(self, novelty_threshold: float = 0.35, max_reference: int = 8, force_levels: list[str] | None = None) -> None:
        super().__init__(
            "novelty",
            novelty_threshold=float(novelty_threshold),
            max_reference=int(max_reference),
            force_levels=list(force_levels or ["decision"]),
        )
        self.novelty_threshold = float(novelty_threshold)
        self.max_reference = int(max_reference)
        self.force_levels = {str(x).lower() for x in (force_levels or ["decision"])}

    def write(
        self,
        chunks_in: list[RepoChunk],
        signals: dict[str, Any] | None = None,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
    ) -> list[RepoChunk]:
        ordered = sorted(chunks_in, key=lambda c: (float(c.t0), float(c.t1), -float(c.importance), str(c.id)))
        out: list[RepoChunk] = []
        refs: list[set[str]] = []
        for chunk in ordered:
            level = str(chunk.level or chunk.scale).lower()
            words = _norm_words(chunk.text) | {str(x).lower() for x in (chunk.tags or [])}
            if level in self.force_levels:
                out.append(chunk)
                refs.append(words)
                refs = refs[-self.max_reference :]
                continue
            best_sim = 0.0
            for ref in refs[-self.max_reference :]:
                best_sim = max(best_sim, _jaccard(words, ref))
            novelty = 1.0 - best_sim
            if novelty >= self.novelty_threshold:
                out.append(chunk)
                refs.append(words)
                refs = refs[-self.max_reference :]
        out.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.level), str(c.id)))
        return out


class BudgetedTopKReadPolicy(ReadPolicy):
    def __init__(self, max_chunks: int = 16, max_tokens: int = 200, max_seconds: float | None = None) -> None:
        super().__init__(
            "budgeted_topk",
            max_chunks=int(max_chunks),
            max_tokens=int(max_tokens),
            max_seconds=max_seconds,
        )
        self.max_chunks = int(max_chunks)
        self.max_tokens = int(max_tokens)
        self.max_seconds = None if max_seconds is None else float(max_seconds)

    def _effective_budget(self, budget_cfg: dict[str, Any] | None = None) -> tuple[int, int, float | None]:
        budget = dict(budget_cfg or {})
        max_chunks = int(budget.get("max_repo_chunks", budget.get("max_chunks", self.max_chunks)))
        max_tokens = int(budget.get("max_tokens", budget.get("max_repo_tokens", self.max_tokens)))
        max_seconds_raw = budget.get("max_seconds", self.max_seconds)
        max_seconds = None if max_seconds_raw in (None, "", "none") else float(max_seconds_raw)
        return max(1, max_chunks), max(1, max_tokens), max_seconds

    def select(
        self,
        chunks: list[RepoChunk],
        query: str,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
        query_info: dict[str, Any] | None = None,
        query_hints: dict[str, Any] | None = None,
    ) -> list[RepoChunk]:
        max_chunks, max_tokens, max_seconds = self._effective_budget(budget_cfg)
        ranked = sorted(
            chunks,
            key=lambda c: (float(c.importance) + _query_bonus(c, query), float(c.t1), str(c.id)),
            reverse=True,
        )
        chosen: list[RepoChunk] = []
        used_tokens = 0
        for chunk in ranked:
            if len(chosen) >= max_chunks:
                break
            if max_seconds is not None and float(chunk.t0) > max_seconds:
                continue
            token_est = int(chunk.meta.get("token_est", _estimate_tokens(chunk.text)))
            if used_tokens + token_est > max_tokens:
                continue
            chosen.append(chunk)
            used_tokens += token_est
        chosen.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
        return chosen


class DiverseReadPolicy(BudgetedTopKReadPolicy):
    def __init__(
        self,
        max_chunks: int = 16,
        max_tokens: int = 200,
        max_seconds: float | None = None,
        diversity_threshold: float = 0.88,
    ) -> None:
        super().__init__(max_chunks=max_chunks, max_tokens=max_tokens, max_seconds=max_seconds)
        self.name = "diverse"
        self.params["name"] = self.name
        self.params["diversity_threshold"] = float(diversity_threshold)
        self.diversity_threshold = float(diversity_threshold)

    def select(
        self,
        chunks: list[RepoChunk],
        query: str,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
        query_info: dict[str, Any] | None = None,
        query_hints: dict[str, Any] | None = None,
    ) -> list[RepoChunk]:
        max_chunks, max_tokens, max_seconds = self._effective_budget(budget_cfg)
        ranked = sorted(
            chunks,
            key=lambda c: (float(c.importance) + _query_bonus(c, query), float(c.t1), str(c.id)),
            reverse=True,
        )
        chosen: list[RepoChunk] = []
        chosen_words: list[set[str]] = []
        used_tokens = 0
        for chunk in ranked:
            if len(chosen) >= max_chunks:
                break
            if max_seconds is not None and float(chunk.t0) > max_seconds:
                continue
            token_est = int(chunk.meta.get("token_est", _estimate_tokens(chunk.text)))
            if used_tokens + token_est > max_tokens:
                continue
            words = _norm_words(chunk.text) | {str(x).lower() for x in (chunk.tags or [])}
            if chosen_words:
                max_sim = max(_jaccard(words, w) for w in chosen_words)
                if max_sim >= self.diversity_threshold:
                    continue
            chosen.append(chunk)
            chosen_words.append(words)
            used_tokens += token_est
        chosen.sort(key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
        return chosen


class QueryAwareReadPolicyV0(BudgetedTopKReadPolicy):
    def __init__(
        self,
        *,
        max_chunks: int = 16,
        max_tokens: int = 200,
        max_seconds: float | None = None,
        level_priors: dict[str, float] | None = None,
        intent_level_boost: dict[str, dict[str, float]] | None = None,
        constraint_level_boost: dict[str, dict[str, float]] | None = None,
        recency_weight: float = 0.1,
        redundancy_penalty: float = 0.2,
        hint_weight: float = 0.35,
        max_chunks_per_level: dict[str, int] | None = None,
        dedup_sim_threshold: float = 0.92,
        dropped_topn: int = 8,
    ) -> None:
        super().__init__(max_chunks=max_chunks, max_tokens=max_tokens, max_seconds=max_seconds)
        self.name = "query_aware"
        self.level_priors = {str(k).lower(): float(v) for k, v in (level_priors or {}).items()}
        if not self.level_priors:
            self.level_priors = {"event": 0.25, "decision": 0.45, "place": 0.35, "segment": 0.2, "window": 0.15}
        self.intent_level_boost = {
            str(intent).lower(): {str(k).lower(): float(v) for k, v in dict(levels).items()}
            for intent, levels in dict(intent_level_boost or {}).items()
        }
        if not self.intent_level_boost:
            self.intent_level_boost = {
                "event": {"event": 0.3, "segment": 0.12},
                "decision": {"decision": 0.45, "event": 0.15},
                "token": {"segment": 0.2, "window": 0.2, "event": 0.1},
                "anchor": {"place": 0.25, "event": 0.2, "decision": 0.15},
                "mixed": {"event": 0.12, "decision": 0.12, "place": 0.12},
            }
        self.constraint_level_boost = {
            str(key).lower(): {str(k).lower(): float(v) for k, v in dict(levels).items()}
            for key, levels in dict(constraint_level_boost or {}).items()
        }
        if not self.constraint_level_boost:
            self.constraint_level_boost = {
                "place": {"place": 0.35, "event": 0.2},
                "place_segment_id": {"place": 0.4, "event": 0.15},
                "interaction_min": {"place": 0.25, "decision": 0.2, "event": 0.15},
                "interaction_object": {"place": 0.3, "decision": 0.2, "event": 0.1},
                "anchor_type": {"decision": 0.2, "event": 0.2},
                "decision_type": {"decision": 0.3},
                "token_type": {"segment": 0.2, "window": 0.15, "event": 0.1},
                "time_range": {"window": 0.15, "segment": 0.15, "event": 0.1},
            }
        self.recency_weight = float(recency_weight)
        self.redundancy_penalty = float(redundancy_penalty)
        self.hint_weight = float(hint_weight)
        self.max_chunks_per_level = {str(k).lower(): int(v) for k, v in dict(max_chunks_per_level or {}).items()}
        self.dedup_sim_threshold = float(dedup_sim_threshold)
        self.dropped_topn = int(max(1, dropped_topn))
        self.params.update(
            {
                "name": self.name,
                "level_priors": self.level_priors,
                "intent_level_boost": self.intent_level_boost,
                "constraint_level_boost": self.constraint_level_boost,
                "recency_weight": self.recency_weight,
                "redundancy_penalty": self.redundancy_penalty,
                "hint_weight": self.hint_weight,
                "max_chunks_per_level": self.max_chunks_per_level,
                "dedup_sim_threshold": self.dedup_sim_threshold,
                "dropped_topn": self.dropped_topn,
            }
        )

    @staticmethod
    def _normalize_query_hints(query_hints: dict[str, Any] | None) -> dict[str, Any]:
        raw = dict(query_hints or {})
        if isinstance(raw.get("derived_constraints"), dict):
            dc = dict(raw.get("derived_constraints", {}))
        else:
            dc = {}
        time_hint = dict(dc.get("time", {})) if isinstance(dc.get("time", {}), dict) else {}
        place_hint = dict(dc.get("place", {})) if isinstance(dc.get("place", {}), dict) else {}
        object_hint = dict(dc.get("object", {})) if isinstance(dc.get("object", {}), dict) else {}

        def _norm_mode(value: Any, default: str = "soft") -> str:
            mode = str(value or default).strip().lower()
            if mode not in {"hard", "soft", "off"}:
                mode = default
            return mode

        def _to_float(value: Any) -> float | None:
            try:
                out = float(value)
            except Exception:
                return None
            if out != out:
                return None
            return float(out)

        t_min = _to_float(time_hint.get("t_min_s", raw.get("chain_time_min_s", None)))
        t_max = _to_float(time_hint.get("t_max_s", raw.get("chain_time_max_s", None)))
        time_mode = _norm_mode(time_hint.get("mode", raw.get("chain_time_mode", "hard")), default="hard")
        time_enabled = bool(time_hint.get("enabled", time_mode != "off" and (t_min is not None or t_max is not None)))

        place_value = str(place_hint.get("value", raw.get("chain_place_value", ""))).strip().lower()
        place_mode = _norm_mode(place_hint.get("mode", raw.get("chain_place_mode", "soft")))
        place_enabled = bool(place_hint.get("enabled", place_mode != "off" and bool(place_value)))

        object_value = str(object_hint.get("value", raw.get("chain_object_value", ""))).strip().lower()
        object_mode = _norm_mode(object_hint.get("mode", raw.get("chain_object_mode", "soft")))
        object_enabled = bool(object_hint.get("enabled", object_mode != "off" and bool(object_value)))

        return {
            "derived_constraints": {
                "time": {
                    "enabled": bool(time_enabled),
                    "mode": str(time_mode),
                    "t_min_s": t_min,
                    "t_max_s": t_max,
                    "source": str(time_hint.get("source", raw.get("chain_time_source", ""))),
                    "disabled_reason": str(time_hint.get("disabled_reason", "")),
                },
                "place": {
                    "enabled": bool(place_enabled),
                    "mode": str(place_mode),
                    "value": str(place_value),
                    "source": str(place_hint.get("source", raw.get("chain_place_source", ""))),
                    "disabled_reason": str(place_hint.get("disabled_reason", "")),
                },
                "object": {
                    "enabled": bool(object_enabled),
                    "mode": str(object_mode),
                    "value": str(object_value),
                    "source": str(object_hint.get("source", raw.get("chain_object_source", ""))),
                    "disabled_reason": str(object_hint.get("disabled_reason", "")),
                },
            },
            "chain_meta": dict(raw.get("chain_meta", {})) if isinstance(raw.get("chain_meta", {}), dict) else {},
        }

    @classmethod
    def _safe_query_info(
        cls,
        query: str,
        query_info: dict[str, Any] | None,
        query_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = dict(query_info or {})
        parsed_constraints = out.get("parsed_constraints")
        if not isinstance(parsed_constraints, dict):
            parsed = parse_query(query)
            parsed_constraints = {
                "place": parsed.place,
                "place_segment_id": list(parsed.place_segment_ids),
                "interaction_min": parsed.interaction_min,
                "interaction_object": parsed.interaction_object,
                "anchor_type": list(parsed.anchor_types),
                "decision_type": list(parsed.decision_types),
                "token_type": list(parsed.token_types),
                "time_range": parsed.time_range,
                "chain_time_min_s": parsed.chain_time_min_s,
                "chain_time_max_s": parsed.chain_time_max_s,
                "chain_time_mode": parsed.chain_time_mode,
                "chain_place_value": parsed.chain_place_value,
                "chain_place_mode": parsed.chain_place_mode,
                "chain_object_value": parsed.chain_object_value,
                "chain_object_mode": parsed.chain_object_mode,
            }
        cleaned_constraints: dict[str, Any] = {}
        for key, value in dict(parsed_constraints).items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list) and not value:
                continue
            cleaned_constraints[str(key).lower()] = value
        out["parsed_constraints"] = cleaned_constraints
        out["plan_intent"] = str(out.get("plan_intent", "mixed")).lower()
        if out["plan_intent"] not in {"event", "token", "decision", "anchor", "mixed"}:
            out["plan_intent"] = "mixed"
        try:
            out["top_k"] = int(out.get("top_k", 6))
        except Exception:
            out["top_k"] = 6
        out["query_hints"] = cls._normalize_query_hints(query_hints)
        return out

    def _score_chunks(
        self,
        *,
        chunks: list[RepoChunk],
        query: str,
        budget_cfg: dict[str, Any] | None,
        query_info: dict[str, Any],
    ) -> tuple[list[tuple[RepoChunk, dict[str, float]]], dict[str, dict[str, float]]]:
        t1_max = max((float(c.t1) for c in chunks), default=1.0)
        intent = str(query_info.get("plan_intent", "mixed")).lower()
        constraints = dict(query_info.get("parsed_constraints", {}))
        query_hints = dict(query_info.get("query_hints", {}))
        derived = dict(query_hints.get("derived_constraints", {})) if isinstance(query_hints.get("derived_constraints", {}), dict) else {}
        time_hint = dict(derived.get("time", {})) if isinstance(derived.get("time", {}), dict) else {}
        place_hint = dict(derived.get("place", {})) if isinstance(derived.get("place", {}), dict) else {}
        object_hint = dict(derived.get("object", {})) if isinstance(derived.get("object", {}), dict) else {}
        hint_time_enabled = bool(time_hint.get("enabled", False))
        hint_time_min = time_hint.get("t_min_s")
        hint_time_max = time_hint.get("t_max_s")
        hint_place_enabled = bool(place_hint.get("enabled", False))
        hint_place_value = str(place_hint.get("value", "")).strip().lower()
        hint_object_enabled = bool(object_hint.get("enabled", False))
        hint_object_value = str(object_hint.get("value", "")).strip().lower()
        per_chunk: dict[str, dict[str, float]] = {}
        scored_rows: list[tuple[RepoChunk, dict[str, float]]] = []
        for chunk in chunks:
            level = str(chunk.level or chunk.scale).lower()
            base_score = float(chunk.importance) + _query_bonus(chunk, query)
            level_boost = float(self.level_priors.get(level, 0.0))
            intent_boost = float(self.intent_level_boost.get(intent, {}).get(level, 0.0))
            constraint_boost = 0.0
            for ckey in constraints.keys():
                constraint_boost += float(self.constraint_level_boost.get(str(ckey).lower(), {}).get(level, 0.0))
            recency = float(self.recency_weight) * float(max(0.0, min(1.0, float(chunk.t1) / max(1e-9, t1_max))))
            hint_time = _time_overlap_ratio(float(chunk.t0), float(chunk.t1), hint_time_min, hint_time_max) if hint_time_enabled else 0.0
            place_values = _chunk_place_values(chunk)
            hint_place = 0.0
            if hint_place_enabled and hint_place_value:
                if hint_place_value in {"first", "last", "any"}:
                    hint_place = 1.0 if hint_place_value in place_values else 0.0
                else:
                    hint_place = 1.0 if hint_place_value in place_values else 0.0
            object_values = _chunk_object_values(chunk)
            hint_object = 0.0
            if hint_object_enabled and hint_object_value:
                hint_object = 1.0 if any((hint_object_value in ov or ov in hint_object_value) for ov in object_values) else 0.0
            hint_score = (0.5 * hint_time) + (0.25 * hint_place) + (0.25 * hint_object)
            hint_bonus = float(self.hint_weight) * float(hint_score)
            final = base_score + level_boost + intent_boost + constraint_boost + recency + hint_bonus
            fields = {
                "base_score": float(base_score),
                "level_boost": float(level_boost),
                "intent_boost": float(intent_boost),
                "constraint_boost": float(constraint_boost),
                "recency_boost": float(recency),
                "hint_time": float(hint_time),
                "hint_place": float(hint_place),
                "hint_object": float(hint_object),
                "hint_score": float(hint_score),
                "w_hint": float(self.hint_weight),
                "dedup_penalty": 0.0,
                "final_score": float(final),
            }
            per_chunk[str(chunk.id)] = fields
            scored_rows.append((chunk, fields))
        scored_rows.sort(
            key=lambda item: (
                -float(item[1]["final_score"]),
                -float(item[0].importance),
                float(item[0].t0),
                str(item[0].id),
            )
        )
        return scored_rows, per_chunk

    def select_with_trace(
        self,
        chunks: list[RepoChunk],
        query: str,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
        query_info: dict[str, Any] | None = None,
        query_hints: dict[str, Any] | None = None,
    ) -> tuple[list[RepoChunk], dict[str, Any]]:
        budget = dict(budget_cfg or {})
        max_chunks, max_tokens, max_seconds = self._effective_budget(budget)
        info = self._safe_query_info(query, query_info, query_hints=query_hints)
        ranked, per_chunk = self._score_chunks(chunks=list(chunks), query=query, budget_cfg=budget, query_info=info)
        derived = (
            dict(info.get("query_hints", {}).get("derived_constraints", {}))
            if isinstance(info.get("query_hints", {}).get("derived_constraints", {}), dict)
            else {}
        )
        time_hint = dict(derived.get("time", {})) if isinstance(derived.get("time", {}), dict) else {}
        place_hint = dict(derived.get("place", {})) if isinstance(derived.get("place", {}), dict) else {}
        object_hint = dict(derived.get("object", {})) if isinstance(derived.get("object", {}), dict) else {}
        hint_filter_before = int(len(ranked))
        hint_filtered_reason_counts: dict[str, int] = {}
        repo_time_filter_mode = "overlap"
        hint_time_window_ms: dict[str, float | None] = {"t_min_ms": None, "t_max_ms": None}
        ranked_after_hints: list[tuple[RepoChunk, dict[str, float]]] = []
        for chunk, fields in ranked:
            reason = ""
            if bool(time_hint.get("enabled", False)) and str(time_hint.get("mode", "soft")).strip().lower() == "hard":
                t_min = time_hint.get("t_min_s")
                t_max = time_hint.get("t_max_s")
                keep_by_overlap, t_min_ms, t_max_ms = _chunk_overlap_with_hint_ms(chunk, _to_float(t_min), _to_float(t_max))
                hint_time_window_ms["t_min_ms"] = float(t_min_ms)
                hint_time_window_ms["t_max_ms"] = None if t_max_ms == float("inf") else float(t_max_ms)
                if not keep_by_overlap:
                    reason = "filtered_by_chain_time"
            if not reason and bool(place_hint.get("enabled", False)) and str(place_hint.get("mode", "soft")).strip().lower() == "hard":
                place_value = str(place_hint.get("value", "")).strip().lower()
                if place_value:
                    if place_value in {"first", "last", "any"}:
                        has_place = place_value in _chunk_place_values(chunk)
                    else:
                        has_place = place_value in _chunk_place_values(chunk)
                    if not has_place:
                        reason = "filtered_by_chain_place"
            if not reason and bool(object_hint.get("enabled", False)) and str(object_hint.get("mode", "soft")).strip().lower() == "hard":
                object_value = str(object_hint.get("value", "")).strip().lower()
                if object_value:
                    object_values = _chunk_object_values(chunk)
                    matched = any((object_value in v or v in object_value) for v in object_values)
                    if not matched:
                        reason = "filtered_by_chain_object"
            if reason:
                hint_filtered_reason_counts[reason] = hint_filtered_reason_counts.get(reason, 0) + 1
                if "drop_reason" not in fields:
                    fields["drop_reason"] = 0.0
                continue
            ranked_after_hints.append((chunk, fields))
        ranked = ranked_after_hints

        selected: list[RepoChunk] = []
        selected_words: list[set[str]] = []
        selected_ids: list[str] = []
        by_level: dict[str, int] = {}
        dropped: list[dict[str, Any]] = []
        used_tokens = 0
        level_caps = dict(self.max_chunks_per_level)
        if "max_chunks_per_level" in budget and isinstance(budget.get("max_chunks_per_level"), dict):
            for key, value in dict(budget.get("max_chunks_per_level", {})).items():
                try:
                    level_caps[str(key).lower()] = int(value)
                except Exception:
                    continue

        for chunk, fields in ranked:
            cid = str(chunk.id)
            level = str(chunk.level or chunk.scale).lower()
            reason = ""
            token_est = int(chunk.meta.get("token_est", _estimate_tokens(chunk.text)))
            if max_seconds is not None and float(chunk.t0) > float(max_seconds):
                reason = "budget_exceeded"
            elif len(selected) >= max_chunks:
                reason = "budget_exceeded"
            elif used_tokens + token_est > max_tokens:
                reason = "budget_exceeded"
            elif level in level_caps and by_level.get(level, 0) >= int(level_caps[level]):
                reason = "level_cap"
            else:
                words = _norm_words(chunk.text) | {str(x).lower() for x in (chunk.tags or [])}
                if selected_words:
                    max_sim = max(_jaccard(words, w) for w in selected_words)
                    if max_sim >= float(self.dedup_sim_threshold):
                        reason = "dedup"
                        penalty = float(self.redundancy_penalty) * float(max_sim)
                        fields["dedup_penalty"] = -penalty
                        fields["final_score"] = float(fields["final_score"]) - penalty
                if not reason:
                    selected.append(chunk)
                    selected_words.append(words)
                    selected_ids.append(cid)
                    by_level[level] = by_level.get(level, 0) + 1
                    used_tokens += token_est

            if reason and len(dropped) < int(self.dropped_topn):
                dropped.append(
                    {
                        "chunk_id": cid,
                        "level": level,
                        "reason": reason,
                        "final_score": float(fields.get("final_score", 0.0)),
                    }
                )

        selected_sorted = sorted(selected, key=lambda c: (float(c.t0), float(c.t1), str(c.id)))
        trace = {
            "policy_name": self.name,
            "policy_hash": self.stable_hash(),
            "selected_chunk_ids": selected_ids,
            "selected_breakdown_by_level": dict(sorted(by_level.items())),
            "per_chunk_score_fields": per_chunk,
            "dropped_topN": dropped,
            "budget_used": {
                "max_chunks": int(max_chunks),
                "max_tokens": int(max_tokens),
                "max_seconds": None if max_seconds is None else float(max_seconds),
            },
            "query_info": {
                "plan_intent": str(info.get("plan_intent", "mixed")),
                "parsed_constraints": dict(info.get("parsed_constraints", {})),
                "top_k": int(info.get("top_k", 6)),
            },
            "query_hints": dict(info.get("query_hints", {})),
            "hint_filter_before": int(hint_filter_before),
            "hint_filter_after": int(len(ranked)),
            "hint_filtered_reason_counts": dict(sorted(hint_filtered_reason_counts.items())),
            "repo_time_filter_mode": str(repo_time_filter_mode),
            "repo_time_window_ms": dict(hint_time_window_ms),
        }
        return selected_sorted, trace

    def select(
        self,
        chunks: list[RepoChunk],
        query: str,
        budget_cfg: dict[str, Any] | None = None,
        rng: random.Random | None = None,
        query_info: dict[str, Any] | None = None,
        query_hints: dict[str, Any] | None = None,
    ) -> list[RepoChunk]:
        selected, _ = self.select_with_trace(
            chunks=chunks,
            query=query,
            budget_cfg=budget_cfg,
            rng=rng,
            query_info=query_info,
            query_hints=query_hints,
        )
        return selected


def build_write_policy(cfg: dict[str, Any] | None = None) -> WritePolicy:
    payload = dict(cfg or {})
    name = str(payload.get("name", payload.get("write_policy", "fixed_interval"))).strip().lower()
    if name in {"fixed", "fixed_interval"}:
        return FixedIntervalWritePolicy(
            chunk_step_s=float(payload.get("chunk_step_s", payload.get("step_s", 8.0))),
            keep_levels=list(payload.get("keep_levels", payload.get("force_levels", [])) or []),
        )
    if name in {"event_triggered", "triggered"}:
        return EventTriggeredWritePolicy(
            cooldown_s=float(payload.get("cooldown_s", 4.0)),
            decision_tag=str(payload.get("decision_tag", "decision")),
            interaction_tag=str(payload.get("interaction_tag", "interaction")),
        )
    if name in {"novelty", "novelty_write"}:
        return NoveltyWritePolicy(
            novelty_threshold=float(payload.get("novelty_threshold", 0.35)),
            max_reference=int(payload.get("max_reference", 8)),
            force_levels=list(payload.get("force_levels", ["decision"])),
        )
    raise ValueError(f"unsupported write policy: {name}")


def build_read_policy(cfg: dict[str, Any] | None = None) -> ReadPolicy:
    payload = dict(cfg or {})
    name = str(payload.get("name", payload.get("read_policy", payload.get("strategy", "budgeted_topk")))).strip().lower()
    if name in {"importance_greedy", "budgeted_topk", "topk"}:
        return BudgetedTopKReadPolicy(
            max_chunks=int(payload.get("max_chunks", payload.get("max_repo_chunks", 16))),
            max_tokens=int(payload.get("max_tokens", payload.get("max_repo_tokens", 200))),
            max_seconds=payload.get("max_seconds", None),
        )
    if name in {"diverse", "diverse_read", "mmr"}:
        return DiverseReadPolicy(
            max_chunks=int(payload.get("max_chunks", payload.get("max_repo_chunks", 16))),
            max_tokens=int(payload.get("max_tokens", payload.get("max_repo_tokens", 200))),
            max_seconds=payload.get("max_seconds", None),
            diversity_threshold=float(payload.get("diversity_threshold", payload.get("dedup_threshold", 0.88))),
        )
    if name in {"query_aware", "query_aware_v0", "queryaware"}:
        return QueryAwareReadPolicyV0(
            max_chunks=int(payload.get("max_chunks", payload.get("max_repo_chunks", 16))),
            max_tokens=int(payload.get("max_tokens", payload.get("max_repo_tokens", 200))),
            max_seconds=payload.get("max_seconds", None),
            level_priors=dict(payload.get("level_priors", {})),
            intent_level_boost=dict(payload.get("intent_level_boost", {})),
            constraint_level_boost=dict(payload.get("constraint_level_boost", {})),
            recency_weight=float(payload.get("recency_weight", 0.1)),
            redundancy_penalty=float(payload.get("redundancy_penalty", 0.2)),
            hint_weight=float(payload.get("hint_weight", 0.35)),
            max_chunks_per_level=dict(payload.get("max_chunks_per_level", {})),
            dedup_sim_threshold=float(payload.get("dedup_sim_threshold", 0.92)),
            dropped_topn=int(payload.get("dropped_topn", 8)),
        )
    if name in {"recency_greedy", "recency"}:
        # Backward-compat alias; implemented as budgeted top-k over recency by assigning importance externally.
        return BudgetedTopKReadPolicy(
            max_chunks=int(payload.get("max_chunks", payload.get("max_repo_chunks", 16))),
            max_tokens=int(payload.get("max_tokens", payload.get("max_repo_tokens", 200))),
            max_seconds=payload.get("max_seconds", None),
        )
    raise ValueError(f"unsupported read policy: {name}")


def load_policy_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"policy config must be object: {p}")
        return payload
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"policy config must be mapping: {p}")
        return payload
    except ImportError as exc:
        raise RuntimeError(
            "pyyaml is required to load yaml policy files. Install with `pip install pyyaml` "
            "or pass a .json config."
        ) from exc


def dump_policy_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    try:
        import yaml  # type: ignore

        p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    except ImportError as exc:
        raise RuntimeError(
            "pyyaml is required to write yaml policy files. Install with `pip install pyyaml` "
            "or use a .json output path."
        ) from exc


def policy_cfg_hash(payload: dict[str, Any]) -> str:
    return _stable_hash(dict(payload))
