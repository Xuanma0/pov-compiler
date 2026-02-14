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
    ) -> list[RepoChunk]:
        raise NotImplementedError


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

