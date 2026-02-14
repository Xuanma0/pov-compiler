from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Anchor(BaseModel):
    type: str
    t: float
    conf: float
    meta: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    id: str
    t0: float
    t1: float
    scores: dict[str, float] = Field(default_factory=dict)
    anchors: list[Anchor] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class KeyClip(BaseModel):
    id: str
    t0: float
    t1: float
    source_event: str
    anchor_type: str
    anchor_t: float
    conf: float
    meta: dict[str, Any] = Field(default_factory=dict)


class Token(BaseModel):
    id: str
    t0: float
    t1: float
    type: str
    conf: float
    source_event: str
    source: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class TokenCodec(BaseModel):
    version: str = "0.2"
    vocab: list[str] = Field(default_factory=list)
    tokens: list[Token] = Field(default_factory=list)


class Alternative(BaseModel):
    action_type: str
    rationale: str
    expected_outcome: str
    risk_delta: float | None = None
    conf: float = 0.5
    meta: dict[str, Any] = Field(default_factory=dict)


class DecisionPoint(BaseModel):
    id: str
    t: float
    t0: float
    t1: float
    source_event: str
    source_highlight: str | None = None
    trigger: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    action: dict[str, Any] = Field(default_factory=dict)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    outcome: dict[str, Any] = Field(default_factory=dict)
    alternatives: list[Alternative] = Field(default_factory=list)
    conf: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    base_score: float = 0.0
    semantic_score: float = 0.0
    decision_align_score: float = 0.0
    intent_bonus: float = 0.0
    match_score: float = 0.0
    first_last_bonus: float = 0.0
    scene_penalty: float = 0.0
    distractor_penalty: float = 0.0
    conf_bonus: float = 0.0
    boundary_bonus: float = 0.0
    priority_bonus: float = 0.0
    trigger_match: float = 0.0
    action_match: float = 0.0
    constraint_match: float = 0.0
    outcome_match: float = 0.0
    evidence_quality: float = 0.0
    total: float = 0.0


class ConstraintTrace(BaseModel):
    source_query: str = ""
    chosen_plan_intent: str = ""
    applied_constraints: list[str] = Field(default_factory=list)
    constraints_relaxed: list[str] = Field(default_factory=list)
    filtered_hits_before: int = 0
    filtered_hits_after: int = 0
    used_fallback: bool = False
    rerank_cfg_hash: str = ""
    top1_kind: str = ""
    top1_in_distractor: bool = False
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    meta: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    kind: str
    id: str
    t0: float
    t1: float
    score: float = 0.0
    rank: int = 0
    source_query: str = ""
    chosen_plan_intent: str = ""
    applied_constraints: list[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    rerank_cfg_hash: str = ""
    top1_kind: str = ""
    top1_in_distractor: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    id: str
    type: str
    t0: float
    t1: float
    conf: float = 0.0
    source: dict[str, Any] = Field(default_factory=dict)
    retrieval_hit: RetrievalHit | None = None
    constraint_trace: ConstraintTrace | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class EventV1(BaseModel):
    id: str
    t0: float
    t1: float
    label: str = ""
    source_event_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class ContextSchema(BaseModel):
    video_id: str
    meta: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    mode: str = "highlights"
    budget: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    highlights: list[dict[str, Any]] = Field(default_factory=list)
    decision_points: list[dict[str, Any]] = Field(default_factory=list)
    tokens: list[Token] = Field(default_factory=list)
    token_stats: dict[str, Any] = Field(default_factory=dict)


class Output(BaseModel):
    video_id: str
    meta: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
    highlights: list[KeyClip] = Field(default_factory=list)
    token_codec: TokenCodec = Field(default_factory=TokenCodec)
    events: list[Event] = Field(default_factory=list)
    events_v0: list[Event] = Field(default_factory=list)
    events_v1: list[EventV1] = Field(default_factory=list)
    decision_points: list[DecisionPoint] = Field(default_factory=list)
    perception: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(
        default_factory=lambda: {
            "signals": {
                "time": [],
                "motion_energy": [],
                "embed_dist": [],
                "boundary_score": [],
            }
        }
    )
