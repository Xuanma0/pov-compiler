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
    decision_points: list[DecisionPoint] = Field(default_factory=list)
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
