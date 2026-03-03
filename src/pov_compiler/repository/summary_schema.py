from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RepoSummaryStepV0(BaseModel):
    title: str = ""
    t0_ms: int | None = None
    t1_ms: int | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class RepoSummaryEntityV0(BaseModel):
    name: str = ""
    type: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class RepoSummaryInteractionV0(BaseModel):
    object: str = ""
    score: float | None = None
    verb: str | None = None


class RepoSummaryDecisionV0(BaseModel):
    type: str = ""
    description: str = ""
    confidence: float | None = None


class RepoSummaryProvenanceV0(BaseModel):
    input_chunk_ids: list[str] = Field(default_factory=list)
    policy: str = ""
    model: dict[str, Any] = Field(default_factory=dict)


class RepoSummaryV0(BaseModel):
    video_id: str
    t0_ms: int
    t1_ms: int
    goals: list[str] = Field(default_factory=list)
    steps: list[RepoSummaryStepV0] = Field(default_factory=list)
    entities: list[RepoSummaryEntityV0] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    interactions: list[RepoSummaryInteractionV0] = Field(default_factory=list)
    decisions: list[RepoSummaryDecisionV0] = Field(default_factory=list)
    notes: str | None = None
    provenance: RepoSummaryProvenanceV0 = Field(default_factory=RepoSummaryProvenanceV0)

