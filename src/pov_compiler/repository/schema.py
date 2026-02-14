from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, model_validator


class RepoChunk(BaseModel):
    id: str = ""
    chunk_id: str = ""
    scale: str = ""
    level: str = ""
    t0: float = 0.0
    t1: float = 0.0
    t0_ms: int = 0
    t1_ms: int = 0
    text: str = ""
    importance: float = 0.0
    source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    score_fields: dict[str, float] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_compat_fields(self) -> "RepoChunk":
        chunk_id = str(self.chunk_id or self.id).strip()
        self.chunk_id = chunk_id
        if not str(self.id).strip():
            self.id = chunk_id
        if not str(self.chunk_id).strip():
            self.chunk_id = str(self.id).strip()

        level = str(self.level or self.scale).strip() or "event"
        self.level = level
        if not str(self.scale).strip():
            self.scale = level
        if not str(self.level).strip():
            self.level = str(self.scale).strip() or "event"

        # Keep second and millisecond spans mutually consistent for backward compatibility.
        if self.t0_ms == 0 and self.t1_ms == 0 and (self.t0 != 0.0 or self.t1 != 0.0):
            self.t0_ms = int(round(float(self.t0) * 1000.0))
            self.t1_ms = int(round(float(self.t1) * 1000.0))
        elif (self.t0 == 0.0 and self.t1 == 0.0) and (self.t0_ms != 0 or self.t1_ms != 0):
            self.t0 = float(self.t0_ms) / 1000.0
            self.t1 = float(self.t1_ms) / 1000.0

        self.t0 = float(self.t0)
        self.t1 = float(self.t1)
        if self.t1 < self.t0:
            self.t0, self.t1 = self.t1, self.t0
        if self.t0_ms > self.t1_ms:
            self.t0_ms, self.t1_ms = self.t1_ms, self.t0_ms
        if self.t0_ms == 0 and self.t1_ms == 0 and (self.t0 != 0.0 or self.t1 != 0.0):
            self.t0_ms = int(round(float(self.t0) * 1000.0))
            self.t1_ms = int(round(float(self.t1) * 1000.0))
        if (self.t0 == 0.0 and self.t1 == 0.0) and (self.t0_ms != 0 or self.t1_ms != 0):
            self.t0 = float(self.t0_ms) / 1000.0
            self.t1 = float(self.t1_ms) / 1000.0

        self.importance = float(max(0.0, min(1.0, float(self.importance))))
        if math.isnan(self.importance):
            self.importance = 0.0
        if not self.text:
            self.text = f"{self.level} [{self.t0:.2f}-{self.t1:.2f}]"
        return self


class RepoWriteOp(BaseModel):
    op: str
    chunk_id: str
    reason: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class RepoReadOp(BaseModel):
    strategy: str
    query: str = ""
    budget: dict[str, Any] = Field(default_factory=dict)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    selected_total_chars: int = 0
    selected_total_seconds: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class RepoSnapshot(BaseModel):
    video_id: str
    cfg_hash: str = ""
    write_policy: str = ""
    read_policy: str = ""
    chunks_total: int = 0
    chunks_after_dedup: int = 0
    by_scale: dict[str, int] = Field(default_factory=dict)
    by_tag: dict[str, int] = Field(default_factory=dict)
    write_ops: list[RepoWriteOp] = Field(default_factory=list)
    read_ops: list[RepoReadOp] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
