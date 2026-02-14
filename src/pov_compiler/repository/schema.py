from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RepoChunk(BaseModel):
    id: str
    scale: str
    t0: float
    t1: float
    text: str
    importance: float = 0.0
    source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


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
    chunks_total: int = 0
    chunks_after_dedup: int = 0
    by_scale: dict[str, int] = Field(default_factory=dict)
    by_tag: dict[str, int] = Field(default_factory=dict)
    write_ops: list[RepoWriteOp] = Field(default_factory=list)
    read_ops: list[RepoReadOp] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
