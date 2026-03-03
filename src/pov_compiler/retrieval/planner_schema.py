from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PlannerBackend = Literal["heuristic", "model", "auto"]
RetrievalPlanType = Literal[
    "baseline",
    "summary_then_token",
    "summary_then_decision",
    "summary_then_event",
    "summary_then_token_then_decision",
    "direct",
]


@dataclass(slots=True)
class PlannerPlan:
    retrieval_plan: str = "baseline"
    normalized_query: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        rp = str(self.retrieval_plan or "baseline").strip().lower()
        allowed = {
            "baseline",
            "summary_then_token",
            "summary_then_decision",
            "summary_then_event",
            "summary_then_token_then_decision",
            "direct",
        }
        self.retrieval_plan = rp if rp in allowed else "baseline"
        self.normalized_query = str(self.normalized_query or "").strip()
        self.constraints = dict(self.constraints or {})
        try:
            conf = float(self.confidence)
        except Exception:
            conf = 0.0
        if conf != conf:
            conf = 0.0
        self.confidence = float(max(0.0, min(1.0, conf)))
        self.notes = str(self.notes or "")

    @classmethod
    def from_obj(cls, obj: dict[str, Any] | None) -> "PlannerPlan":
        data = dict(obj or {})
        constraints = data.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}
        return cls(
            retrieval_plan=str(data.get("retrieval_plan", "baseline")),
            normalized_query=str(data.get("normalized_query", "")),
            constraints=constraints,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_plan": str(self.retrieval_plan),
            "normalized_query": str(self.normalized_query),
            "constraints": dict(self.constraints),
            "confidence": float(self.confidence),
            "notes": str(self.notes),
        }

    def to_query_plan(self, *, default_priority: int = 0):
        from pov_compiler.retrieval.query_planner import QueryCandidate, QueryPlan

        normalized = str(self.normalized_query or "").strip()
        if not normalized:
            normalized = "text= top_k=6"
        candidates: list[QueryCandidate] = [
            QueryCandidate(
                query=normalized,
                reason="planner_model",
                priority=int(default_priority),
            )
        ]
        debug = {
            "planner_model": True,
            "planner_confidence": float(self.confidence),
            "planner_notes": str(self.notes),
        }
        return QueryPlan(
            intent="mixed",
            candidates=candidates,
            constraints=dict(self.constraints),
            debug=debug,
            retrieval_plan=(
                str(self.retrieval_plan)
                if str(self.retrieval_plan) in {"baseline", "summary_then_token", "summary_then_decision", "summary_then_event"}
                else "baseline"
            ),
            plan_steps=[str(self.retrieval_plan)],
        )
