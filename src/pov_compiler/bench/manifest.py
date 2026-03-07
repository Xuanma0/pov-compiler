from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required to load benchmark manifests.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _resolve_path(raw_value: str | None, base_dir: Path) -> str | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return str(path.resolve())
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return str(base_candidate)
    return str((repo_root() / path).resolve())


class BudgetPoint(BaseModel):
    key: str = ""
    budget_seconds: float | None = None
    max_total_s: float | None = None
    max_tokens: int | None = None
    max_decisions: int | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "BudgetPoint":
        key = str(self.key or "").strip()
        if key and "/" in key:
            parts = [part.strip() for part in key.split("/") if part.strip()]
            if len(parts) == 3:
                try:
                    seconds = float(parts[0])
                    tokens = int(parts[1])
                    decisions = int(parts[2])
                except Exception:
                    seconds = None
                    tokens = None
                    decisions = None
                else:
                    if self.budget_seconds is None:
                        self.budget_seconds = seconds
                    if self.max_total_s is None:
                        self.max_total_s = seconds
                    if self.max_tokens is None:
                        self.max_tokens = tokens
                    if self.max_decisions is None:
                        self.max_decisions = decisions
        if self.max_total_s is not None and self.budget_seconds is None:
            self.budget_seconds = float(self.max_total_s)
        if not key:
            seconds = int(round(float(self.budget_seconds or self.max_total_s or 0.0)))
            tokens = int(self.max_tokens or 0)
            decisions = int(self.max_decisions or 0)
            key = f"{seconds}/{tokens}/{decisions}"
        self.key = key
        if self.budget_seconds is not None:
            self.budget_seconds = float(self.budget_seconds)
        if self.max_total_s is not None:
            self.max_total_s = float(self.max_total_s)
        if self.max_tokens is not None:
            self.max_tokens = int(self.max_tokens)
        if self.max_decisions is not None:
            self.max_decisions = int(self.max_decisions)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "budget_seconds": self.budget_seconds,
            "max_total_s": self.max_total_s,
            "max_tokens": self.max_tokens,
            "max_decisions": self.max_decisions,
        }


class BudgetConfig(BaseModel):
    points: list[BudgetPoint] = Field(
        default_factory=lambda: [
            BudgetPoint(key="20/50/4"),
            BudgetPoint(key="40/100/8"),
            BudgetPoint(key="60/200/12"),
        ]
    )


class SelectionConfig(BaseModel):
    compare_dir: str = "data/outputs/v1_42_compare"
    tasks: list[str] = Field(default_factory=lambda: ["nlq", "streaming", "bye"])
    labels: dict[str, str] = Field(default_factory=lambda: {"a": "stub", "b": "real"})
    task_sources: dict[str, str] = Field(
        default_factory=lambda: {
            "nlq": "nlq_budget/{label}/aggregate/metrics_by_budget.csv",
            "streaming": "streaming_budget/{label}/aggregate/metrics_by_budget.csv",
            "bye": "bye_budget/{label}/aggregate/metrics_by_budget.csv",
        }
    )
    pair_sources: dict[str, str] = Field(default_factory=dict)


class QueryConfig(BaseModel):
    profile: str = "paper_main"
    families: list[str] = Field(default_factory=list)


class VariantConfig(BaseModel):
    baseline: str = "a"
    treatment: str = "b"


class MetricConfig(BaseModel):
    primary: dict[str, str] = Field(
        default_factory=lambda: {
            "nlq": "nlq_full_hit_at_k_strict",
            "streaming": "hit@k_strict",
            "bye": "qualityScore",
        }
    )
    binary: dict[str, str] = Field(default_factory=dict)
    failure: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "nlq": [
                "safety_reason_budget_insufficient_rate",
                "safety_reason_evidence_missing_rate",
                "safety_reason_constraints_over_filtered_rate",
                "safety_reason_retrieval_distractor_rate",
                "safety_reason_other_rate",
            ]
        }
    )


class PromptConfig(BaseModel):
    registry: str = "configs/prompts/registry_v1.yaml"
    profile: str = "v1.42_main"
    lock_required: bool = True


class OutputConfig(BaseModel):
    root: str = "data/outputs/v1_42_suite"
    figure_formats: list[str] = Field(default_factory=lambda: ["png", "pdf"])
    include_paper_ready: bool = True
    include_submission_pack: bool = True


class ExperimentManifest(BaseModel):
    suite_id: str
    suite_version: str
    seed: int = 0
    description: str = ""
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    queries: QueryConfig = Field(default_factory=QueryConfig)
    variants: VariantConfig = Field(default_factory=VariantConfig)
    metrics: MetricConfig = Field(default_factory=MetricConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_variants(self) -> "ExperimentManifest":
        labels = set(self.selection.labels.keys())
        if self.variants.baseline not in labels:
            raise ValueError(f"Unknown baseline variant key: {self.variants.baseline}")
        if self.variants.treatment not in labels:
            raise ValueError(f"Unknown treatment variant key: {self.variants.treatment}")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> "ExperimentManifest":
        manifest_path = Path(path)
        payload = _load_yaml(manifest_path)
        return cls.model_validate(payload)

    def resolved_dict(self, manifest_path: str | Path) -> dict[str, Any]:
        base_dir = Path(manifest_path).resolve().parent
        payload = self.model_dump(mode="json")
        payload["manifest_path"] = str(Path(manifest_path).resolve())
        selection = dict(payload.get("selection", {}))
        selection["compare_dir"] = _resolve_path(selection.get("compare_dir"), base_dir)
        payload["selection"] = selection
        prompts = dict(payload.get("prompts", {}))
        prompts["registry"] = _resolve_path(prompts.get("registry"), base_dir)
        payload["prompts"] = prompts
        output = dict(payload.get("output", {}))
        output["root"] = _resolve_path(output.get("root"), base_dir)
        payload["output"] = output
        return payload

    def budget_index(self) -> dict[str, BudgetPoint]:
        return {point.key: point for point in self.budgets.points}


def load_manifest(path: str | Path) -> tuple[ExperimentManifest, dict[str, Any]]:
    manifest_path = Path(path)
    manifest = ExperimentManifest.from_path(manifest_path)
    return manifest, manifest.resolved_dict(manifest_path)


def write_resolved_manifest(resolved_manifest: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolved_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
