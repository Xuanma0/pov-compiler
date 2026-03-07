from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required to load prompt registry files.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt registry must be a mapping: {path}")
    return payload


def stable_prompt_hash(text: str) -> str:
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _resolve_path(raw_value: str | None, base_dir: Path) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("Prompt path may not be empty.")
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return ( _repo_root() / path).resolve()


class PromptRegistryError(RuntimeError):
    pass


class PromptEntry(BaseModel):
    prompt_id: str
    task: str
    version: str
    path: str
    owner: str = ""
    tags: list[str] = Field(default_factory=list)
    frozen: bool = True
    notes: str = ""

    def resolved_path(self, registry_path: str | Path) -> Path:
        return _resolve_path(self.path, Path(registry_path).resolve().parent)

    def read_text(self, registry_path: str | Path) -> str:
        return self.resolved_path(registry_path).read_text(encoding="utf-8")

    def content_hash(self, registry_path: str | Path) -> str:
        return stable_prompt_hash(self.read_text(registry_path))


class PromptProfile(BaseModel):
    description: str = ""
    prompts: dict[str, str] = Field(default_factory=dict)


class PromptRegistry(BaseModel):
    registry_id: str = "prompt_registry_v1"
    version: str = "1"
    entries: list[PromptEntry] = Field(default_factory=list)
    profiles: dict[str, PromptProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_entries(self) -> "PromptRegistry":
        ids = [entry.prompt_id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("Prompt registry contains duplicate prompt_id values.")
        task_versions = [(entry.task, entry.version) for entry in self.entries]
        if len(set(task_versions)) != len(task_versions):
            raise ValueError("Prompt registry contains duplicate task/version pairs.")
        by_id = {entry.prompt_id for entry in self.entries}
        for profile_name, profile in self.profiles.items():
            for prompt_id in profile.prompts.values():
                if prompt_id not in by_id:
                    raise ValueError(f"Profile `{profile_name}` references unknown prompt `{prompt_id}`.")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> "PromptRegistry":
        registry_path = Path(path)
        payload = _load_yaml(registry_path)
        return cls.model_validate(payload)

    def get_entry(self, prompt_id: str) -> PromptEntry:
        for entry in self.entries:
            if entry.prompt_id == prompt_id:
                return entry
        raise PromptRegistryError(f"Unknown prompt_id: {prompt_id}")

    def available_profiles(self) -> list[str]:
        return sorted(self.profiles.keys())

    def build_prompt_lock(self, registry_path: str | Path, profile_name: str) -> dict[str, Any]:
        if profile_name not in self.profiles:
            raise PromptRegistryError(f"Unknown prompt profile: {profile_name}")
        profile = self.profiles[profile_name]
        lock_prompts: list[dict[str, Any]] = []
        for task, prompt_id in sorted(profile.prompts.items(), key=lambda item: item[0]):
            entry = self.get_entry(prompt_id)
            resolved_path = entry.resolved_path(registry_path)
            text = entry.read_text(registry_path)
            relative_path: str
            try:
                relative_path = str(resolved_path.relative_to(_repo_root()))
            except Exception:
                relative_path = str(resolved_path)
            lock_prompts.append(
                {
                    "task": task,
                    "prompt_id": entry.prompt_id,
                    "version": entry.version,
                    "path": relative_path,
                    "owner": entry.owner,
                    "frozen": bool(entry.frozen),
                    "hash": stable_prompt_hash(text),
                    "chars": len(text),
                }
            )
        return {
            "registry_id": self.registry_id,
            "registry_version": self.version,
            "registry_path": str(Path(registry_path).resolve()),
            "profile": profile_name,
            "profile_description": profile.description,
            "prompts": lock_prompts,
        }

    def validate_paths(self, registry_path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in self.entries:
            resolved_path = entry.resolved_path(registry_path)
            rows.append(
                {
                    "prompt_id": entry.prompt_id,
                    "task": entry.task,
                    "exists": bool(resolved_path.exists()),
                    "resolved_path": str(resolved_path),
                    "hash": entry.content_hash(registry_path) if resolved_path.exists() else "",
                }
            )
        return rows


def write_prompt_lock(lock_payload: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
