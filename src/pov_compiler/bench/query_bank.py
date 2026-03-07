from __future__ import annotations

import csv
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
        raise RuntimeError("PyYAML is required to load query bank files.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Query bank must be a mapping: {path}")
    return payload


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_query_bank_hash(payload: dict[str, Any]) -> str:
    cloned = json.loads(json.dumps(payload, ensure_ascii=False))
    if isinstance(cloned, dict):
        cloned.pop("query_bank_hash", None)
    return hashlib.sha256(_canonical_json(cloned).encode("utf-8")).hexdigest()


def _resolve_path(raw_value: str | None, base_dir: Path) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("Query bank path may not be empty.")
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (_repo_root() / path).resolve()


class QueryBankGroup(BaseModel):
    group_id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class QueryBankEntry(BaseModel):
    query_id: str
    group: str
    task: str
    mode: str = ""
    query: str
    signal_tags: list[str] = Field(default_factory=list)
    sensitive_to: list[str] = Field(default_factory=list)
    enabled: bool = True
    notes: str = ""


class QueryBank(BaseModel):
    query_bank_id: str
    query_bank_version: str
    description: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    groups: list[QueryBankGroup] = Field(default_factory=list)
    queries: list[QueryBankEntry] = Field(default_factory=list)
    query_bank_hash: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "QueryBank":
        group_ids = {group.group_id for group in self.groups}
        if len(group_ids) != len(self.groups):
            raise ValueError("Query bank contains duplicate group_id values.")
        query_ids = [query.query_id for query in self.queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Query bank contains duplicate query_id values.")
        for query in self.queries:
            if query.group not in group_ids:
                raise ValueError(f"Query `{query.query_id}` references unknown group `{query.group}`.")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> "QueryBank":
        bank_path = Path(path)
        payload = _load_yaml(bank_path)
        return cls.model_validate(payload)

    def declared_hash_ok(self, bank_path: str | Path) -> bool:
        payload = _load_yaml(Path(bank_path))
        declared = str(payload.get("query_bank_hash", "")).strip()
        if not declared:
            return False
        return declared == stable_query_bank_hash(payload)

    def build_lock(self, bank_path: str | Path, *, is_primary: bool = False) -> dict[str, Any]:
        resolved = Path(bank_path).resolve()
        payload = _load_yaml(resolved)
        computed_hash = stable_query_bank_hash(payload)
        try:
            relative_path = str(resolved.relative_to(_repo_root()))
        except Exception:
            relative_path = str(resolved)
        groups = [group.group_id for group in self.groups]
        enabled_queries = [query for query in self.queries if bool(query.enabled)]
        return {
            "query_bank_id": self.query_bank_id,
            "query_bank_version": self.query_bank_version,
            "query_bank_hash": computed_hash,
            "declared_hash": str(payload.get("query_bank_hash", "")).strip(),
            "hash_matches": bool(str(payload.get("query_bank_hash", "")).strip() == computed_hash),
            "path": relative_path,
            "is_primary": bool(is_primary),
            "groups": groups,
            "queries_total": int(len(self.queries)),
            "queries_enabled": int(len(enabled_queries)),
            "provenance": self.provenance,
        }


def load_query_banks_from_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    payload = _load_yaml(manifest_file)
    queries = payload.get("queries", {}) if isinstance(payload, dict) else {}
    if not isinstance(queries, dict):
        queries = {}
    primary_raw = queries.get("query_bank")
    extra_raw = queries.get("auxiliary_banks", [])
    if isinstance(extra_raw, str):
        extra_raw = [extra_raw]
    bank_paths: list[Path] = []
    if primary_raw:
        bank_paths.append(_resolve_path(str(primary_raw), manifest_file.parent))
    for item in extra_raw if isinstance(extra_raw, list) else []:
        text = str(item or "").strip()
        if not text:
            continue
        bank_paths.append(_resolve_path(text, manifest_file.parent))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in bank_paths:
        norm = str(path.resolve())
        if norm in seen:
            continue
        deduped.append(path.resolve())
        seen.add(norm)

    banks: list[dict[str, Any]] = []
    for idx, bank_path in enumerate(deduped):
        bank = QueryBank.from_path(bank_path)
        banks.append(bank.build_lock(bank_path, is_primary=idx == 0))

    return {
        "primary": banks[0] if banks else None,
        "banks": banks,
        "groups": list(queries.get("groups", [])) if isinstance(queries.get("groups"), list) else [],
        "top_k": int(queries.get("top_k", 0) or 0) if str(queries.get("top_k", "")).strip() else None,
    }


def write_query_bank_lock(lock_payload: dict[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def copy_query_bank_files(lock_payload: dict[str, Any], out_dir: str | Path) -> list[Path]:
    import shutil

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for bank in lock_payload.get("banks", []):
        path_raw = str(bank.get("path", "")).strip()
        if not path_raw:
            continue
        src = _resolve_path(path_raw, _repo_root())
        dst = out_root / src.name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def selection_artifact_paths(compare_dir: str | Path, manifest_path: str | Path) -> dict[str, Path]:
    manifest_file = Path(manifest_path).resolve()
    payload = _load_yaml(manifest_file)
    selection = payload.get("selection", {}) if isinstance(payload, dict) else {}
    if not isinstance(selection, dict):
        selection = {}
    compare_root = Path(compare_dir).resolve()
    signal_root_raw = str(selection.get("signal_selection_dir", "")).strip()
    if signal_root_raw:
        candidate = Path(signal_root_raw)
        if not candidate.is_absolute():
            compare_candidate = (compare_root / candidate).resolve()
            if compare_candidate.exists():
                signal_root = compare_candidate
            else:
                signal_root = _resolve_path(signal_root_raw, manifest_file.parent)
        else:
            signal_root = candidate.resolve()
    else:
        signal_root = compare_root / "signal_selection"
    return {
        "root": signal_root,
        "coverage_csv": signal_root / "coverage.csv",
        "selected_uids": signal_root / "selected_uids.txt",
        "snapshot_json": signal_root / "snapshot.json",
    }


def selection_stats(compare_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    paths = selection_artifact_paths(compare_dir, manifest_path)
    selected_count = 0
    if paths["selected_uids"].exists():
        selected_count = len([line.strip() for line in paths["selected_uids"].read_text(encoding="utf-8").splitlines() if line.strip()])

    coverage_stats = {"count": 0, "min": None, "max": None, "mean": None}
    missing_breakdown: dict[str, int] = {}
    if paths["coverage_csv"].exists():
        with paths["coverage_csv"].open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        scores: list[float] = []
        for row in rows:
            try:
                score = float(row.get("coverage_score", ""))
            except Exception:
                score = None  # type: ignore[assignment]
            if score is not None:
                scores.append(float(score))
            for key, value in row.items():
                if not str(key).startswith("missing_"):
                    continue
                try:
                    count = int(float(value))
                except Exception:
                    count = 0
                missing_breakdown[str(key)] = int(missing_breakdown.get(str(key), 0) + count)
        if scores:
            coverage_stats = {
                "count": len(scores),
                "min": min(scores),
                "max": max(scores),
                "mean": sum(scores) / float(len(scores)),
            }

    selection_mode = ""
    if paths["snapshot_json"].exists():
        try:
            snapshot = json.loads(paths["snapshot_json"].read_text(encoding="utf-8"))
        except Exception:
            snapshot = {}
        selection_mode = str(snapshot.get("selection_mode", snapshot.get("mode", ""))).strip()

    return {
        "selection_mode": selection_mode or "",
        "selected_uids_count": int(selected_count),
        "coverage_score_stats": coverage_stats,
        "missing_signal_breakdown": missing_breakdown,
        "signal_selection_root": str(paths["root"]),
    }
