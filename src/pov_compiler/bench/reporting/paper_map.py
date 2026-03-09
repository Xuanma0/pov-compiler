from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required to load paper-map files.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Paper map must be a mapping: {path}")
    return payload


def stable_paper_map_hash(payload: dict[str, Any]) -> str:
    cloned = json.loads(json.dumps(payload, ensure_ascii=False))
    if isinstance(cloned, dict):
        cloned.pop("paper_map_hash", None)
    canonical = json.dumps(cloned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip()).strip("_")
    return slug or "artifact"


class PaperMapEntry(BaseModel):
    canonical_id: str
    kind: str
    title: str = ""
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "PaperMapEntry":
        kind = str(self.kind).strip().lower()
        if kind not in {"table", "figure"}:
            raise ValueError(f"Unsupported paper-map kind: {self.kind}")
        if not self.sources:
            raise ValueError(f"Paper-map entry has no sources: {self.canonical_id}")
        suffixes = sorted({Path(source).suffix.lower() for source in self.sources})
        if kind == "table" and not any(suffix in {".csv", ".md"} for suffix in suffixes):
            raise ValueError(f"Table entry requires csv/md sources: {self.canonical_id}")
        if kind == "figure" and not any(suffix in {".png", ".pdf"} for suffix in suffixes):
            raise ValueError(f"Figure entry requires png/pdf sources: {self.canonical_id}")
        self.kind = kind
        return self


class PaperMap(BaseModel):
    paper_map_id: str
    paper_map_version: str
    description: str = ""
    entries: list[PaperMapEntry] = Field(default_factory=list)
    paper_map_hash: str = ""

    @model_validator(mode="after")
    def _validate(self) -> "PaperMap":
        canonical_ids = [entry.canonical_id for entry in self.entries]
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("Paper map contains duplicate canonical_id values.")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> "PaperMap":
        payload = _load_yaml(Path(path))
        return cls.model_validate(payload)


def load_paper_map(path: str | Path) -> tuple[PaperMap, dict[str, Any]]:
    paper_map_path = Path(path).resolve()
    payload = _load_yaml(paper_map_path)
    paper_map = PaperMap.model_validate(payload)
    return paper_map, payload


def _resolve_source(source_relpath: str, roots: dict[str, Path | None]) -> Path:
    path = Path(str(source_relpath or "").strip())
    parts = path.parts
    if len(parts) < 2:
        raise ValueError(f"Paper-map source must start with a root alias: {source_relpath}")
    alias = str(parts[0])
    root = roots.get(alias)
    if root is None:
        raise ValueError(f"Unknown or unavailable paper-map root alias `{alias}` for {source_relpath}")
    return (Path(root) / Path(*parts[1:])).resolve()


def validate_paper_map_sources(paper_map_path: str | Path, roots: dict[str, Path | None]) -> dict[str, Any]:
    paper_map, payload = load_paper_map(paper_map_path)
    entries: list[dict[str, Any]] = []
    for entry in paper_map.entries:
        suffixes: list[str] = []
        for source_relpath in entry.sources:
            resolved = _resolve_source(source_relpath, roots)
            if not resolved.exists():
                raise FileNotFoundError(f"Paper-map target not found: {source_relpath} -> {resolved}")
            suffix = resolved.suffix.lower()
            suffixes.append(suffix)
            if entry.kind == "table" and suffix not in {".csv", ".md"}:
                raise ValueError(f"Table paper-map entry has non-table source: {source_relpath}")
            if entry.kind == "figure" and suffix not in {".png", ".pdf"}:
                raise ValueError(f"Figure paper-map entry has non-figure source: {source_relpath}")
            entries.append(
                {
                    "canonical_id": entry.canonical_id,
                    "kind": entry.kind,
                    "title": entry.title,
                    "source_relpath": source_relpath,
                    "resolved_path": str(resolved),
                    "suffix": suffix,
                }
            )
        if entry.kind == "table" and ".csv" not in suffixes:
            raise ValueError(f"Table paper-map entry is missing csv source: {entry.canonical_id}")
        if entry.kind == "figure" and ".png" not in suffixes:
            raise ValueError(f"Figure paper-map entry is missing png source: {entry.canonical_id}")
    return {
        "paper_map_id": paper_map.paper_map_id,
        "paper_map_version": paper_map.paper_map_version,
        "paper_map_hash": stable_paper_map_hash(payload),
        "entries": entries,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [
        "# Canonical Paper Artifact Map",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_canonical_artifacts(
    *,
    paper_map_path: str | Path,
    roots: dict[str, Path | None],
    paper_ready_dir: str | Path,
) -> dict[str, Any]:
    validated = validate_paper_map_sources(paper_map_path, roots)
    paper_ready_root = Path(paper_ready_dir).resolve()
    canonical_root = paper_ready_root / "canonical"
    tables_root = canonical_root / "tables"
    figures_root = canonical_root / "figures"
    manifest_root = paper_ready_root / "manifest"
    tables_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    copied_rows: list[dict[str, Any]] = []
    for row in validated["entries"]:
        canonical_id = str(row["canonical_id"])
        kind = str(row["kind"])
        resolved_path = Path(str(row["resolved_path"]))
        source_relpath = str(row["source_relpath"])
        base_name = canonical_slug(canonical_id)
        target_root = tables_root if kind == "table" else figures_root
        target = target_root / f"{base_name}{resolved_path.suffix.lower()}"
        shutil.copyfile(resolved_path, target)
        copied_rows.append(
            {
                "canonical_id": canonical_id,
                "kind": kind,
                "title": str(row.get("title", "")),
                "source_relpath": source_relpath,
                "canonical_relpath": str(target.relative_to(paper_ready_root)),
            }
        )

    map_src = Path(paper_map_path).resolve()
    map_copy = manifest_root / map_src.name
    shutil.copyfile(map_src, map_copy)

    csv_path = canonical_root / "table_paper_artifact_map.csv"
    md_path = canonical_root / "table_paper_artifact_map.md"
    resolved_json = canonical_root / "paper_map_resolved.json"
    columns = ["canonical_id", "kind", "title", "source_relpath", "canonical_relpath"]
    _write_csv(csv_path, copied_rows, columns)
    _write_markdown(md_path, copied_rows, columns)
    resolved_payload = {
        "paper_map_id": validated["paper_map_id"],
        "paper_map_version": validated["paper_map_version"],
        "paper_map_hash": validated["paper_map_hash"],
        "rows": copied_rows,
        "map_copy": str(map_copy),
    }
    resolved_json.write_text(json.dumps(resolved_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "paper_map_id": validated["paper_map_id"],
        "paper_map_version": validated["paper_map_version"],
        "paper_map_hash": validated["paper_map_hash"],
        "map_copy": str(map_copy),
        "mapping_csv": str(csv_path),
        "mapping_md": str(md_path),
        "resolved_json": str(resolved_json),
        "rows": copied_rows,
    }


def canonical_artifacts_for_freeze(paper_ready_dir: str | Path, paper_map_path: str | Path) -> dict[str, Any]:
    paper_ready_root = Path(paper_ready_dir).resolve()
    paper_map, payload = load_paper_map(paper_map_path)
    files: list[Path] = []
    rows: list[dict[str, str]] = []
    for entry in paper_map.entries:
        target_root = paper_ready_root / "canonical" / ("tables" if entry.kind == "table" else "figures")
        slug = canonical_slug(entry.canonical_id)
        suffixes = sorted({Path(source).suffix.lower() for source in entry.sources})
        for suffix in suffixes:
            target = target_root / f"{slug}{suffix}"
            if not target.exists():
                raise FileNotFoundError(f"Canonical paper artifact missing: {target}")
            files.append(target)
            rows.append(
                {
                    "canonical_id": entry.canonical_id,
                    "kind": entry.kind,
                    "relpath": str(target.relative_to(paper_ready_root)),
                }
            )
    map_copy = paper_ready_root / "manifest" / Path(paper_map_path).name
    if map_copy.exists():
        files.append(map_copy)
    else:
        files.append(Path(paper_map_path).resolve())
    return {
        "paper_map_id": paper_map.paper_map_id,
        "paper_map_version": paper_map.paper_map_version,
        "paper_map_hash": stable_paper_map_hash(payload),
        "files": files,
        "rows": rows,
    }
