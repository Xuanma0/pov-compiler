from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _norm_name(path: Path) -> str:
    return path.name.lower()


def _search_script(bye_root: Path, candidates: list[str]) -> Path | None:
    wanted = {str(x).lower() for x in candidates}
    hits: list[Path] = []
    for p in bye_root.rglob("*.py"):
        if _norm_name(p) in wanted:
            hits.append(p)
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.relative_to(bye_root).parts), str(p.relative_to(bye_root)).lower()))
    return hits[0]


def _resolve_override(path_str: str | None, bye_root: Path) -> Path | None:
    if not path_str:
        return None
    raw = Path(path_str)
    p = raw if raw.is_absolute() else (bye_root / raw)
    if p.exists() and p.is_file():
        return p.resolve()
    return None


@dataclass
class EntryPointResolution:
    lint: Path | None = None
    report: Path | None = None
    regression: Path | None = None
    found: bool = False
    notes: list[str] = field(default_factory=list)
    methods: dict[str, str] = field(default_factory=dict)

    def as_dict(self, bye_root: Path | None = None) -> dict[str, Any]:
        def _to_rel(p: Path | None) -> str | None:
            if p is None:
                return None
            if bye_root is None:
                return str(p)
            try:
                return str(p.resolve().relative_to(bye_root.resolve())).replace("\\", "/")
            except Exception:
                return str(p)

        return {
            "lint": _to_rel(self.lint),
            "report": _to_rel(self.report),
            "regression": _to_rel(self.regression),
            "found": bool(self.found),
            "notes": list(self.notes),
            "methods": dict(self.methods),
        }


class EntryPointResolver:
    def __init__(self, bye_root: Path, overrides: dict[str, str | None] | None = None):
        self.bye_root = Path(bye_root)
        self.overrides = dict(overrides or {})

    def resolve(self) -> EntryPointResolution:
        res = EntryPointResolution()
        specs = {
            "lint": ["lint_run_package.py"],
            "report": ["report_run.py"],
            "regression": ["run_regression_suite.py", "regression_suite.py"],
        }

        for key, names in specs.items():
            override = _resolve_override(self.overrides.get(key), self.bye_root)
            if override is not None:
                setattr(res, key, override)
                res.methods[key] = "override"
                continue

            found = _search_script(self.bye_root, names)
            if found is not None:
                setattr(res, key, found.resolve())
                res.methods[key] = "search"
            else:
                res.methods[key] = "missing"
                res.notes.append(
                    f"missing_{key}_entrypoint: not found via search. "
                    f"Use override --bye-{key} to set explicit script path."
                )

        res.found = any(getattr(res, key) is not None for key in ("lint", "report", "regression"))
        return res

