from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def is_uuid_like(value: str) -> bool:
    return bool(UUID_RE.match(str(value).strip()))


def md5_text(value: str) -> str:
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()


def to_posix_relative(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve())
    return rel.as_posix()


def list_mp4_files(root: Path, min_size_bytes: int = 10 * 1024 * 1024) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    mp4s: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".mp4":
            continue
        try:
            size = int(path.stat().st_size)
        except OSError:
            continue
        if size < int(min_size_bytes):
            continue
        mp4s.append(path)
    mp4s.sort(key=lambda p: str(p).lower())
    return mp4s


def make_video_uid(path: Path, root: Path) -> str:
    stem = str(path.stem)
    if is_uuid_like(stem):
        return stem
    rel = to_posix_relative(path, root)
    return md5_text(rel)


def choose_indices(total: int, n: int, seed: int) -> set[int]:
    n = max(0, min(int(n), int(total)))
    if n == 0 or total <= 0:
        return set()
    if n >= total:
        return set(range(total))
    rng = np.random.default_rng(int(seed))
    picks = rng.choice(total, size=n, replace=False)
    return {int(x) for x in picks.tolist()}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

