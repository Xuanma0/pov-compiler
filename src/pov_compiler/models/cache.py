from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "v1"


@dataclass(slots=True)
class ModelCacheStats:
    hit: int = 0
    miss: int = 0
    write_fail: int = 0
    hash_prefix: str = ""

    def to_dict(self) -> dict[str, int]:
        return {
            "hit": int(self.hit),
            "miss": int(self.miss),
            "write_fail": int(self.write_fail),
            "hash_prefix": str(self.hash_prefix),
        }


class ModelCallCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_entries: int = 0,
        max_mb: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_entries = max(0, int(max_entries))
        self.max_mb = max(0, int(max_mb))
        self.stats = ModelCacheStats()

    @staticmethod
    def build_key(
        *,
        provider: str,
        model: str,
        base_url: str,
        request_payload: dict[str, Any],
        schema_version: str = SCHEMA_VERSION,
    ) -> str:
        payload = {
            "provider": str(provider),
            "model": str(model),
            "base_url": str(base_url or ""),
            "request": request_payload,
            "schema_version": str(schema_version),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()

    def _entry_path(self, key: str) -> Path:
        safe = str(key).strip().lower()
        return self.cache_dir / safe[:2] / f"{safe}.json"

    def _current_size_bytes(self) -> int:
        total = 0
        if not self.cache_dir.exists():
            return 0
        for p in self.cache_dir.rglob("*.json"):
            try:
                total += int(p.stat().st_size)
            except Exception:
                continue
        return total

    def _current_entries(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return sum(1 for _ in self.cache_dir.rglob("*.json"))

    def get(self, key: str) -> dict[str, Any] | None:
        self.stats.hash_prefix = str(key)[:12]
        path = self._entry_path(key)
        if not path.exists():
            self.stats.miss += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.stats.hit += 1
                return payload
        except Exception:
            pass
        self.stats.miss += 1
        return None

    def set(self, key: str, value: dict[str, Any]) -> bool:
        try:
            self.stats.hash_prefix = str(key)[:12]
            if self.max_entries > 0 and self._current_entries() >= self.max_entries:
                self.stats.write_fail += 1
                return False
            if self.max_mb > 0 and (self._current_size_bytes() / (1024 * 1024)) >= float(self.max_mb):
                self.stats.write_fail += 1
                return False
            path = self._entry_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            return True
        except Exception:
            self.stats.write_fail += 1
            return False
