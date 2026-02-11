from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dep
    faiss = None  # type: ignore


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = vec.astype(np.float32, copy=False).reshape(-1)
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-12:
        return arr
    return arr / denom


def _meta_match(meta: dict[str, Any], filter_meta: dict[str, Any] | Callable[[dict[str, Any]], bool] | None) -> bool:
    if filter_meta is None:
        return True
    if callable(filter_meta):
        return bool(filter_meta(meta))

    for key, expected in filter_meta.items():
        if key not in meta:
            return False
        actual = meta[key]
        if callable(expected):
            if not bool(expected(actual)):
                return False
            continue
        if isinstance(expected, (list, tuple, set)):
            expected_set = set(expected)
            if isinstance(actual, (list, tuple, set)):
                if not expected_set.intersection(set(actual)):
                    return False
            elif actual not in expected_set:
                return False
            continue
        if isinstance(actual, (list, tuple, set)):
            if expected not in set(actual):
                return False
        elif actual != expected:
            return False
    return True


@dataclass
class SearchHit:
    id: str
    score: float
    meta: dict[str, Any]


class VectorIndex:
    def __init__(self, use_faiss: bool | None = None):
        self.ids: list[str] = []
        self._vectors: list[np.ndarray] = []
        self.metas: list[dict[str, Any]] = []
        self.dim: int = 0

        self._faiss_enabled = bool(faiss is not None) if use_faiss is None else bool(use_faiss and faiss is not None)
        self._faiss_index: Any | None = None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss_enabled else "numpy"

    @property
    def size(self) -> int:
        return len(self.ids)

    def add(self, item_id: str, vec: np.ndarray, meta: dict[str, Any]) -> None:
        emb = _l2_normalize(np.asarray(vec, dtype=np.float32))
        if emb.ndim != 1:
            raise ValueError("Embedding must be a 1D vector")
        if emb.size == 0:
            raise ValueError("Embedding must not be empty")

        if self.dim == 0:
            self.dim = int(emb.size)
        if emb.size != self.dim:
            raise ValueError(f"Inconsistent embedding dim: expected={self.dim}, got={emb.size}")

        self.ids.append(str(item_id))
        self._vectors.append(emb)
        self.metas.append(dict(meta))
        self._faiss_index = None

    def _matrix(self) -> np.ndarray:
        if not self._vectors:
            if self.dim <= 0:
                return np.zeros((0, 0), dtype=np.float32)
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(self._vectors).astype(np.float32, copy=False)

    def _ensure_faiss(self) -> None:
        if not self._faiss_enabled or self.size == 0:
            return
        if self._faiss_index is not None:
            return
        matrix = self._matrix()
        self._faiss_index = faiss.IndexFlatIP(self.dim)  # type: ignore[attr-defined]
        self._faiss_index.add(matrix)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 8,
        filter_meta: dict[str, Any] | Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0 or self.size == 0:
            return []

        query = _l2_normalize(np.asarray(query_vec, dtype=np.float32))
        if query.size != self.dim:
            raise ValueError(f"Query dim mismatch: expected={self.dim}, got={query.size}")

        if self._faiss_enabled and filter_meta is None:
            self._ensure_faiss()
            if self._faiss_index is not None:
                scores, idxs = self._faiss_index.search(query.reshape(1, -1), int(top_k))
                hits: list[SearchHit] = []
                for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
                    if idx < 0:
                        continue
                    hits.append(
                        SearchHit(
                            id=self.ids[idx],
                            score=float(score),
                            meta=dict(self.metas[idx]),
                        )
                    )
                return hits

        candidate_idxs = [i for i, meta in enumerate(self.metas) if _meta_match(meta, filter_meta)]
        if not candidate_idxs:
            return []
        matrix = self._matrix()[candidate_idxs]
        sims = matrix @ query
        order = np.argsort(-sims)[: int(top_k)]

        hits = [
            SearchHit(
                id=self.ids[candidate_idxs[int(o)]],
                score=float(sims[int(o)]),
                meta=dict(self.metas[candidate_idxs[int(o)]]),
            )
            for o in order.tolist()
        ]
        return hits

    def save(self, path_prefix: str | Path) -> tuple[Path, Path]:
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        npz_path = Path(f"{prefix}.index.npz")
        meta_path = Path(f"{prefix}.index_meta.json")
        np.savez_compressed(
            npz_path,
            vectors=self._matrix().astype(np.float32),
            ids=np.asarray(self.ids, dtype=object),
        )
        payload = {
            "dim": int(self.dim),
            "backend": self.backend,
            "meta": self.metas,
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return npz_path, meta_path

    @classmethod
    def load(cls, path_prefix: str | Path, use_faiss: bool | None = None) -> "VectorIndex":
        prefix = Path(path_prefix)
        npz_path = Path(f"{prefix}.index.npz")
        meta_path = Path(f"{prefix}.index_meta.json")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing index file: {npz_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing index meta file: {meta_path}")

        npz = np.load(npz_path, allow_pickle=True)
        vectors = np.asarray(npz["vectors"], dtype=np.float32)
        ids = [str(x) for x in npz["ids"].tolist()]

        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(meta_payload, list):
            metas = [dict(x) for x in meta_payload]
            dim = int(vectors.shape[1] if vectors.ndim == 2 else 0)
        else:
            metas = [dict(x) for x in meta_payload.get("meta", [])]
            dim = int(meta_payload.get("dim", vectors.shape[1] if vectors.ndim == 2 else 0))

        if vectors.ndim == 1 and vectors.size > 0:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim == 2 and len(ids) != vectors.shape[0]:
            raise ValueError("Index vectors/ids size mismatch")
        if len(metas) != len(ids):
            raise ValueError("Index ids/meta size mismatch")

        index = cls(use_faiss=use_faiss)
        index.dim = int(dim)
        index.ids = ids
        index._vectors = [_l2_normalize(vectors[i]) for i in range(vectors.shape[0])] if vectors.size > 0 else []
        index.metas = metas
        return index
