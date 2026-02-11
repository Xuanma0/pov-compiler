from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.memory.vector_index import VectorIndex


def test_vector_index_search_order_and_filter() -> None:
    index = VectorIndex(use_faiss=False)
    index.add("a", np.array([1.0, 0.0], dtype=np.float32), {"kind": "event"})
    index.add("b", np.array([0.0, 1.0], dtype=np.float32), {"kind": "highlight"})
    index.add("c", np.array([0.8, 0.2], dtype=np.float32), {"kind": "highlight"})

    hits = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert [hit.id for hit in hits] == ["a", "c"]

    hits_filtered = index.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=5,
        filter_meta={"kind": "highlight"},
    )
    assert [hit.id for hit in hits_filtered] == ["c", "b"]


def test_vector_index_save_load_consistent(tmp_path: Path) -> None:
    prefix = tmp_path / "demo"
    index = VectorIndex(use_faiss=False)
    index.add("x", np.array([1.0, 0.0, 0.0], dtype=np.float32), {"kind": "event", "id": "x"})
    index.add("y", np.array([0.0, 1.0, 0.0], dtype=np.float32), {"kind": "highlight", "id": "y"})
    index.save(prefix)

    loaded = VectorIndex.load(prefix, use_faiss=False)
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits_a = index.search(q, top_k=2)
    hits_b = loaded.search(q, top_k=2)
    assert [h.id for h in hits_a] == [h.id for h in hits_b]
    assert [round(h.score, 6) for h in hits_a] == [round(h.score, 6) for h in hits_b]
