from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ego4d_smoke import scan_and_plan


def _write_bytes(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * int(n))


def test_scan_and_plan_deterministic_and_fields(tmp_path: Path) -> None:
    _write_bytes(tmp_path / "a.mp4", 128)
    _write_bytes(tmp_path / "sub" / "b.mp4", 256)
    _write_bytes(tmp_path / "sub2" / "c.mp4", 512)
    _write_bytes(tmp_path / "ignore.txt", 512)

    plan1 = scan_and_plan(root=tmp_path, n=2, seed=7, min_size_bytes=0)
    plan2 = scan_and_plan(root=tmp_path, n=2, seed=7, min_size_bytes=0)

    assert len(plan1) == 3
    assert len(plan2) == 3

    required = {"video_uid", "src_path", "size_bytes", "proxy_path", "chosen"}
    for row in plan1:
        assert required.issubset(set(row.keys()))
        assert str(row["src_path"]).lower().endswith(".mp4")
        assert isinstance(row["chosen"], bool)

    chosen1 = sorted([row["video_uid"] for row in plan1 if row["chosen"]])
    chosen2 = sorted([row["video_uid"] for row in plan2 if row["chosen"]])
    assert chosen1 == chosen2
    assert len(chosen1) == 2


def test_scan_and_plan_respects_n_and_min_size(tmp_path: Path) -> None:
    _write_bytes(tmp_path / "x.mp4", 50)
    _write_bytes(tmp_path / "y.mp4", 60)
    _write_bytes(tmp_path / "z.mp4", 70)

    plan = scan_and_plan(root=tmp_path, n=10, seed=0, min_size_bytes=55)
    assert len(plan) == 2
    assert all(row["chosen"] for row in plan)
