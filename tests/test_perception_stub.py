from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.perception.runner import run_perception


def _make_tiny_video(path: Path, *, fps: float = 10.0, frames: int = 20, w: int = 160, h: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h)))
    assert writer.isOpened()
    for i in range(frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (10 + i, 20), (40 + i, 50), (0, 255, 0), -1)
        writer.write(img)
    writer.release()


def test_perception_stub_schema_and_summary(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    _make_tiny_video(video)

    output = run_perception(
        video_path=video,
        sample_fps=10.0,
        max_frames=12,
        backend_name="stub",
        contact_min_score=0.2,
    )
    assert isinstance(output, dict)
    assert "frames" in output and isinstance(output["frames"], list)
    assert "signals" in output and isinstance(output["signals"], dict)
    assert "summary" in output and isinstance(output["summary"], dict)

    frames = output["frames"]
    assert 1 <= len(frames) <= 12
    first = frames[0]
    assert isinstance(first.get("objects", []), list)
    assert isinstance(first.get("hands", []), list)
    assert isinstance(first.get("contact", {}), dict)

    summary = output["summary"]
    assert "objects_topk" in summary
    assert float(summary.get("hand_presence_rate", 0.0)) >= 0.0
    assert int(summary.get("contact_events_count", 0)) >= 0

