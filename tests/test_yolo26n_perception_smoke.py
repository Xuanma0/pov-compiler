from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.perception.runner import run_perception


class _FakeRealBackend:
    name = "real"
    backend_used = "real"

    def __init__(self, *, model_path: Path, hand_task_path: Path) -> None:
        self.model_path = str(model_path)
        self.model_name = model_path.stem
        self.hand_task_model_path = str(hand_task_path)

    def detect(self, frame_bgr: np.ndarray, *, frame_index: int, t: float) -> dict[str, object]:
        return {
            "objects": [
                {
                    "label": "cup",
                    "score": 0.92,
                    "bbox": [8.0, 8.0, 24.0, 24.0],
                }
            ],
            "hands": [],
        }


def _write_test_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (48, 48))
    assert writer.isOpened()
    for idx in range(3):
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        frame[:, :, :] = (idx + 1) * 30
        writer.write(frame)
    writer.release()


def test_yolo26n_perception_smoke(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "clip.avi"
    _write_test_video(video_path)
    model_path = tmp_path / "yolo26n.pt"
    model_path.write_bytes(b"FAKE")
    hand_task_path = tmp_path / "hand_landmarker.task"
    hand_task_path.write_bytes(b"TASK")
    cache_dir = tmp_path / "cache"

    backend = _FakeRealBackend(model_path=model_path, hand_task_path=hand_task_path)
    payload = run_perception(
        video_path=video_path,
        sample_fps=2.0,
        max_frames=2,
        backend_name="real",
        backend=backend,
        backend_kwargs={
            "model_candidates": [str(model_path)],
            "hand_task_model_path": str(hand_task_path),
        },
        fallback_to_stub=False,
        cache_dir=cache_dir,
    )
    assert payload["summary"]["perception_backend_used"] == "real"
    assert payload["summary"]["perception_model_name"] == "yolo26n"
    assert payload["summary"]["perception_model_path"] == str(model_path)
    assert payload["meta"]["perception_hand_task_model_path"] == str(hand_task_path)
    assert payload["summary"]["cache_hit"] is False
    assert payload["summary"]["objects_total"] > 0

    def _should_not_create_backend(*args, **kwargs):
        raise AssertionError("cache miss unexpectedly called create_backend")

    monkeypatch.setattr("pov_compiler.perception.runner.create_backend", _should_not_create_backend)
    cached = run_perception(
        video_path=video_path,
        sample_fps=2.0,
        max_frames=2,
        backend_name="real",
        backend=None,
        backend_kwargs={
            "model_candidates": [str(model_path)],
            "hand_task_model_path": str(hand_task_path),
        },
        fallback_to_stub=False,
        cache_dir=cache_dir,
    )
    assert cached["summary"]["cache_hit"] is True
    assert cached["summary"]["perception_model_name"] == "yolo26n"
    assert cached["summary"]["perception_model_path"] == str(model_path)
