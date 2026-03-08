from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.perception.runner import run_perception


class _FakeSam3Backend:
    name = "real"

    def __init__(self, *, yolo_path: Path, sam3_path: Path, repo_path: Path) -> None:
        self.backend_used = "real"
        self.model_name = "yolo26n"
        self.model_path = str(yolo_path)
        self.hand_task_model_path = ""
        self.perception_variant = "yolo26n_plus_sam3"
        self.segmentation_backend_used = "sam3_local_proxy"
        self.segmentation_model_name = "sam3"
        self.segmentation_model_path = str(sam3_path)
        self.segmentation_repo_path = str(repo_path)
        self.segmentation_runtime_status = "proxy_cpu_only"

    def detect(self, frame_bgr: np.ndarray, *, frame_index: int, t: float) -> dict[str, object]:
        persistent = frame_index >= 1
        return {
            "objects": [
                {
                    "id": f"obj_{frame_index}",
                    "label": "bowl",
                    "conf": 0.9,
                    "bbox": [10.0, 12.0, 42.0, 48.0],
                    "track_id": "trk_0001",
                    "mask_area": 512.0,
                    "persistence_count": frame_index + 1,
                    "persistence_score": 1.0 if persistent else 0.5,
                    "persistent": persistent,
                    "segmentation_backend_used": "sam3_local_proxy",
                    "segmentation_model_name": "sam3",
                    "segmentation_model_path": str(self.segmentation_model_path),
                }
            ],
            "hands": [],
            "segmentation": {
                "backend_used": "sam3_local_proxy",
                "model_name": "sam3",
                "model_path": str(self.segmentation_model_path),
                "persistent_tracks": ["trk_0001"] if persistent else [],
            },
        }


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 4.0, (64, 64))
    assert writer.isOpened()
    for idx in range(3):
        frame = np.full((64, 64, 3), 40 + idx * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_sam3_local_smoke(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    _write_video(video_path)
    repo_path = tmp_path / "sam3_repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    yolo_path = tmp_path / "yolo26n.pt"
    sam3_path = tmp_path / "sam3.pt"
    yolo_path.write_bytes(b"YOLO")
    sam3_path.write_bytes(b"SAM3")
    cache_dir = tmp_path / "cache"

    payload = run_perception(
        video_path=video_path,
        sample_fps=4.0,
        max_frames=3,
        backend_name="real",
        backend=_FakeSam3Backend(yolo_path=yolo_path, sam3_path=sam3_path, repo_path=repo_path),
        backend_kwargs={
            "model_candidates": [str(yolo_path)],
            "segmentation_enabled": True,
            "segmentation_backend": "sam3_local",
            "segmentation_model_path": str(sam3_path),
            "segmentation_repo_path": str(repo_path),
            "perception_variant": "yolo26n_plus_sam3",
        },
        fallback_to_stub=False,
        cache_dir=cache_dir,
    )

    summary = payload["summary"]
    assert summary["segmentation_backend_used"] == "sam3_local_proxy"
    assert summary["segmentation_model_name"] == "sam3"
    assert summary["segmentation_model_path"] == str(sam3_path)
    assert summary["segmentation_frames_total"] == 3
    assert summary["persistent_tracks_total"] == 1
    assert summary["persistent_object_instances_total"] >= 2
    assert summary["cache_hit"] is False

    cached = run_perception(
        video_path=video_path,
        sample_fps=4.0,
        max_frames=3,
        backend_name="real",
        backend=_FakeSam3Backend(yolo_path=yolo_path, sam3_path=sam3_path, repo_path=repo_path),
        backend_kwargs={
            "model_candidates": [str(yolo_path)],
            "segmentation_enabled": True,
            "segmentation_backend": "sam3_local",
            "segmentation_model_path": str(sam3_path),
            "segmentation_repo_path": str(repo_path),
            "perception_variant": "yolo26n_plus_sam3",
        },
        fallback_to_stub=False,
        cache_dir=cache_dir,
    )
    assert cached["summary"]["cache_hit"] is True

    other_sam3_path = tmp_path / "sam3_other.pt"
    other_sam3_path.write_bytes(b"SAM3-OTHER")
    miss = run_perception(
        video_path=video_path,
        sample_fps=4.0,
        max_frames=3,
        backend_name="real",
        backend=_FakeSam3Backend(yolo_path=yolo_path, sam3_path=other_sam3_path, repo_path=repo_path),
        backend_kwargs={
            "model_candidates": [str(yolo_path)],
            "segmentation_enabled": True,
            "segmentation_backend": "sam3_local",
            "segmentation_model_path": str(other_sam3_path),
            "segmentation_repo_path": str(repo_path),
            "perception_variant": "yolo26n_plus_sam3",
        },
        fallback_to_stub=False,
        cache_dir=cache_dir,
    )
    assert miss["summary"]["cache_hit"] is False
