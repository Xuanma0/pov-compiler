from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


class VideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video does not exist: {self.path}")
        self._meta = self._probe()

    @property
    def meta(self) -> VideoMeta:
        return self._meta

    @property
    def fps(self) -> float:
        return self._meta.fps

    @property
    def duration_s(self) -> float:
        return self._meta.duration_s

    def _probe(self) -> VideoMeta:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_s = float(frame_count / fps) if frame_count > 0 else 0.0
        cap.release()
        return VideoMeta(
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
            width=width,
            height=height,
        )

    def iter_samples(self, sample_fps: float = 4.0) -> Iterator[tuple[float, np.ndarray]]:
        if sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            src_fps = self.fps if self.fps > 0 else 30.0

        frame_idx = 0
        sample_period = 1.0 / sample_fps
        next_sample_t = 0.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                t = frame_idx / src_fps
                if t + 1e-9 >= next_sample_t:
                    yield float(t), frame
                    next_sample_t += sample_period

                frame_idx += 1
        finally:
            cap.release()
