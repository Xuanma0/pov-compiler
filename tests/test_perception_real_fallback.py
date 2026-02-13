from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.perception.backends import StubPerceptionBackend
from pov_compiler.perception import runner as perception_runner


def _make_tiny_video(path: Path, *, fps: float = 8.0, frames: int = 12, w: int = 128, h: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h)))
    assert writer.isOpened()
    for i in range(frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(img, (20 + i * 2, 40), 8, (255, 255, 255), -1)
        writer.write(img)
    writer.release()


def test_real_backend_missing_deps_fallback_to_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _make_tiny_video(video)

    def _fake_create_backend(name: str, **kwargs):  # type: ignore[no-untyped-def]
        if str(name) == "real":
            raise ImportError("ultralytics missing")
        return StubPerceptionBackend()

    monkeypatch.setattr(perception_runner, "create_backend", _fake_create_backend)

    output = perception_runner.run_perception(
        video_path=video,
        sample_fps=6.0,
        max_frames=8,
        backend_name="real",
        fallback_to_stub=True,
        strict=False,
        cache_dir=tmp_path / "cache",
    )
    summary = output.get("summary", {})
    assert bool(summary.get("fallback_used", False))
    assert str(summary.get("backend", "")) == "stub"
    assert bool(summary.get("deps_ok", True)) is False


def test_real_backend_missing_deps_strict_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    _make_tiny_video(video)

    def _fake_create_backend(name: str, **kwargs):  # type: ignore[no-untyped-def]
        if str(name) == "real":
            raise ImportError("mediapipe missing")
        return StubPerceptionBackend()

    monkeypatch.setattr(perception_runner, "create_backend", _fake_create_backend)

    with pytest.raises(RuntimeError):
        perception_runner.run_perception(
            video_path=video,
            sample_fps=6.0,
            max_frames=8,
            backend_name="real",
            fallback_to_stub=False,
            strict=True,
            cache_dir=tmp_path / "cache",
        )

