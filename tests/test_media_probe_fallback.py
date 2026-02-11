from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.utils.media import probe_video_metadata


class _FakeCap:
    def __init__(self, *_: object) -> None:
        self._opened = True

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def get(self, prop: float) -> float:
        if prop == 1.0:  # CAP_PROP_FRAME_COUNT
            return 300.0
        if prop == 2.0:  # CAP_PROP_FPS
            return 30.0
        if prop == 3.0:  # CAP_PROP_FRAME_WIDTH
            return 1280.0
        if prop == 4.0:  # CAP_PROP_FRAME_HEIGHT
            return 720.0
        return 0.0

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_PROP_FRAME_COUNT = 1.0
    CAP_PROP_FPS = 2.0
    CAP_PROP_FRAME_WIDTH = 3.0
    CAP_PROP_FRAME_HEIGHT = 4.0
    VideoCapture = _FakeCap


def test_probe_fallback_to_cv2_when_ffprobe_missing(tmp_path: Path) -> None:
    video = tmp_path / "fake.mp4"
    video.write_bytes(b"123")
    result = probe_video_metadata(
        video,
        which_fn=lambda _: None,
        cv2_module=_FakeCv2(),
    )
    assert result["ok"] is True
    assert result["probe_backend"] == "cv2"
    assert abs(float(result["duration_s"]) - 10.0) < 1e-6
    assert int(result["width"]) == 1280
    assert int(result["height"]) == 720
