from __future__ import annotations

import cv2
import numpy as np


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _downscale_gray(gray: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return gray
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_motion_energy(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Motion signal for two consecutive sampled frames.
    Prefers Farneback optical flow; falls back to mean absolute frame diff.
    """
    prev_small = _downscale_gray(prev_gray)
    gray_small = _downscale_gray(gray)

    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_small,
            gray_small,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(magnitude))
    except Exception:
        diff = cv2.absdiff(prev_small, gray_small)
        return float(np.mean(diff.astype(np.float32)))
