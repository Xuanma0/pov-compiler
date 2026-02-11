from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom <= 1e-12:
        return vec
    return vec / denom


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a.astype(np.float32, copy=False)
    b = vec_b.astype(np.float32, copy=False)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return 0.0
    sim = float(np.dot(a, b) / (a_norm * b_norm))
    sim = float(np.clip(sim, -1.0, 1.0))
    return float(1.0 - sim)


class HistogramEmbedder:
    """Torch-free fallback embedder based on grayscale + color histograms."""

    backend_name = "histogram"

    def __init__(self, image_size: int = 64):
        self.image_size = image_size

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self.image_size, self.image_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1)

        color_hists = []
        for channel in range(3):
            hist = cv2.calcHist([img], [channel], None, [16], [0, 256]).reshape(-1)
            color_hists.append(hist)

        embedding = np.concatenate([gray_hist, *color_hists], axis=0).astype(np.float32)
        return _l2_normalize(embedding)


@dataclass
class CLIPSettings:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "cpu"


class CLIPEmbedder:
    backend_name = "clip"

    def __init__(self, settings: CLIPSettings | None = None):
        cfg = settings or CLIPSettings()
        self._model_name = cfg.model_name
        self._pretrained = cfg.pretrained
        self._device = cfg.device

        import open_clip  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        self._torch = torch
        self._Image = Image
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name,
            pretrained=self._pretrained,
            device=self._device,
        )
        self._model.eval()

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(rgb)
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].detach().cpu().numpy().astype(np.float32)


class Embedder:
    def __init__(
        self,
        use_clip: bool = False,
        clip_settings: CLIPSettings | None = None,
    ):
        self.backend = HistogramEmbedder()
        if use_clip:
            try:
                self.backend = CLIPEmbedder(clip_settings)
            except Exception:
                self.backend = HistogramEmbedder()

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "backend_name", "unknown")

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.backend.embed(frame_bgr)
