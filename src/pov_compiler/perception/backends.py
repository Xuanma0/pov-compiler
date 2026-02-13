from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class PerceptionBackend(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray, *, frame_index: int, t: float) -> dict[str, Any]:
        """Return frame-level perception result with keys: objects, hands."""


def _clamp_bbox(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> list[float]:
    x1 = max(0.0, min(float(w - 1), float(x1)))
    y1 = max(0.0, min(float(h - 1), float(y1)))
    x2 = max(0.0, min(float(w - 1), float(x2)))
    y2 = max(0.0, min(float(h - 1), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [float(x1), float(y1), float(x2), float(y2)]


@dataclass
class StubPerceptionBackend:
    name: str = "stub"
    object_label: str = "cup"
    hand_label: str = "right"

    def detect(self, frame_bgr: np.ndarray, *, frame_index: int, t: float) -> dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        cx = int((0.45 + 0.15 * np.sin(float(frame_index) * 0.25)) * w)
        cy = int((0.55 + 0.08 * np.cos(float(frame_index) * 0.21)) * h)
        bw = max(24, int(w * 0.18))
        bh = max(24, int(h * 0.16))
        obj_bbox = _clamp_bbox(cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2, w=w, h=h)

        hand_shift = int(max(8, w * 0.04))
        hand_bbox = _clamp_bbox(
            obj_bbox[0] - hand_shift,
            obj_bbox[1] - hand_shift * 0.5,
            obj_bbox[0] + hand_shift * 1.2,
            obj_bbox[3] + hand_shift * 0.3,
            w=w,
            h=h,
        )
        # Landmarks loosely around the hand bbox, fingertips near object bbox.
        hx1, hy1, hx2, hy2 = hand_bbox
        fingertips = [
            [obj_bbox[0] + 2.0, obj_bbox[1] + 2.0],
            [obj_bbox[0] + 4.0, obj_bbox[1] + 3.0],
            [obj_bbox[0] + 6.0, obj_bbox[1] + 4.0],
            [obj_bbox[0] + 8.0, obj_bbox[1] + 5.0],
            [obj_bbox[0] + 10.0, obj_bbox[1] + 6.0],
        ]
        palm = [
            [hx1 + 4.0, hy1 + 4.0],
            [hx2 - 4.0, hy1 + 4.0],
            [hx2 - 4.0, hy2 - 4.0],
            [hx1 + 4.0, hy2 - 4.0],
        ]
        landmarks = palm + fingertips

        return {
            "objects": [
                {
                    "id": f"obj_{frame_index:06d}_0",
                    "label": self.object_label,
                    "conf": 0.95,
                    "bbox": obj_bbox,
                }
            ],
            "hands": [
                {
                    "id": f"hand_{frame_index:06d}_0",
                    "handedness": self.hand_label,
                    "conf": 0.9,
                    "bbox": hand_bbox,
                    "landmarks": landmarks,
                }
            ],
        }


class RealPerceptionBackend:
    def __init__(
        self,
        *,
        model_candidates: list[str] | None = None,
        hand_task_model_path: str | None = None,
        hand_task_model_candidates: list[str] | None = None,
        yolo_conf: float = 0.25,
        max_objects: int = 24,
        max_hands: int = 2,
        hand_detection_conf: float = 0.35,
        hand_presence_conf: float = 0.35,
        hand_tracking_conf: float = 0.35,
    ):
        self.name = "real"
        self._yolo_conf = float(yolo_conf)
        self._max_objects = int(max_objects)
        self._max_hands = int(max_hands)

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Perception real backend requires ultralytics. Install with: pip install ultralytics"
            ) from exc
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Perception real backend requires mediapipe. Install with: pip install mediapipe"
            ) from exc

        if model_candidates is None:
            model_candidates = ["yolo26n.pt", "yolov8n.pt"]
        self._yolo = None
        last_exc: Exception | None = None
        for weight in model_candidates:
            try:
                self._yolo = YOLO(str(weight))
                break
            except Exception as exc:
                last_exc = exc
                continue
        if self._yolo is None:
            raise RuntimeError(
                f"Failed to load YOLO model from candidates={model_candidates}. "
                f"Please ensure weights are available/downloadable."
            ) from last_exc
        self._mp = mp
        self._mp_image_cls = getattr(mp, "Image", None)
        self._mp_image_format = getattr(mp, "ImageFormat", None)
        if self._mp_image_cls is None or self._mp_image_format is None:
            raise RuntimeError("mediapipe Image API is unavailable in current installation")

        tasks_mod = getattr(mp, "tasks", None)
        vision_mod = getattr(tasks_mod, "vision", None) if tasks_mod is not None else None
        base_options_cls = getattr(tasks_mod, "BaseOptions", None) if tasks_mod is not None else None
        if tasks_mod is None or vision_mod is None or base_options_cls is None:
            raise RuntimeError("mediapipe Tasks API is unavailable; require mediapipe.tasks.vision")

        task_candidates: list[str] = []
        if hand_task_model_path:
            task_candidates.append(str(hand_task_model_path))
        if hand_task_model_candidates:
            task_candidates.extend([str(x) for x in hand_task_model_candidates if str(x).strip()])
        task_candidates.extend(
            [
                "assets/mediapipe/hand_landmarker.task",
                "models/mediapipe/hand_landmarker.task",
            ]
        )
        task_path: Path | None = None
        for candidate in task_candidates:
            p = Path(candidate)
            if p.exists():
                task_path = p
                break
        if task_path is None:
            raise RuntimeError(
                "MediaPipe hand task model not found. Expected one of: "
                f"{task_candidates}. Download hand_landmarker.task first."
            )

        try:
            options = vision_mod.HandLandmarkerOptions(
                base_options=base_options_cls(model_asset_path=str(task_path)),
                running_mode=vision_mod.RunningMode.IMAGE,
                num_hands=self._max_hands,
                min_hand_detection_confidence=float(hand_detection_conf),
                min_hand_presence_confidence=float(hand_presence_conf),
                min_tracking_confidence=float(hand_tracking_conf),
            )
            self._hands = vision_mod.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError("Failed to initialize MediaPipe HandLandmarker (Tasks API)") from exc

    def detect(self, frame_bgr: np.ndarray, *, frame_index: int, t: float) -> dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        objects: list[dict[str, Any]] = []
        hands: list[dict[str, Any]] = []

        yolo_out = self._yolo.predict(frame_bgr, verbose=False, conf=self._yolo_conf, device="cpu")
        if yolo_out:
            result = yolo_out[0]
            names = result.names if hasattr(result, "names") else {}
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
                n = min(len(xyxy), self._max_objects)
                for i in range(n):
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                    cls_idx = int(clss[i]) if i < len(clss) else -1
                    label = str(names.get(cls_idx, f"class_{cls_idx}")) if isinstance(names, dict) else str(cls_idx)
                    objects.append(
                        {
                            "id": f"obj_{frame_index:06d}_{i}",
                            "label": label,
                            "conf": float(confs[i]) if i < len(confs) else 0.0,
                            "bbox": _clamp_bbox(x1, y1, x2, y2, w=w, h=h),
                        }
                    )

        rgb = frame_bgr[:, :, ::-1]
        mp_image = self._mp_image_cls(image_format=self._mp_image_format.SRGB, data=rgb)
        hand_result = self._hands.detect(mp_image)

        landmarks_groups = getattr(hand_result, "hand_landmarks", []) or []
        handedness_groups = getattr(hand_result, "handedness", []) or []

        for i, hand_lm in enumerate(landmarks_groups[: self._max_hands]):
            pts: list[list[float]] = []
            xs: list[float] = []
            ys: list[float] = []

            for lm in hand_lm:
                x_val = getattr(lm, "x", None)
                y_val = getattr(lm, "y", None)
                if x_val is None or y_val is None:
                    continue
                px = float(x_val) * float(w)
                py = float(y_val) * float(h)
                pts.append([px, py])
                xs.append(px)
                ys.append(py)
            if not xs or not ys:
                continue

            handedness = "unknown"
            conf = 0.0
            if i < len(handedness_groups):
                cats = handedness_groups[i] or []
                if cats:
                    top_cat = cats[0]
                    handedness = str(
                        getattr(top_cat, "category_name", None)
                        or getattr(top_cat, "display_name", None)
                        or "unknown"
                    ).lower()
                    conf = float(getattr(top_cat, "score", 0.0))

            hands.append(
                {
                    "id": f"hand_{frame_index:06d}_{i}",
                    "handedness": handedness,
                    "conf": conf,
                    "bbox": _clamp_bbox(min(xs), min(ys), max(xs), max(ys), w=w, h=h),
                    "landmarks": pts,
                }
            )

        return {"objects": objects, "hands": hands}


def create_backend(name: str, **kwargs: Any) -> PerceptionBackend:
    normalized = str(name).strip().lower()
    if normalized == "stub":
        return StubPerceptionBackend()
    if normalized == "real":
        return RealPerceptionBackend(**kwargs)
    raise ValueError("perception backend must be one of: stub, real")
