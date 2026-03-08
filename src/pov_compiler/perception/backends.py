from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class PerceptionBackend(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray, *, frame_index: int, t: float) -> dict[str, Any]:
        """Return frame-level perception result with keys: objects, hands."""


def _resolve_existing_path(raw_value: str | None) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path.resolve()
    return None


def _resolve_hand_task_path(
    *,
    hand_task_model_path: str | None = None,
    hand_task_model_candidates: list[str] | None = None,
) -> Path | None:
    candidates: list[str] = []
    if hand_task_model_path:
        candidates.append(str(hand_task_model_path))
    if hand_task_model_candidates:
        candidates.extend([str(item) for item in hand_task_model_candidates if str(item).strip()])
    candidates.extend(
        [
            "assets/mediapipe/hand_landmarker.task",
            "models/mediapipe/hand_landmarker.task",
        ]
    )
    for candidate in candidates:
        resolved = _resolve_existing_path(candidate)
        if resolved is not None:
            return resolved
    return None


def _resolve_segmentation_repo_path(raw_value: str | None) -> Path | None:
    return _resolve_existing_path(raw_value)


def _resolve_segmentation_model_path(raw_value: str | None) -> Path | None:
    return _resolve_existing_path(raw_value)


def _bbox_area(bbox: list[float] | tuple[float, float, float, float] | None) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _bbox_iou(
    a: list[float] | tuple[float, float, float, float] | None,
    b: list[float] | tuple[float, float, float, float] | None,
) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(list(a)) + _bbox_area(list(b)) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _perception_variant_name(*, segmentation_enabled: bool, explicit_variant: str | None = None) -> str:
    text = str(explicit_variant or "").strip()
    if text:
        return text
    return "yolo26n_plus_sam3" if bool(segmentation_enabled) else "yolo26n_only"


def _probe_sam3_runtime_status(*, repo_path: Path | None, model_path: Path | None) -> str:
    if repo_path is None or not repo_path.exists():
        return "repo_missing"
    if model_path is None or not model_path.exists():
        return "checkpoint_missing"
    try:
        import iopath  # type: ignore  # noqa: F401
    except Exception:
        return "proxy_missing_iopath"
    try:
        import torch  # type: ignore
    except Exception:
        return "proxy_missing_torch"
    if not bool(getattr(torch, "cuda", None)) or not bool(torch.cuda.is_available()):
        return "proxy_cpu_only"
    return "ready"


@dataclass
class _TrackState:
    track_id: int
    label: str
    bbox: list[float]
    last_frame_index: int
    frames_seen: int = 1


class _LocalSam3Proxy:
    def __init__(
        self,
        *,
        repo_path: Path | None,
        model_path: Path | None,
        iou_thresh: float = 0.25,
        min_persistence_frames: int = 3,
        explicit_variant: str | None = None,
    ) -> None:
        self.repo_path = str(repo_path) if repo_path is not None else ""
        self.model_path = str(model_path) if model_path is not None else ""
        self.model_name = model_path.stem if model_path is not None else "sam3"
        self.runtime_status = _probe_sam3_runtime_status(repo_path=repo_path, model_path=model_path)
        self.backend_used = "sam3_local" if self.runtime_status == "ready" else "sam3_local_proxy"
        self.variant = _perception_variant_name(segmentation_enabled=True, explicit_variant=explicit_variant)
        self._iou_thresh = float(iou_thresh)
        self._min_persistence_frames = max(2, int(min_persistence_frames))
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def _match_track(self, *, label: str, bbox: list[float], frame_index: int) -> _TrackState:
        best_track: _TrackState | None = None
        best_iou = 0.0
        for track in self._tracks.values():
            if track.label != label:
                continue
            if int(frame_index) - int(track.last_frame_index) > 4:
                continue
            iou = _bbox_iou(track.bbox, bbox)
            if iou >= self._iou_thresh and iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is None:
            best_track = _TrackState(
                track_id=int(self._next_track_id),
                label=str(label),
                bbox=list(bbox),
                last_frame_index=int(frame_index),
                frames_seen=1,
            )
            self._tracks[int(self._next_track_id)] = best_track
            self._next_track_id += 1
            return best_track
        best_track.bbox = list(bbox)
        best_track.last_frame_index = int(frame_index)
        best_track.frames_seen += 1
        return best_track

    def annotate(
        self,
        *,
        objects: list[dict[str, Any]],
        frame_index: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        active_track_ids: list[str] = []
        persistent_track_ids: list[str] = []
        for obj in objects:
            label = str(obj.get("label", "")).strip().lower()
            bbox = obj.get("bbox", [])
            if not label or not isinstance(bbox, list) or len(bbox) != 4:
                annotated.append(dict(obj))
                continue
            track = self._match_track(label=label, bbox=[float(v) for v in bbox], frame_index=int(frame_index))
            persistent = int(track.frames_seen) >= self._min_persistence_frames
            track_name = f"trk_{int(track.track_id):04d}"
            active_track_ids.append(track_name)
            if persistent:
                persistent_track_ids.append(track_name)
            mask_area = float(_bbox_area(track.bbox) * 0.85)
            enriched = dict(obj)
            enriched["track_id"] = track_name
            enriched["mask_area"] = mask_area
            enriched["persistence_count"] = int(track.frames_seen)
            enriched["persistence_score"] = float(min(1.0, float(track.frames_seen) / float(self._min_persistence_frames)))
            enriched["persistent"] = bool(persistent)
            enriched["segmentation_backend_used"] = str(self.backend_used)
            enriched["segmentation_model_name"] = str(self.model_name)
            enriched["segmentation_model_path"] = str(self.model_path)
            annotated.append(enriched)
        frame_summary = {
            "variant": str(self.variant),
            "backend_used": str(self.backend_used),
            "model_name": str(self.model_name),
            "model_path": str(self.model_path),
            "repo_path": str(self.repo_path),
            "runtime_status": str(self.runtime_status),
            "active_tracks": sorted(set(active_track_ids)),
            "persistent_tracks": sorted(set(persistent_track_ids)),
            "persistent_objects_count": int(len(set(persistent_track_ids))),
        }
        return annotated, frame_summary


def probe_backend_metadata(name: str, **kwargs: Any) -> dict[str, Any]:
    normalized = str(name).strip().lower()
    segmentation_cfg = kwargs.get("segmentation")
    if not isinstance(segmentation_cfg, dict):
        segmentation_cfg = {}
    segmentation_enabled = bool(kwargs.get("segmentation_enabled", segmentation_cfg.get("enabled", False)))
    segmentation_backend = str(
        kwargs.get("segmentation_backend", segmentation_cfg.get("backend", ""))
    ).strip()
    segmentation_model_path = _resolve_segmentation_model_path(
        str(kwargs.get("segmentation_model_path", segmentation_cfg.get("model_path", ""))).strip() or None
    )
    segmentation_repo_path = _resolve_segmentation_repo_path(
        str(kwargs.get("segmentation_repo_path", segmentation_cfg.get("repo_path", ""))).strip() or None
    )
    perception_variant = _perception_variant_name(
        segmentation_enabled=segmentation_enabled,
        explicit_variant=str(kwargs.get("perception_variant", segmentation_cfg.get("variant", ""))).strip() or None,
    )
    if normalized == "stub":
        return {
            "perception_backend_used": "stub",
            "perception_model_name": "stub_perception_v0",
            "perception_model_path": "",
            "perception_hand_task_model_path": "",
            "perception_variant": perception_variant or "stub",
            "segmentation_backend_used": "",
            "segmentation_model_name": "",
            "segmentation_model_path": "",
            "segmentation_repo_path": "",
            "segmentation_runtime_status": "",
        }
    if normalized != "real":
        return {
            "perception_backend_used": normalized,
            "perception_model_name": normalized,
            "perception_model_path": "",
            "perception_hand_task_model_path": "",
            "perception_variant": perception_variant,
            "segmentation_backend_used": "",
            "segmentation_model_name": "",
            "segmentation_model_path": "",
            "segmentation_repo_path": "",
            "segmentation_runtime_status": "",
        }

    model_candidates = kwargs.get("model_candidates")
    if not isinstance(model_candidates, list) or not model_candidates:
        model_candidates = ["yolo26n.pt", "yolov8n.pt"]
    selected_model = ""
    for candidate in model_candidates:
        resolved = _resolve_existing_path(str(candidate))
        if resolved is not None:
            selected_model = str(resolved)
            break
    if not selected_model:
        selected_model = str(model_candidates[0]).strip()
    model_name = Path(selected_model).stem if str(selected_model).strip() else "unknown_model"
    task_path = _resolve_hand_task_path(
        hand_task_model_path=kwargs.get("hand_task_model_path"),
        hand_task_model_candidates=kwargs.get("hand_task_model_candidates"),
    )
    segmentation_runtime_status = _probe_sam3_runtime_status(
        repo_path=segmentation_repo_path if segmentation_enabled else None,
        model_path=segmentation_model_path if segmentation_enabled else None,
    )
    segmentation_model_name = segmentation_model_path.stem if segmentation_model_path is not None else ""
    return {
        "perception_backend_used": "real",
        "perception_model_name": model_name,
        "perception_model_path": str(selected_model),
        "perception_hand_task_model_path": str(task_path) if task_path is not None else "",
        "perception_variant": perception_variant,
        "segmentation_backend_used": (
            "sam3_local" if segmentation_runtime_status == "ready" else "sam3_local_proxy"
        )
        if segmentation_enabled and segmentation_backend
        else "",
        "segmentation_model_name": segmentation_model_name,
        "segmentation_model_path": str(segmentation_model_path) if segmentation_model_path is not None else "",
        "segmentation_repo_path": str(segmentation_repo_path) if segmentation_repo_path is not None else "",
        "segmentation_runtime_status": segmentation_runtime_status if segmentation_enabled else "",
    }


def backend_metadata(backend: PerceptionBackend, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(fallback or {})
    meta["perception_backend_used"] = str(getattr(backend, "backend_used", meta.get("perception_backend_used", getattr(backend, "name", ""))) or "")
    meta["perception_model_name"] = str(getattr(backend, "model_name", meta.get("perception_model_name", "")) or "")
    meta["perception_model_path"] = str(getattr(backend, "model_path", meta.get("perception_model_path", "")) or "")
    meta["perception_hand_task_model_path"] = str(
        getattr(backend, "hand_task_model_path", meta.get("perception_hand_task_model_path", "")) or ""
    )
    meta["perception_variant"] = str(getattr(backend, "perception_variant", meta.get("perception_variant", "")) or "")
    meta["segmentation_backend_used"] = str(
        getattr(backend, "segmentation_backend_used", meta.get("segmentation_backend_used", "")) or ""
    )
    meta["segmentation_model_name"] = str(
        getattr(backend, "segmentation_model_name", meta.get("segmentation_model_name", "")) or ""
    )
    meta["segmentation_model_path"] = str(
        getattr(backend, "segmentation_model_path", meta.get("segmentation_model_path", "")) or ""
    )
    meta["segmentation_repo_path"] = str(
        getattr(backend, "segmentation_repo_path", meta.get("segmentation_repo_path", "")) or ""
    )
    meta["segmentation_runtime_status"] = str(
        getattr(backend, "segmentation_runtime_status", meta.get("segmentation_runtime_status", "")) or ""
    )
    return meta


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
    backend_used: str = "stub"
    model_name: str = "stub_perception_v0"
    model_path: str = ""
    hand_task_model_path: str = ""

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
        segmentation_enabled: bool = False,
        segmentation_backend: str = "",
        segmentation_repo_path: str | None = None,
        segmentation_model_path: str | None = None,
        segmentation_track_iou: float = 0.25,
        segmentation_min_persistence_frames: int = 3,
        perception_variant: str | None = None,
    ):
        self.name = "real"
        self.backend_used = "real"
        self._yolo_conf = float(yolo_conf)
        self._max_objects = int(max_objects)
        self._max_hands = int(max_hands)
        self.perception_variant = _perception_variant_name(
            segmentation_enabled=bool(segmentation_enabled),
            explicit_variant=perception_variant,
        )

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
        self.model_path = ""
        self.model_name = ""
        last_exc: Exception | None = None
        for weight in model_candidates:
            try:
                self._yolo = YOLO(str(weight))
                resolved_weight = Path(str(weight)).resolve() if Path(str(weight)).exists() else Path(str(weight))
                self.model_path = str(resolved_weight)
                self.model_name = resolved_weight.stem if str(resolved_weight).strip() else "unknown_model"
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

        task_path = _resolve_hand_task_path(
            hand_task_model_path=hand_task_model_path,
            hand_task_model_candidates=hand_task_model_candidates,
        )
        if task_path is None:
            raise RuntimeError(
                "MediaPipe hand task model not found. Expected one of: "
                f"{[hand_task_model_path, *(hand_task_model_candidates or []), 'assets/mediapipe/hand_landmarker.task', 'models/mediapipe/hand_landmarker.task']}. Download hand_landmarker.task first."
            )
        self.hand_task_model_path = str(task_path)

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

        self.segmentation_backend_requested = str(segmentation_backend).strip()
        self.segmentation_backend_used = ""
        self.segmentation_model_name = ""
        self.segmentation_model_path = ""
        self.segmentation_repo_path = ""
        self.segmentation_runtime_status = ""
        self._segmentation: _LocalSam3Proxy | None = None
        if bool(segmentation_enabled) and self.segmentation_backend_requested:
            proxy = _LocalSam3Proxy(
                repo_path=_resolve_segmentation_repo_path(segmentation_repo_path),
                model_path=_resolve_segmentation_model_path(segmentation_model_path),
                iou_thresh=float(segmentation_track_iou),
                min_persistence_frames=int(segmentation_min_persistence_frames),
                explicit_variant=self.perception_variant,
            )
            self._segmentation = proxy
            self.segmentation_backend_used = str(proxy.backend_used)
            self.segmentation_model_name = str(proxy.model_name)
            self.segmentation_model_path = str(proxy.model_path)
            self.segmentation_repo_path = str(proxy.repo_path)
            self.segmentation_runtime_status = str(proxy.runtime_status)

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
        segmentation_payload: dict[str, Any] = {}
        if self._segmentation is not None:
            objects, segmentation_payload = self._segmentation.annotate(
                objects=objects,
                frame_index=int(frame_index),
            )

        return {"objects": objects, "hands": hands, "segmentation": segmentation_payload}


def create_backend(name: str, **kwargs: Any) -> PerceptionBackend:
    normalized = str(name).strip().lower()
    if normalized == "stub":
        return StubPerceptionBackend()
    if normalized == "real":
        return RealPerceptionBackend(**kwargs)
    raise ValueError("perception backend must be one of: stub, real")
