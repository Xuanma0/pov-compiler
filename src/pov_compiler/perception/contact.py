from __future__ import annotations

from typing import Any

import numpy as np


def _bbox_area(bbox: list[float] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: list[float] | tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = max(1e-9, _bbox_area(a) + _bbox_area(b) - inter)
    return float(inter / union)


def _point_to_bbox_distance(px: float, py: float, bbox: list[float] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    return float(np.hypot(px - cx, py - cy))


def _hand_contact_points(hand: dict[str, Any]) -> list[tuple[float, float]]:
    landmarks = hand.get("landmarks", [])
    points: list[tuple[float, float]] = []
    if isinstance(landmarks, list) and landmarks:
        tip_indices = [4, 8, 12, 16, 20]
        for idx in tip_indices:
            if 0 <= idx < len(landmarks):
                item = landmarks[idx]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    points.append((float(item[0]), float(item[1])))
    if points:
        return points
    bbox = hand.get("bbox", [0.0, 0.0, 0.0, 0.0])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [((x1 + x2) * 0.5, (y1 + y2) * 0.5)]


def contact_score(
    hand: dict[str, Any],
    obj: dict[str, Any],
    *,
    frame_diag: float,
    iou_weight: float = 0.35,
    dist_weight: float = 0.65,
) -> float:
    hand_bbox = hand.get("bbox", [0.0, 0.0, 0.0, 0.0])
    obj_bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
    iou = bbox_iou(hand_bbox, obj_bbox)
    points = _hand_contact_points(hand)
    min_dist = min(_point_to_bbox_distance(px, py, obj_bbox) for px, py in points)
    norm_d = min(1.0, float(min_dist) / max(1.0, float(frame_diag) * 0.08))
    proximity = 1.0 - norm_d
    score = float(iou_weight) * float(iou) + float(dist_weight) * float(proximity)
    return float(max(0.0, min(1.0, score)))


def select_active_contact(
    *,
    hands: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    frame_shape: tuple[int, int, int] | tuple[int, int],
    t: float,
    min_score: float = 0.25,
) -> dict[str, Any]:
    if len(frame_shape) >= 2:
        h = int(frame_shape[0])
        w = int(frame_shape[1])
    else:
        h, w = 1, 1
    frame_diag = float(np.hypot(max(1, w), max(1, h)))
    candidates: list[dict[str, Any]] = []
    for hand in hands:
        for obj in objects:
            score = contact_score(hand, obj, frame_diag=frame_diag)
            if score < float(min_score):
                continue
            candidates.append(
                {
                    "object_id": str(obj.get("id", "")),
                    "label": str(obj.get("label", "")),
                    "hand_id": str(hand.get("id", "")),
                    "score": float(score),
                    "handedness": str(hand.get("handedness", "unknown")),
                    "t": float(t),
                }
            )
    candidates.sort(key=lambda x: (-float(x["score"]), str(x["object_id"]), str(x["hand_id"])))
    active = candidates[0] if candidates else None
    return {
        "active": active,
        "candidates": candidates,
        "active_score": float(active["score"]) if active else 0.0,
    }

