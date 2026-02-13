from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_VALID_CATEGORIES = {"metric", "scenario", "tool"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ByeEventV1:
    tsMs: int
    category: str
    name: str
    payload: dict[str, Any]
    frameSeq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "tsMs": int(self.tsMs),
            "category": str(self.category),
            "name": str(self.name),
            "payload": dict(self.payload),
        }
        if self.frameSeq is not None:
            data["frameSeq"] = int(self.frameSeq)
        return data


def validate_minimal(event_dict: dict[str, Any]) -> None:
    if not isinstance(event_dict, dict):
        raise ValueError("event must be a dict")

    if "tsMs" not in event_dict or not _is_int(event_dict["tsMs"]):
        raise ValueError("event.tsMs must be int")
    if "category" not in event_dict or not isinstance(event_dict["category"], str):
        raise ValueError("event.category must be str")
    if event_dict["category"] not in _VALID_CATEGORIES:
        raise ValueError("event.category must be one of metric|scenario|tool")
    if "name" not in event_dict or not isinstance(event_dict["name"], str):
        raise ValueError("event.name must be str")
    if "payload" not in event_dict or not isinstance(event_dict["payload"], dict):
        raise ValueError("event.payload must be dict")

    payload = event_dict["payload"]
    if "video_id" not in payload or not isinstance(payload["video_id"], str):
        raise ValueError("event.payload.video_id must be str")
    if "t0_ms" not in payload or not _is_int(payload["t0_ms"]):
        raise ValueError("event.payload.t0_ms must be int")
    if "t1_ms" not in payload or not _is_int(payload["t1_ms"]):
        raise ValueError("event.payload.t1_ms must be int")

