from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pov_compiler.integrations.bye.schema import ByeEventV1, validate_minimal

_NAME_PRIORITY = {
    "pov.event": 0,
    "pov.highlight": 1,
    "pov.decision": 2,
    "pov.token": 3,
}


def _to_int_ms(seconds: Any) -> int:
    if isinstance(seconds, bool):
        return 0
    if isinstance(seconds, (int, float)):
        return int(round(float(seconds) * 1000.0))
    return 0


def _extract_span_ms(item: dict[str, Any]) -> tuple[int, int]:
    if isinstance(item.get("t0_ms"), int) and isinstance(item.get("t1_ms"), int):
        return int(item["t0_ms"]), int(item["t1_ms"])

    t0_ms = _to_int_ms(item.get("t0"))
    t1_ms = _to_int_ms(item.get("t1"))
    if t1_ms < t0_ms:
        t1_ms = t0_ms
    return t0_ms, t1_ms


def _extract_ts_ms(item: dict[str, Any], *, t0_ms: int, t1_ms: int, ts_strategy: str) -> int:
    if isinstance(item.get("tsMs"), int):
        return int(item["tsMs"])
    strategy = str(ts_strategy or "t0").strip().lower()
    if strategy == "t1":
        return t1_ms
    if strategy == "mid":
        return int((t0_ms + t1_ms) // 2)
    return t0_ms


def _resolve_video_id(output_dict: dict[str, Any], video_id: str | None) -> str:
    if video_id:
        return str(video_id)
    value = output_dict.get("video_id")
    if isinstance(value, str) and value:
        return value
    return "unknown_video"


def _base_payload(
    *,
    output_video_id: str,
    item: dict[str, Any],
    source_kind: str,
    source_event: str | None,
    conf: float | None,
    t0_ms: int,
    t1_ms: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "video_id": output_video_id,
        "t0_ms": int(t0_ms),
        "t1_ms": int(t1_ms),
        "source_kind": str(source_kind),
        "source_event": str(source_event or ""),
        "conf": None if conf is None else float(conf),
    }
    if "id" in item:
        payload["source_id"] = str(item.get("id", ""))
    return payload


def _safe_conf(item: dict[str, Any], fallback: float | None = None) -> float | None:
    conf = item.get("conf", fallback)
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return float(conf)
    return None


def _event_conf(item: dict[str, Any]) -> float | None:
    conf = _safe_conf(item, None)
    if conf is not None:
        return conf
    scores = item.get("scores")
    if isinstance(scores, dict):
        for key in ("boundary_conf", "score", "conf"):
            val = scores.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
    return None


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]


def _pick_events_source(output_dict: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    events_v1 = _as_list(output_dict.get("events_v1"))
    if events_v1:
        return "events_v1", events_v1
    events_v0 = _as_list(output_dict.get("events_v0"))
    if events_v0:
        return "events_v0", events_v0
    events = _as_list(output_dict.get("events"))
    if events:
        return "events", events
    return "events", []


def export_bye_events_from_output_dict(
    output_dict: dict[str, Any],
    video_id: str | None = None,
    *,
    include: tuple[str, ...] = ("events_v1", "highlights", "tokens", "decisions"),
    ts_strategy: str = "t0",
    sort: bool = True,
) -> list[dict[str, Any]]:
    include_set = {str(x).strip() for x in include if str(x).strip()}
    output_video_id = _resolve_video_id(output_dict, video_id)
    emitted: list[tuple[dict[str, Any], int, int, int]] = []
    stable_idx = 0

    def emit(name: str, item: dict[str, Any], source_kind: str, payload: dict[str, Any], t0_ms: int, t1_ms: int) -> None:
        nonlocal stable_idx
        ts_ms = _extract_ts_ms(item, t0_ms=t0_ms, t1_ms=t1_ms, ts_strategy=ts_strategy)
        frame_seq = item.get("frameSeq")
        if not isinstance(frame_seq, int):
            frame_seq = item.get("frame_seq")
        model = ByeEventV1(
            tsMs=int(ts_ms),
            category="scenario",
            name=name,
            payload=payload,
            frameSeq=int(frame_seq) if isinstance(frame_seq, int) else None,
        )
        row = model.to_dict()
        validate_minimal(row)
        emitted.append((row, int(t0_ms), int(t1_ms), int(stable_idx)))
        stable_idx += 1

    if "events_v1" in include_set or "events" in include_set:
        source_kind, events_rows = _pick_events_source(output_dict)
        for item in events_rows:
            t0_ms, t1_ms = _extract_span_ms(item)
            payload = _base_payload(
                output_video_id=output_video_id,
                item=item,
                source_kind=source_kind,
                source_event=str(item.get("id", "")),
                conf=_event_conf(item),
                t0_ms=t0_ms,
                t1_ms=t1_ms,
            )
            payload["label"] = str(item.get("label", ""))
            payload["source_event_ids"] = list(item.get("source_event_ids", [])) if isinstance(item.get("source_event_ids"), list) else []
            emit("pov.event", item, source_kind, payload, t0_ms, t1_ms)

    if "highlights" in include_set:
        for item in _as_list(output_dict.get("highlights")):
            t0_ms, t1_ms = _extract_span_ms(item)
            payload = _base_payload(
                output_video_id=output_video_id,
                item=item,
                source_kind="highlight",
                source_event=str(item.get("source_event", "")),
                conf=_safe_conf(item),
                t0_ms=t0_ms,
                t1_ms=t1_ms,
            )
            payload["anchor_type"] = str(item.get("anchor_type", ""))
            payload["anchor_t_ms"] = _to_int_ms(item.get("anchor_t"))
            emit("pov.highlight", item, "highlight", payload, t0_ms, t1_ms)

    if "tokens" in include_set:
        token_codec = output_dict.get("token_codec")
        token_rows = _as_list(token_codec.get("tokens")) if isinstance(token_codec, dict) else []
        for item in token_rows:
            t0_ms, t1_ms = _extract_span_ms(item)
            payload = _base_payload(
                output_video_id=output_video_id,
                item=item,
                source_kind="token",
                source_event=str(item.get("source_event", "")),
                conf=_safe_conf(item),
                t0_ms=t0_ms,
                t1_ms=t1_ms,
            )
            payload["token_type"] = str(item.get("type", ""))
            payload["source"] = dict(item.get("source", {})) if isinstance(item.get("source"), dict) else {}
            emit("pov.token", item, "token", payload, t0_ms, t1_ms)

    if "decisions" in include_set:
        for item in _as_list(output_dict.get("decision_points")):
            t0_ms, t1_ms = _extract_span_ms(item)
            if t0_ms == 0 and t1_ms == 0 and isinstance(item.get("t"), (int, float)):
                t0_ms = _to_int_ms(item.get("t"))
                t1_ms = t0_ms
            payload = _base_payload(
                output_video_id=output_video_id,
                item=item,
                source_kind="decision",
                source_event=str(item.get("source_event", "")),
                conf=_safe_conf(item),
                t0_ms=t0_ms,
                t1_ms=t1_ms,
            )
            core_fields = ("trigger", "state", "action", "constraints", "outcome")
            found_core = False
            for key in core_fields:
                if key in item:
                    payload[key] = item.get(key)
                    found_core = True
            if not found_core:
                payload["raw"] = dict(item)
            emit("pov.decision", item, "decision", payload, t0_ms, t1_ms)

    if not sort:
        return [row for row, _, _, _ in emitted]

    def sort_key(entry: tuple[dict[str, Any], int, int, int]) -> tuple[int, int, int, int, str, int]:
        row, t0_ms, t1_ms, idx = entry
        name = str(row.get("name", ""))
        return (
            int(row.get("tsMs", 0)),
            int(_NAME_PRIORITY.get(name, 99)),
            int(t0_ms),
            int(t1_ms),
            name,
            int(idx),
        )

    emitted.sort(key=sort_key)
    return [row for row, _, _, _ in emitted]


def write_jsonl(events: list[dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in events:
            validate_minimal(row)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")

