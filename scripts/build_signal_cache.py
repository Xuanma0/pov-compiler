from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.ir.events_v1 import ensure_events_v1
from pov_compiler.perception.object_memory_v0 import build_object_memory_v0
from pov_compiler.schemas import EventV1, ObjectMemoryItemV0, Output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minimal signal cache (events_v1 meta + place/interaction + object_memory)")
    parser.add_argument("--pov-json-dir", required=True, help="Directory containing *_v03_decisions.json")
    parser.add_argument("--out_dir", required=True, help="Output signal cache directory")
    parser.add_argument("--uids-file", default=None, help="Optional uid list file")
    parser.add_argument("--contact-threshold", type=float, default=0.6)
    return parser.parse_args()


def _normalize_uid(text: str) -> str:
    token = str(text).replace("\ufeff", "").strip()
    if not token:
        return ""
    if token.lower().endswith(".mp4"):
        token = token[:-4]
    return token.lower()


def _uid_from_json_path(path: Path) -> str:
    stem = path.stem
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    m = re.search(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", cleaned)
    if m:
        return _normalize_uid(m.group(1))
    return _normalize_uid(cleaned)


def _read_uids(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "")
        if "#" in text:
            text = text.split("#", 1)[0]
        text = text.strip()
        if not text:
            continue
        for token in re.split(r"[,\s]+", text):
            uid = _normalize_uid(token)
            if uid:
                out.append(uid)
    return out


def _discover_jsons(pov_json_dir: Path) -> dict[str, Path]:
    files = sorted(pov_json_dir.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(pov_json_dir.glob("*.json"), key=lambda p: p.name.lower())
    out: dict[str, Path] = {}
    for path in files:
        uid = _uid_from_json_path(path)
        if uid and uid not in out:
            out[uid] = path
    return out


def _load_output(path: Path) -> Output:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if hasattr(Output, "model_validate"):
        try:
            return Output.model_validate(payload)  # type: ignore[attr-defined]
        except Exception:
            if isinstance(payload, dict) and "events_v1" in payload:
                payload = dict(payload)
                payload.pop("events_v1", None)
                return Output.model_validate(payload)  # type: ignore[attr-defined]
            raise
    try:
        return Output.parse_obj(payload)
    except Exception:
        if isinstance(payload, dict) and "events_v1" in payload:
            payload = dict(payload)
            payload.pop("events_v1", None)
            return Output.parse_obj(payload)
        raise


def _event_object_vocab(events: list[EventV1]) -> set[str]:
    out: set[str] = set()
    for ev in events:
        primary = str(getattr(ev, "interaction_primary_object", "") or "").strip().lower()
        if primary:
            out.add(primary)
        sig = getattr(ev, "interaction_signature", {}) or {}
        if isinstance(sig, dict):
            for key in ("active_object_top1", "object_name"):
                value = str(sig.get(key, "")).strip().lower()
                if value:
                    out.add(value)
    return out


def _events_v1_meta(events: list[EventV1]) -> dict[str, Any]:
    evidence_span_count = sum(len(list(getattr(ev, "evidence", []) or [])) for ev in events)
    retrieval_hit_count = sum(1 for ev in events if len(list(getattr(ev, "evidence", []) or [])) > 0)
    object_vocab_size = len(_event_object_vocab(events))
    return {
        "events_v1_count": int(len(events)),
        "evidence_span_count": int(evidence_span_count),
        "retrieval_hit_count": int(retrieval_hit_count),
        "object_vocab_size": int(object_vocab_size),
    }


def _place_interaction_meta(events: list[EventV1]) -> dict[str, Any]:
    place_segments: set[str] = set()
    interaction_events_count = 0
    object_counter: Counter[str] = Counter()
    for ev in events:
        place_id = str(getattr(ev, "place_segment_id", "") or "").strip()
        if place_id:
            place_segments.add(place_id)
        score = float(getattr(ev, "interaction_score", 0.0) or 0.0)
        obj = str(getattr(ev, "interaction_primary_object", "") or "").strip().lower()
        if score > 0.0 or obj:
            interaction_events_count += 1
        if obj:
            object_counter[obj] += 1
    top_objects = [name for name, _ in object_counter.most_common(10)]
    return {
        "place_segments_count": int(len(place_segments)),
        "interaction_events_count": int(interaction_events_count),
        "top_objects": top_objects,
    }


def _ensure_object_memory(output: Output, contact_threshold: float) -> list[ObjectMemoryItemV0]:
    current = list(getattr(output, "object_memory_v0", []) or [])
    if current:
        return current
    built = build_object_memory_v0(
        perception=getattr(output, "perception", {}) or {},
        events_v1=list(getattr(output, "events_v1", []) or []),
        contact_threshold=float(contact_threshold),
    )
    return list(built or [])


def _lost_object_queries_total(items: list[ObjectMemoryItemV0]) -> int:
    total = 0
    for item in items:
        object_name = str(getattr(item, "object_name", "")).strip()
        if not object_name:
            continue
        anchor_contact = getattr(item, "last_contact_t_ms", None)
        anchor_seen = getattr(item, "last_seen_t_ms", None)
        if anchor_contact is None and anchor_seen is None:
            continue
        total += 1
    return int(total)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    pov_json_dir = Path(args.pov_json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    discovered = _discover_jsons(pov_json_dir)
    if not discovered:
        print(f"error=no json found under {pov_json_dir}")
        return 2

    requested: list[str] = []
    selected_uids: list[str]
    missing: list[str] = []
    if args.uids_file:
        requested = _read_uids(Path(args.uids_file))
        selected_uids = [uid for uid in requested if uid in discovered]
        missing = [uid for uid in requested if uid not in discovered]
    else:
        selected_uids = sorted(discovered.keys())

    uid_rows: list[dict[str, Any]] = []
    built_ok = 0
    built_fail = 0
    for uid in selected_uids:
        json_path = discovered[uid]
        uid_dir = out_dir / uid
        uid_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "uid": uid,
            "json_path": str(json_path),
            "status": "ok",
            "missing_reason": [],
            "events_v1_meta": str(uid_dir / "events_v1_meta.json"),
            "place_interaction": str(uid_dir / "place_interaction.json"),
            "object_memory": str(uid_dir / "object_memory.json"),
            "lost_object_queries": str(uid_dir / "lost_object_queries.json"),
        }
        try:
            output = ensure_events_v1(_load_output(json_path))
            events = list(output.events_v1 or [])

            ev_meta = _events_v1_meta(events)
            place_meta = _place_interaction_meta(events)
            object_memory = _ensure_object_memory(output, contact_threshold=float(args.contact_threshold))
            object_rows = [
                x.model_dump() if hasattr(x, "model_dump") else (x.dict() if hasattr(x, "dict") else dict(x))
                for x in object_memory
            ]
            lost_total = _lost_object_queries_total(object_memory)

            _dump_json(uid_dir / "events_v1_meta.json", ev_meta)
            _dump_json(uid_dir / "place_interaction.json", place_meta)
            _dump_json(
                uid_dir / "object_memory.json",
                {"objects_total": len(object_rows), "objects": object_rows},
            )
            _dump_json(
                uid_dir / "lost_object_queries.json",
                {"lost_object_queries_total": int(lost_total), "generated_from": "object_memory_v0"},
            )
            row["events_v1_count"] = int(ev_meta.get("events_v1_count", 0))
            row["place_segments_count"] = int(place_meta.get("place_segments_count", 0))
            row["interaction_events_count"] = int(place_meta.get("interaction_events_count", 0))
            row["objects_total"] = int(len(object_rows))
            row["lost_object_queries_total"] = int(lost_total)
            built_ok += 1
        except Exception as exc:
            row["status"] = "error"
            row["missing_reason"] = [str(exc)]
            built_fail += 1
        uid_rows.append(row)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(pov_json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "contact_threshold": float(args.contact_threshold),
        },
        "selection": {
            "uids_requested": len(requested),
            "uids_found": len(selected_uids),
            "uids_missing_count": len(missing),
            "uids_missing_sample": missing[:10],
        },
        "build": {
            "built_ok": int(built_ok),
            "built_fail": int(built_fail),
        },
        "rows": uid_rows,
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected_uids={len(selected_uids)}")
    print(f"built_ok={built_ok}")
    print(f"built_fail={built_fail}")
    print(f"saved_cache={out_dir}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
