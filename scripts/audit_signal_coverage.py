from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit signal coverage for POV json outputs and rank UID candidates")
    parser.add_argument("--pov-json-dir", required=True, help="Directory containing *_v03_decisions.json")
    parser.add_argument("--out-dir", required=True, help="Output directory for coverage artifacts")
    parser.add_argument("--uids-file", default=None, help="Optional UID list file")
    parser.add_argument("--perception-dir", default=None, help="Optional perception output directory")
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


def _read_uids_file(path: Path) -> list[str]:
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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _extract_events_v1(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events_v1 = payload.get("events_v1", [])
    if isinstance(events_v1, list):
        return [x for x in events_v1 if isinstance(x, dict)]
    if isinstance(events_v1, dict):
        cand = events_v1.get("events", [])
        if isinstance(cand, list):
            return [x for x in cand if isinstance(x, dict)]
    return []


def _extract_object_memory(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("object_memory_v0", [])
    if isinstance(value, dict):
        cand = value.get("items", value.get("objects", []))
        if isinstance(cand, list):
            return [x for x in cand if isinstance(x, dict)]
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def _extract_lost_object_queries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("lost_object_queries", [])
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def _extract_chain_metrics(uid: str, pov_json_dir: Path) -> tuple[float | None, float | None]:
    parent = pov_json_dir.parent
    candidates = [
        parent / "nlq_chain" / uid / "table_chain_summary.csv",
        parent / "nlq" / uid / "table_chain_summary.csv",
        parent / "nlq_chain" / "table_chain_summary.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            row = rows[0]
            total = _to_float(row.get("chain_queries_total"), default=0.0)
            succ = _to_float(row.get("chain_success_rate"), default=0.0)
            return total, succ
        except Exception:
            continue
    return None, None


def _extract_perception_summary(uid: str, perception_dir: Path | None) -> tuple[float | None, float | None]:
    if perception_dir is None:
        return None, None
    path = perception_dir / uid / "perception.json"
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    frames = _to_float(summary.get("frames_processed"), default=0.0)
    contacts = _to_float(
        summary.get("contact_events_count", summary.get("contact_events", summary.get("contact_events_total", 0.0))),
        default=0.0,
    )
    return frames, contacts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# Signal Coverage\n\nNo rows.\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [
        "# Signal Coverage",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    pov_json_dir = Path(args.pov_json_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    perception_dir = Path(args.perception_dir) if args.perception_dir else None

    discovered = _discover_jsons(pov_json_dir)
    if not discovered:
        print(f"error=no json found under {pov_json_dir}")
        return 2

    requested: list[str] = []
    missing: list[str] = []
    selected_uids: list[str]
    selection_mode = "all_json"
    if args.uids_file:
        selection_mode = "uids_file"
        requested = _read_uids_file(Path(args.uids_file))
        selected_uids = []
        seen: set[str] = set()
        for uid in requested:
            if uid in discovered and uid not in seen:
                selected_uids.append(uid)
                seen.add(uid)
            elif uid not in discovered:
                missing.append(uid)
    else:
        selected_uids = sorted(discovered.keys())

    rows: list[dict[str, Any]] = []
    weight_terms = {
        "has_place": 1.0,
        "has_interaction": 1.0,
        "object_vocab_nonzero": 1.0,
        "lost_object_nonzero": 1.0,
        "perception_contact_nonzero": 1.0,
        "events_v1_nonzero": 1.0,
    }

    for uid in selected_uids:
        json_path = discovered[uid]
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        events_v1 = _extract_events_v1(payload if isinstance(payload, dict) else {})
        events_v1_count = len(events_v1)
        evidence_span_count = 0
        place_ids: set[str] = set()
        interaction_events_count = 0
        object_vocab: set[str] = set()
        for event in events_v1:
            evidence = event.get("evidence", [])
            if isinstance(evidence, list):
                evidence_span_count += len([x for x in evidence if isinstance(x, dict)])
            place = event.get("place_segment_id", event.get("place_segment"))
            if isinstance(place, str) and place.strip():
                place_ids.add(place.strip())
            score = _to_float(event.get("interaction_score"), default=0.0)
            obj = str(event.get("interaction_primary_object", "")).strip()
            if score > 0.0 or obj:
                interaction_events_count += 1
            if obj:
                object_vocab.add(obj.lower())

        object_memory = _extract_object_memory(payload if isinstance(payload, dict) else {})
        object_memory_total = len(object_memory)
        for item in object_memory:
            name = str(item.get("object_name", "")).strip()
            if name:
                object_vocab.add(name.lower())

        lost_queries = _extract_lost_object_queries(payload if isinstance(payload, dict) else {})
        lost_object_queries_total = len(lost_queries)
        has_retrieval_hits = 1 if events_v1_count > 0 or evidence_span_count > 0 else 0
        place_segments_count = len(place_ids)
        object_vocab_size = len(object_vocab)

        chain_queries_total, chain_success_rate = _extract_chain_metrics(uid, pov_json_dir)
        frames_processed, contact_events = _extract_perception_summary(uid, perception_dir)

        has_place = 1 if place_segments_count > 0 else 0
        has_interaction = 1 if interaction_events_count > 0 else 0
        object_vocab_nonzero = 1 if object_vocab_size > 0 else 0
        lost_object_nonzero = 1 if lost_object_queries_total > 0 else 0
        perception_contact_nonzero = 1 if (contact_events is not None and contact_events > 0.0) else 0
        events_v1_nonzero = 1 if events_v1_count > 0 else 0

        coverage_score = (
            weight_terms["has_place"] * has_place
            + weight_terms["has_interaction"] * has_interaction
            + weight_terms["object_vocab_nonzero"] * object_vocab_nonzero
            + weight_terms["lost_object_nonzero"] * lost_object_nonzero
            + weight_terms["perception_contact_nonzero"] * perception_contact_nonzero
            + weight_terms["events_v1_nonzero"] * events_v1_nonzero
        )

        row = {
            "uid": uid,
            "json_path": str(json_path),
            "events_v1_count": events_v1_count,
            "evidence_span_count": evidence_span_count,
            "has_retrieval_hits": has_retrieval_hits,
            "place_segments_count": place_segments_count,
            "interaction_events_count": interaction_events_count,
            "object_vocab_size": object_vocab_size,
            "object_memory_objects_total": object_memory_total,
            "lost_object_queries_total": lost_object_queries_total,
            "chain_queries_total": "" if chain_queries_total is None else chain_queries_total,
            "chain_success_rate": "" if chain_success_rate is None else round(chain_success_rate, 6),
            "perception_frames_processed": "" if frames_processed is None else frames_processed,
            "perception_contact_events": "" if contact_events is None else contact_events,
            "missing_place_interaction": 1 if (place_segments_count == 0 and interaction_events_count == 0) else 0,
            "missing_lost_object": 1 if (lost_object_queries_total == 0) else 0,
            "missing_chain": 1 if (chain_queries_total is None) else 0,
            "missing_perception": 1 if (frames_processed is None and contact_events is None) else 0,
            "coverage_score": round(float(coverage_score), 6),
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (-_to_float(r.get("coverage_score"), 0.0), str(r.get("uid", ""))))
    coverage_csv = out_dir / "coverage.csv"
    coverage_md = out_dir / "coverage.md"
    uid_candidates = out_dir / "uid_candidates.txt"
    snapshot_path = out_dir / "snapshot.json"
    _write_csv(coverage_csv, rows_sorted)
    _write_md(coverage_md, rows_sorted)
    uid_candidates.write_text("\n".join(str(r.get("uid", "")).strip() for r in rows_sorted if str(r.get("uid", "")).strip()) + "\n", encoding="utf-8")

    scores = [_to_float(r.get("coverage_score"), 0.0) for r in rows_sorted]
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(pov_json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "perception_dir": str(perception_dir) if perception_dir else None,
        },
        "selection": {
            "selection_mode": selection_mode,
            "uids_requested": len(requested),
            "uids_found": len(selected_uids),
            "uids_missing_count": len(missing),
            "uids_missing_sample": missing[:10],
        },
        "weights": weight_terms,
        "coverage_score_stats": {
            "min": min(scores) if scores else 0.0,
            "median": median(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        },
        "outputs": {
            "coverage_csv": str(coverage_csv),
            "coverage_md": str(coverage_md),
            "uid_candidates": str(uid_candidates),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selection_mode={selection_mode}")
    print(f"uids_found={len(selected_uids)}")
    print(f"saved_coverage_csv={coverage_csv}")
    print(f"saved_uid_candidates={uid_candidates}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

