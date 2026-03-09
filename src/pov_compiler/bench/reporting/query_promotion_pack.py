from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pov_compiler.bench.query_bank import QueryBank


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_primary_query_bank(suite_root: Path) -> tuple[QueryBank, dict[str, Any]]:
    lock_payload = _read_json(suite_root / "manifest" / "query_bank_lock.json")
    primary = lock_payload.get("primary", {})
    primary_path = ""
    if isinstance(primary, dict):
        primary_path = str(primary.get("path", "")).strip()
    banks_root = suite_root / "manifest" / "query_banks"
    if primary_path:
        candidate = banks_root / Path(primary_path).name
        if candidate.exists():
            return QueryBank.from_path(candidate), primary
    candidates = sorted(banks_root.glob("*.yaml"))
    if candidates:
        return QueryBank.from_path(candidates[0]), primary if isinstance(primary, dict) else {}
    raise FileNotFoundError("Primary query bank copy not found under manifest/query_banks.")


def _promotion_confidence(promoted_count: int, weak_groups_count: int, selected_uids_count: int) -> str:
    if promoted_count <= 0:
        return "low"
    if weak_groups_count > promoted_count or selected_uids_count < 2:
        return "partial"
    return "high"


def _pack_payload(
    *,
    pack_id: str,
    queries: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    suite_root: Path,
    source_query_bank_id: str,
    source_query_bank_hash: str,
    source_query_bank_version: str,
    promotion_criteria: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pack_id": pack_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_suite_dir": str(suite_root),
        "source_query_bank_id": source_query_bank_id,
        "source_query_bank_version": source_query_bank_version,
        "source_query_bank_hash": source_query_bank_hash,
        "promotion_criteria": promotion_criteria,
        "groups": groups,
        "queries": queries,
    }


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required for query promotion packs.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_query_promotion_pack_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    suite_root = Path(suite_dir).resolve()
    out_root = Path(out_dir).resolve()
    pack_root = out_root / "query_pack"
    audit_rows = _read_csv_rows(suite_root / "query_strength_audit" / "tables" / "table_query_strength_audit.csv")
    audit_snapshot = _read_json(suite_root / "query_strength_audit" / "snapshot.json")
    bank, primary_lock = _load_primary_query_bank(suite_root)

    promoted_groups: list[dict[str, Any]] = []
    analysis_groups: list[dict[str, Any]] = []
    dropped_groups: list[dict[str, Any]] = []
    promoted_group_ids: set[str] = set()
    analysis_group_ids: set[str] = set()

    for row in audit_rows:
        group_id = str(row.get("query_group", "")).strip()
        if not group_id:
            continue
        action = str(row.get("recommended_action", "")).strip()
        promotion_candidate = str(row.get("promotion_candidate", "")).strip().lower() in {"1", "true", "yes", "y"}
        group_payload = {
            "query_group": group_id,
            "query_type": str(row.get("query_type", "")).strip(),
            "recommended_action": action,
            "promotion_candidate": promotion_candidate,
            "promotion_reason": str(row.get("promotion_reason", "")).strip(),
        }
        if promotion_candidate or action == "promote_to_core_query_bank":
            promoted_groups.append(group_payload)
            promoted_group_ids.add(group_id)
        elif action == "keep_for_analysis_only":
            analysis_groups.append(group_payload)
            analysis_group_ids.add(group_id)
        else:
            dropped_groups.append(group_payload)

    promoted_queries = [
        query.model_dump()
        for query in bank.queries
        if bool(query.enabled) and str(query.group) in promoted_group_ids
    ]
    analysis_only_queries = [
        query.model_dump()
        for query in bank.queries
        if bool(query.enabled) and str(query.group) in analysis_group_ids
    ]
    promotion_criteria = {
        "recommended_action": "promote_to_core_query_bank",
        "source_main_recommendation": audit_snapshot.get("main_recommendation", ""),
        "source_recommended_action_counts": audit_snapshot.get("recommended_action_counts", {}),
        "source_weak_query_groups_count": audit_snapshot.get("weak_query_groups_count", 0),
    }
    source_bank_id = str(primary_lock.get("query_bank_id", bank.query_bank_id))
    source_bank_hash = str(primary_lock.get("query_bank_hash", ""))
    source_bank_version = str(primary_lock.get("query_bank_version", bank.query_bank_version))

    promoted_payload = _pack_payload(
        pack_id="promoted_queries_v1",
        queries=promoted_queries,
        groups=promoted_groups,
        suite_root=suite_root,
        source_query_bank_id=source_bank_id,
        source_query_bank_hash=source_bank_hash,
        source_query_bank_version=source_bank_version,
        promotion_criteria=promotion_criteria,
    )
    analysis_payload = _pack_payload(
        pack_id="analysis_only_queries_v1",
        queries=analysis_only_queries,
        groups=analysis_groups,
        suite_root=suite_root,
        source_query_bank_id=source_bank_id,
        source_query_bank_hash=source_bank_hash,
        source_query_bank_version=source_bank_version,
        promotion_criteria={"recommended_action": "keep_for_analysis_only"},
    )

    promoted_yaml = pack_root / "promoted_queries.yaml"
    analysis_yaml = pack_root / "analysis_only_queries.yaml"
    summary_json = pack_root / "promotion_summary.json"
    report_md = out_root / "report.md"
    _dump_yaml(promoted_yaml, promoted_payload)
    _dump_yaml(analysis_yaml, analysis_payload)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_suite_dir": str(suite_root),
        "promoted_count": int(len(promoted_queries)),
        "analysis_only_count": int(len(analysis_only_queries)),
        "dropped_count": int(
            len([query for query in bank.queries if bool(query.enabled)]) - len(promoted_queries) - len(analysis_only_queries)
        ),
        "source_query_bank_id": source_bank_id,
        "source_query_bank_hash": source_bank_hash,
        "promotion_criteria": promotion_criteria,
        "promotion_confidence": _promotion_confidence(
            promoted_count=len(promoted_queries),
            weak_groups_count=int(audit_snapshot.get("weak_query_groups_count", 0) or 0),
            selected_uids_count=int(audit_snapshot.get("selected_uids_count", 0) or 0),
        ),
        "promoted_groups": promoted_groups,
        "analysis_only_groups": analysis_groups,
        "dropped_groups": dropped_groups,
        "outputs": {
            "promoted_yaml": str(promoted_yaml),
            "analysis_only_yaml": str(analysis_yaml),
            "promotion_summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    _write_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2))
    report_lines = [
        "# Query Promotion Pack",
        "",
        f"- source_suite_dir: `{suite_root}`",
        f"- source_query_bank_id: `{source_bank_id}`",
        f"- source_query_bank_hash: `{source_bank_hash}`",
        f"- promoted_count: `{summary.get('promoted_count', 0)}`",
        f"- analysis_only_count: `{summary.get('analysis_only_count', 0)}`",
        f"- dropped_count: `{summary.get('dropped_count', 0)}`",
        f"- promotion_confidence: `{summary.get('promotion_confidence', 'low')}`",
        f"- promotion_criteria: `{json.dumps(summary.get('promotion_criteria', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Files",
        "",
        f"- promoted_queries: `{promoted_yaml}`",
        f"- analysis_only_queries: `{analysis_yaml}`",
        f"- promotion_summary: `{summary_json}`",
        f"- report_md: `{report_md}`",
    ]
    _write_text(report_md, "\n".join(report_lines))
    return {
        "promoted_yaml": promoted_yaml,
        "analysis_only_yaml": analysis_yaml,
        "promotion_summary_json": summary_json,
        "report_md": report_md,
        "summary": summary,
    }
