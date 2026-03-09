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


def _to_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required for query uplift candidates.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _pack_payload(
    *,
    pack_id: str,
    queries: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    suite_root: Path,
    source_query_bank_id: str,
    source_query_bank_hash: str,
    source_query_bank_version: str,
    uplift_basis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pack_id": pack_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_suite_dir": str(suite_root),
        "source_query_bank_id": source_query_bank_id,
        "source_query_bank_version": source_query_bank_version,
        "source_query_bank_hash": source_query_bank_hash,
        "uplift_basis": uplift_basis,
        "groups": groups,
        "queries": queries,
    }


def write_query_uplift_candidates_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    suite_root = Path(suite_dir).resolve()
    out_root = Path(out_dir).resolve()
    pack_root = out_root / "query_uplift_candidates"
    audit_rows = _read_csv_rows(suite_root / "query_strength_audit" / "tables" / "table_query_strength_audit.csv")
    audit_snapshot = _read_json(suite_root / "query_strength_audit" / "snapshot.json")
    bank, primary_lock = _load_primary_query_bank(suite_root)

    candidate_groups: list[dict[str, Any]] = []
    analysis_groups: list[dict[str, Any]] = []
    drop_groups: list[dict[str, Any]] = []
    candidate_group_ids: set[str] = set()
    analysis_group_ids: set[str] = set()
    drop_group_ids: set[str] = set()

    for row in audit_rows:
        group_id = str(row.get("query_group", "")).strip()
        if not group_id:
            continue
        action = str(row.get("recommended_action", "")).strip()
        payload = {
            "query_group": group_id,
            "query_type": str(row.get("query_type", "")).strip(),
            "recommended_action": action,
            "uplift_candidate": _to_bool(row.get("uplift_candidate")),
            "uplift_reason": str(row.get("uplift_reason", "")).strip(),
            "promotion_candidate": _to_bool(row.get("promotion_candidate")),
            "promotion_reason": str(row.get("promotion_reason", "")).strip(),
        }
        if payload["uplift_candidate"]:
            candidate_groups.append(payload)
            candidate_group_ids.add(group_id)
        elif action == "keep_for_analysis_only":
            analysis_groups.append(payload)
            analysis_group_ids.add(group_id)
        else:
            drop_groups.append(payload)
            drop_group_ids.add(group_id)

    candidate_queries = [
        query.model_dump()
        for query in bank.queries
        if bool(query.enabled) and str(query.group) in candidate_group_ids
    ]
    analysis_only_queries = [
        query.model_dump()
        for query in bank.queries
        if bool(query.enabled) and str(query.group) in analysis_group_ids
    ]
    drop_queries = [
        query.model_dump()
        for query in bank.queries
        if bool(query.enabled) and str(query.group) in drop_group_ids
    ]

    source_bank_id = str(primary_lock.get("query_bank_id", bank.query_bank_id))
    source_bank_hash = str(primary_lock.get("query_bank_hash", ""))
    source_bank_version = str(primary_lock.get("query_bank_version", bank.query_bank_version))
    uplift_basis = {
        "source_main_recommendation": audit_snapshot.get("main_recommendation", ""),
        "source_recommended_action_counts": audit_snapshot.get("recommended_action_counts", {}),
        "source_weak_query_groups_count": audit_snapshot.get("weak_query_groups_count", 0),
        "source_uplift_candidate_count": audit_snapshot.get("uplift_candidate_count", 0),
    }

    candidate_yaml = pack_root / "candidate_queries.yaml"
    analysis_yaml = pack_root / "analysis_only_queries.yaml"
    drop_yaml = pack_root / "drop_queries.yaml"
    summary_json = pack_root / "uplift_summary.json"
    report_md = out_root / "report.md"

    _dump_yaml(
        candidate_yaml,
        _pack_payload(
            pack_id="query_uplift_candidates_v1",
            queries=candidate_queries,
            groups=candidate_groups,
            suite_root=suite_root,
            source_query_bank_id=source_bank_id,
            source_query_bank_hash=source_bank_hash,
            source_query_bank_version=source_bank_version,
            uplift_basis=uplift_basis,
        ),
    )
    _dump_yaml(
        analysis_yaml,
        _pack_payload(
            pack_id="query_uplift_analysis_only_v1",
            queries=analysis_only_queries,
            groups=analysis_groups,
            suite_root=suite_root,
            source_query_bank_id=source_bank_id,
            source_query_bank_hash=source_bank_hash,
            source_query_bank_version=source_bank_version,
            uplift_basis={"recommended_action": "keep_for_analysis_only"},
        ),
    )
    _dump_yaml(
        drop_yaml,
        _pack_payload(
            pack_id="query_uplift_drop_v1",
            queries=drop_queries,
            groups=drop_groups,
            suite_root=suite_root,
            source_query_bank_id=source_bank_id,
            source_query_bank_hash=source_bank_hash,
            source_query_bank_version=source_bank_version,
            uplift_basis={"recommended_action": "drop_from_main_real"},
        ),
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_suite_dir": str(suite_root),
        "candidate_count": int(len(candidate_queries)),
        "analysis_only_count": int(len(analysis_only_queries)),
        "drop_count": int(len(drop_queries)),
        "source_query_bank_id": source_bank_id,
        "source_query_bank_hash": source_bank_hash,
        "uplift_confidence": "low"
        if len(candidate_queries) <= 0
        else ("partial" if int(audit_snapshot.get("weak_query_groups_count", 0) or 0) > len(candidate_groups) else "high"),
        "uplift_basis": uplift_basis,
        "candidate_groups": candidate_groups,
        "analysis_only_groups": analysis_groups,
        "drop_groups": drop_groups,
        "outputs": {
            "candidate_yaml": str(candidate_yaml),
            "analysis_only_yaml": str(analysis_yaml),
            "drop_yaml": str(drop_yaml),
            "uplift_summary_json": str(summary_json),
            "report_md": str(report_md),
        },
    }
    _write_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2))
    report_lines = [
        "# Query Uplift Candidates",
        "",
        f"- source_suite_dir: `{suite_root}`",
        f"- source_query_bank_id: `{source_bank_id}`",
        f"- source_query_bank_hash: `{source_bank_hash}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        f"- analysis_only_count: `{summary.get('analysis_only_count', 0)}`",
        f"- drop_count: `{summary.get('drop_count', 0)}`",
        f"- uplift_confidence: `{summary.get('uplift_confidence', 'low')}`",
        f"- uplift_basis: `{json.dumps(summary.get('uplift_basis', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Files",
        "",
        f"- candidate_queries: `{candidate_yaml}`",
        f"- analysis_only_queries: `{analysis_yaml}`",
        f"- drop_queries: `{drop_yaml}`",
        f"- uplift_summary: `{summary_json}`",
        f"- report_md: `{report_md}`",
    ]
    _write_text(report_md, "\n".join(report_lines))
    return {
        "candidate_yaml": candidate_yaml,
        "analysis_only_yaml": analysis_yaml,
        "drop_yaml": drop_yaml,
        "uplift_summary_json": summary_json,
        "report_md": report_md,
        "summary": summary,
    }
