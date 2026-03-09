from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pov_compiler.bench.query_bank import QueryBank


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for golden real sample outputs.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for golden real sample outputs.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sample_config(suite_root: Path) -> tuple[Path, dict[str, Any]]:
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    raw = str(manifest_payload.get("golden_sample_config", "")).strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (_repo_root() / raw).resolve()
        return candidate, _load_yaml(candidate)
    candidate = (_repo_root() / "configs" / "samples" / "golden_real_sample_v1.yaml").resolve()
    return candidate, _load_yaml(candidate)


def _load_primary_query_bank(suite_root: Path) -> tuple[QueryBank, dict[str, Any]]:
    lock_payload = _read_json(suite_root / "manifest" / "query_bank_lock.json")
    primary = lock_payload.get("primary", {})
    primary_name = Path(str(primary.get("path", "")).strip()).name if isinstance(primary, dict) else ""
    bank_root = suite_root / "manifest" / "query_banks"
    if primary_name:
        candidate = bank_root / primary_name
        if candidate.exists():
            return QueryBank.from_path(candidate), primary if isinstance(primary, dict) else {}
    candidates = sorted(bank_root.glob("*.yaml"))
    if candidates:
        return QueryBank.from_path(candidates[0]), primary if isinstance(primary, dict) else {}
    raise FileNotFoundError("Primary query bank copy missing for golden sample.")


def _artifact_non_empty(suite_root: Path, artifact_name: str) -> tuple[bool, dict[str, Any]]:
    mapping = {
        "provider_reachability": suite_root / "provider_reachability" / "summary.json",
        "provider_telemetry": suite_root / "provider_telemetry" / "summary.json",
        "provider_normalization": suite_root / "provider_normalization" / "snapshot.json",
        "result_diagnosis": suite_root / "result_diagnosis" / "snapshot.json",
        "delta_audit": suite_root / "delta_audit" / "snapshot.json",
        "compare": suite_root / "compare" / "tables" / "table_main_results.csv",
    }
    target = mapping.get(artifact_name)
    if target is None or not target.exists():
        return False, {"artifact": artifact_name, "exists": False}
    if target.suffix == ".csv":
        rows = _read_csv_rows(target)
        return bool(rows), {"artifact": artifact_name, "exists": True, "rows_total": len(rows), "path": str(target)}
    payload = _read_json(target)
    rows_total = int(payload.get("rows_total", 0) or 0) if isinstance(payload, dict) else 0
    if artifact_name == "provider_reachability":
        ok = bool(payload.get("reachable")) and str(payload.get("real_call_status", "")).strip() == "ok"
    elif artifact_name == "provider_telemetry":
        ok = str(payload.get("real_call_status", "")).strip() in {"observed", "ok"}
    else:
        ok = bool(rows_total > 0 or payload)
    return ok, {"artifact": artifact_name, "exists": True, "rows_total": rows_total, "path": str(target)}


def build_golden_real_sample_payload(
    *,
    suite_dir: str | Path,
) -> dict[str, Any]:
    suite_root = Path(suite_dir).resolve()
    config_path, config = _load_sample_config(suite_root)
    selection = config.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    exclude_actions = {
        str(item).strip()
        for item in selection.get("exclude_recommended_actions", [])
        if str(item).strip()
    }
    max_query_groups = int(selection.get("max_query_groups", 3) or 3)
    required_proof_status = {
        str(item).strip() for item in selection.get("require_provider_proof_status", []) if str(item).strip()
    }
    required_real_call_status = {
        str(item).strip() for item in selection.get("require_real_call_status", []) if str(item).strip()
    }
    required_artifacts = [
        str(item).strip() for item in selection.get("required_non_empty_artifacts", []) if str(item).strip()
    ]

    bank, primary_lock = _load_primary_query_bank(suite_root)
    provider_proof = _read_json(suite_root / "provider_reachability" / "summary.json")
    query_audit_rows = _read_csv_rows(suite_root / "query_strength_audit" / "tables" / "table_query_strength_audit.csv")
    promotion_summary = _read_json(suite_root / "query_promotion_pack" / "query_pack" / "promotion_summary.json")
    freeze_manifest = _read_json(suite_root / "freeze" / "freeze_manifest.json")
    manifest_path = suite_root / "manifest" / "experiment_manifest.yaml"
    manifest_payload = _load_yaml(manifest_path)

    artifact_checks: dict[str, Any] = {}
    artifacts_ok = True
    for artifact_name in required_artifacts:
        ok, payload = _artifact_non_empty(suite_root, artifact_name)
        artifact_checks[artifact_name] = payload
        artifacts_ok = artifacts_ok and ok

    proof_ok = True
    if required_proof_status:
        proof_ok = str(provider_proof.get("proof_status", "")).strip() in required_proof_status
    real_call_ok = True
    if required_real_call_status:
        real_call_ok = str(provider_proof.get("real_call_status", "")).strip() in required_real_call_status

    sorted_rows = sorted(
        query_audit_rows,
        key=lambda row: (
            0 if str(row.get("promotion_candidate", "")).strip().lower() in {"1", "true", "yes"} else 1,
            -float(row.get("nonzero_delta_rate", 0.0) or 0.0),
            -float(row.get("significance_available_rate", 0.0) or 0.0),
            str(row.get("query_group", "")),
        ),
    )
    selected_groups: list[str] = []
    selection_rows: list[dict[str, Any]] = []
    for row in sorted_rows:
        group_id = str(row.get("query_group", "")).strip()
        action = str(row.get("recommended_action", "")).strip()
        if not group_id or action in exclude_actions or group_id in selected_groups:
            continue
        selected_groups.append(group_id)
        selection_rows.append(dict(row))
        if len(selected_groups) >= max_query_groups:
            break

    selected_queries = [
        query.model_dump()
        for query in bank.queries
        if bool(query.enabled) and str(query.group) in set(selected_groups)
    ]

    status = "ok"
    if not proof_ok or not real_call_ok or not artifacts_ok:
        status = "partial"
    if not selected_queries:
        status = "empty"

    expected_outputs = {
        "required_non_empty_artifacts": artifact_checks,
        "selected_query_groups": selected_groups,
        "selected_queries_total": len(selected_queries),
    }
    source_suite_hash = freeze_manifest.get("freeze_sha256", "") or (_sha256_path(manifest_path) if manifest_path.exists() else "")
    sample_manifest = {
        "sample_id": str(config.get("sample_id", "golden_real_sample_v1")).strip() or "golden_real_sample_v1",
        "sample_version": str(config.get("sample_version", "v1")).strip() or "v1",
        "golden_sample_status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_suite_dir": str(suite_root),
        "source_suite_id": str(manifest_payload.get("suite_id", "")).strip(),
        "source_suite_version": str(manifest_payload.get("suite_version", "")).strip(),
        "source_suite_hash": source_suite_hash,
        "source_query_bank_id": str(primary_lock.get("query_bank_id", bank.query_bank_id)),
        "source_query_bank_version": str(primary_lock.get("query_bank_version", bank.query_bank_version)),
        "source_query_bank_hash": str(primary_lock.get("query_bank_hash", "")),
        "source_provider_proof": provider_proof,
        "selected_query_groups": selected_groups,
        "selected_query_ids": [str(item.get("query_id", "")).strip() for item in selected_queries],
        "promotion_summary": promotion_summary,
        "expected_non_empty_outputs": expected_outputs,
        "selection_config": selection,
        "sample_config_path": str(config_path),
    }
    query_set = {
        "sample_id": sample_manifest["sample_id"],
        "source_query_bank_id": sample_manifest["source_query_bank_id"],
        "source_query_bank_hash": sample_manifest["source_query_bank_hash"],
        "selected_groups": [
            {
                "query_group": row.get("query_group", ""),
                "query_type": row.get("query_type", ""),
                "recommended_action": row.get("recommended_action", ""),
                "promotion_candidate": row.get("promotion_candidate", ""),
                "promotion_reason": row.get("promotion_reason", ""),
            }
            for row in selection_rows
        ],
        "queries": selected_queries,
    }
    return {
        "sample_manifest": sample_manifest,
        "query_set": query_set,
        "expected_outputs": expected_outputs,
        "status": status,
    }


def write_golden_real_sample_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    golden_root = out_root if out_root.name == "golden_real_sample" else out_root / "golden_real_sample"
    golden_root.mkdir(parents=True, exist_ok=True)
    payload = build_golden_real_sample_payload(suite_dir=suite_dir)
    manifest_path = golden_root / "sample_manifest.json"
    query_set_path = golden_root / "query_set.yaml"
    expected_outputs_path = golden_root / "expected_outputs.json"
    report_path = golden_root / "report.md"
    snapshot_path = golden_root / "snapshot.json"
    _write_text(manifest_path, json.dumps(payload["sample_manifest"], ensure_ascii=False, indent=2))
    _dump_yaml(query_set_path, payload["query_set"])
    _write_text(expected_outputs_path, json.dumps(payload["expected_outputs"], ensure_ascii=False, indent=2))
    report_lines = [
        "# Golden Real Sample",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- golden_sample_status: `{payload['status']}`",
        f"- source_query_bank_id: `{payload['sample_manifest'].get('source_query_bank_id', '')}`",
        f"- source_query_bank_hash: `{payload['sample_manifest'].get('source_query_bank_hash', '')}`",
        f"- selected_query_groups: `{payload['sample_manifest'].get('selected_query_groups', [])}`",
        f"- selected_query_ids: `{payload['sample_manifest'].get('selected_query_ids', [])}`",
        f"- source_provider_proof: `{json.dumps(payload['sample_manifest'].get('source_provider_proof', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Files",
        "",
        f"- sample_manifest_json: `{manifest_path}`",
        f"- query_set_yaml: `{query_set_path}`",
        f"- expected_outputs_json: `{expected_outputs_path}`",
        f"- snapshot_json: `{snapshot_path}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    snapshot = dict(payload["sample_manifest"])
    snapshot["outputs"] = {
        "sample_manifest_json": str(manifest_path),
        "query_set_yaml": str(query_set_path),
        "expected_outputs_json": str(expected_outputs_path),
        "report_md": str(report_path),
        "snapshot_json": str(snapshot_path),
    }
    _write_text(snapshot_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
    return {
        "sample_manifest_json": manifest_path,
        "query_set_yaml": query_set_path,
        "expected_outputs_json": expected_outputs_path,
        "report_md": report_path,
        "snapshot_json": snapshot_path,
        "status": payload["status"],
        "sample_manifest": payload["sample_manifest"],
    }
