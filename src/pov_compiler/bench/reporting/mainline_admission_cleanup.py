from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


def _normalize_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalize_cell(row.get(key)) for key in fieldnames})


def _rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "| status |\n| --- |\n| empty |\n"
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_normalize_cell(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _to_int(value: Any) -> int | None:
    out = _to_float(value)
    if out is None:
        return None
    return int(round(out))


def _stable_relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def _resolve_compare_root(compare_dir: str | Path) -> Path:
    compare_path = Path(compare_dir).resolve()
    candidate = compare_path / "compare"
    if candidate.exists():
        return candidate
    return compare_path


def _load_contract_manifest(manifest_path: str | Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {}
    return _load_yaml(Path(manifest_path))


def _threshold(manifest_payload: dict[str, Any], key: str, default: float) -> float:
    thresholds = manifest_payload.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    raw = thresholds.get(key, default)
    parsed = _to_float(raw)
    return float(parsed) if parsed is not None else float(default)


def _freeze_contract(
    suite_dir: Path,
    *,
    expected_query_bank_id: str,
    expected_query_bank_hash: str,
    manifest_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    freeze_manifest = _read_json(suite_dir / "freeze" / "freeze_manifest.json")
    artifact_count = _to_int(freeze_manifest.get("artifact_count")) or 0
    freeze_sha = str(freeze_manifest.get("freeze_sha256", "")).strip()
    freeze_query_bank_id = str(freeze_manifest.get("query_bank_id", "")).strip()
    freeze_query_bank_hash = str(freeze_manifest.get("query_bank_hash", "")).strip()
    min_artifacts = _threshold(manifest_payload, "freeze_artifact_count_min", 1.0)
    freeze_artifacts_present = bool(freeze_manifest) and artifact_count >= int(min_artifacts) and bool(freeze_sha)
    query_bank_matches = (
        not expected_query_bank_id
        or not expected_query_bank_hash
        or (
            freeze_query_bank_id == expected_query_bank_id
            and freeze_query_bank_hash == expected_query_bank_hash
        )
    )
    if freeze_artifacts_present and query_bank_matches:
        status = "adequate"
    elif freeze_artifacts_present:
        status = "borderline"
    else:
        status = "insufficient"
    details = {
        "freeze_artifacts_present": freeze_artifacts_present,
        "artifact_count": artifact_count,
        "freeze_sha256_present": bool(freeze_sha),
        "freeze_query_bank_matches": query_bank_matches,
        "freeze_manifest_relpath": _stable_relpath(suite_dir / "freeze" / "freeze_manifest.json"),
    }
    return status, details


def build_sample_contract_summary(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    suite_root = Path(suite_dir).resolve()
    compare_root = _resolve_compare_root(compare_dir)
    compare_summary = _read_json(compare_root / "compare_summary.json")
    result_health = _read_json(suite_root / "result_health" / "snapshot.json")
    manifest_payload = _load_contract_manifest(manifest_path)

    selection = result_health.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    coverage_stats = selection.get("coverage_score_stats", {})
    if not isinstance(coverage_stats, dict):
        coverage_stats = {}
    selected_uids_count = _to_int(compare_summary.get("selected_uids_count"))
    if selected_uids_count is None:
        selected_uids_count = _to_int(selection.get("selected_uids_count")) or 0
    paired_sample_count = _to_int(compare_summary.get("paired_sample_count")) or 0
    coverage_mean = _to_float(coverage_stats.get("mean"))
    coverage_min = _to_float(coverage_stats.get("min"))
    budget_keys = compare_summary.get("budget_keys", [])
    if not isinstance(budget_keys, list):
        budget_keys = []
    query_bank_id = str(compare_summary.get("query_bank_b_id", compare_summary.get("query_bank_id", ""))).strip()
    query_bank_hash = str(compare_summary.get("query_bank_b_hash", compare_summary.get("query_bank_hash", ""))).strip()
    provider_signature_match = bool(compare_summary.get("provider_signature_match", False))
    provider_noise_summary_hash_match = (
        str(compare_summary.get("provider_noise_summary_hash_a", "")).strip()
        == str(compare_summary.get("provider_noise_summary_hash_b", "")).strip()
    )
    perception_signature_match = bool(compare_summary.get("perception_signature_match", False))
    object_memory_logic_variant_mainline = str(
        compare_summary.get("object_memory_logic_variant_b", "")
    ).strip()
    uid_set_id = str(
        compare_summary.get(
            "uid_set_id",
            result_health.get("compare_summary", {}).get("uid_set_id", ""),
        )
    ).strip()
    sample_signature = str(
        result_health.get("compare_summary", {}).get(
            "sample_signature_hash",
            compare_summary.get("sample_signature_hash", ""),
        )
    ).strip()

    freeze_contract_status, freeze_details = _freeze_contract(
        suite_root,
        expected_query_bank_id=query_bank_id,
        expected_query_bank_hash=query_bank_hash,
        manifest_payload=manifest_payload,
    )

    selected_borderline_min = _threshold(manifest_payload, "selected_uids_count_borderline_min", 6.0)
    selected_adequate_min = _threshold(manifest_payload, "selected_uids_count_adequate_min", 8.0)
    paired_borderline_min = _threshold(manifest_payload, "paired_sample_count_borderline_min", 12.0)
    paired_adequate_min = _threshold(manifest_payload, "paired_sample_count_adequate_min", 16.0)
    coverage_mean_min = _threshold(manifest_payload, "coverage_mean_min", 2.5)
    coverage_min_min = _threshold(manifest_payload, "coverage_min_min", 2.0)

    alignment_ok = bool(compare_summary.get("alignment_ok", False))
    compare_pair_id = str(compare_summary.get("compare_pair_id", "")).strip()
    expected_compare_pair_id = str(manifest_payload.get("expected_compare_pair_id", "")).strip()
    expected_query_bank_id = str(manifest_payload.get("expected_query_bank_id", "")).strip()
    expected_query_bank_hash = str(manifest_payload.get("expected_query_bank_hash", "")).strip()
    expected_variant = str(manifest_payload.get("expected_object_memory_logic_variant_mainline", "")).strip()
    compare_pair_matches = not expected_compare_pair_id or compare_pair_id == expected_compare_pair_id
    query_bank_matches = (
        (not expected_query_bank_id or query_bank_id == expected_query_bank_id)
        and (not expected_query_bank_hash or query_bank_hash == expected_query_bank_hash)
    )
    variant_matches = not expected_variant or object_memory_logic_variant_mainline == expected_variant

    selected_borderline_ok = selected_uids_count >= int(selected_borderline_min)
    paired_borderline_ok = paired_sample_count >= int(paired_borderline_min)
    selected_adequate_ok = selected_uids_count >= int(selected_adequate_min)
    paired_adequate_ok = paired_sample_count >= int(paired_adequate_min)
    coverage_ok = (
        coverage_mean is not None
        and coverage_mean >= coverage_mean_min
        and coverage_min is not None
        and coverage_min >= coverage_min_min
    )
    if (
        alignment_ok
        and compare_pair_matches
        and query_bank_matches
        and variant_matches
        and provider_signature_match
        and provider_noise_summary_hash_match
        and perception_signature_match
        and selected_adequate_ok
        and paired_adequate_ok
        and coverage_ok
        and freeze_contract_status == "adequate"
    ):
        sample_contract_status = "adequate"
    elif (
        alignment_ok
        and compare_pair_matches
        and query_bank_matches
        and variant_matches
        and provider_signature_match
        and provider_noise_summary_hash_match
        and perception_signature_match
        and selected_borderline_ok
        and paired_borderline_ok
        and coverage_ok
        and freeze_contract_status in {"adequate", "borderline"}
    ):
        sample_contract_status = "borderline"
    else:
        sample_contract_status = "insufficient"

    if sample_contract_status == "adequate":
        large_sample_claim_status = "supported"
    elif sample_contract_status == "borderline":
        large_sample_claim_status = "supported_with_caveat"
    elif alignment_ok and (selected_uids_count >= 4 or paired_sample_count >= 8):
        large_sample_claim_status = "needs_softer_wording"
    else:
        large_sample_claim_status = "unsupported"

    wording_recommendation = {
        "supported": "large-sample real main experiment",
        "supported_with_caveat": "large-sample real main experiment (supported with caveat)",
        "needs_softer_wording": "real main experiment with moderate paired sample coverage",
        "unsupported": "real experiment preview only",
    }[large_sample_claim_status]

    summary = {
        "selected_uids_count": selected_uids_count,
        "paired_sample_count": paired_sample_count,
        "budget_keys": budget_keys,
        "uid_set_id": uid_set_id,
        "sample_signature_hash": sample_signature,
        "query_bank_id": query_bank_id,
        "query_bank_hash": query_bank_hash,
        "provider_signature_match": provider_signature_match,
        "provider_noise_summary_hash_match": provider_noise_summary_hash_match,
        "perception_signature_match": perception_signature_match,
        "object_memory_logic_variant_mainline": object_memory_logic_variant_mainline,
        "coverage_mean": coverage_mean,
        "coverage_min": coverage_min,
        "freeze_artifacts_present": freeze_details["freeze_artifacts_present"],
        "sample_contract_status": sample_contract_status,
        "large_sample_claim_status": large_sample_claim_status,
        "wording_recommendation": wording_recommendation,
        "alignment_ok": alignment_ok,
        "compare_pair_id": compare_pair_id,
        "compare_pair_matches_manifest": compare_pair_matches,
        "query_bank_matches_manifest": query_bank_matches,
        "object_memory_variant_matches_manifest": variant_matches,
        "coverage_contract_status": "adequate" if coverage_ok else "insufficient",
        "freeze_contract_status": freeze_contract_status,
        "freeze_details": freeze_details,
        "thresholds": {
            "selected_uids_count_borderline_min": int(selected_borderline_min),
            "selected_uids_count_adequate_min": int(selected_adequate_min),
            "paired_sample_count_borderline_min": int(paired_borderline_min),
            "paired_sample_count_adequate_min": int(paired_adequate_min),
            "coverage_mean_min": coverage_mean_min,
            "coverage_min_min": coverage_min_min,
        },
        "suite_dir": _stable_relpath(suite_root),
        "compare_dir": _stable_relpath(compare_root),
    }
    return summary


def build_mainline_admission_cleanup(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    suite_root = Path(suite_dir).resolve()
    compare_root = _resolve_compare_root(compare_dir)
    compare_parent = compare_root.parent

    result_health = _read_json(suite_root / "result_health" / "snapshot.json")
    compare_summary = _read_json(compare_root / "compare_summary.json")
    decision_snapshot = _read_json(compare_parent / "promotion_decision" / "snapshot.json")
    admission_snapshot = _read_json(suite_root / "admission_control" / "snapshot.json")
    normalization_snapshot = _read_json(suite_root / "provider_normalization" / "snapshot.json")
    reachability_summary = _read_json(suite_root / "provider_reachability" / "summary.json")

    sample_contract_summary = build_sample_contract_summary(
        suite_dir=suite_root,
        compare_dir=compare_root,
        manifest_path=manifest_path,
    )
    promotion_summary = decision_snapshot.get("promotion_decision_summary", {})
    if not isinstance(promotion_summary, dict):
        promotion_summary = {}

    promotion_decision = str(promotion_summary.get("promotion_decision", ""))
    promotion_ready = bool(promotion_summary.get("promotion_ready", False))
    decision_confidence = str(promotion_summary.get("decision_confidence", "unknown"))
    persistent_memory_main_status = str(compare_summary.get("persistent_memory_main_status", "unknown"))
    main_gain_state = str(compare_summary.get("main_gain_state", "unknown"))
    admission_status_suite = str(result_health.get("admission_status", "unknown"))
    admission_status_compare_side = str(compare_summary.get("admission_status_b", admission_status_suite))
    calibration_status = str(result_health.get("calibration_status", "unknown"))
    normalization_status = str(normalization_snapshot.get("normalization_status", "unknown"))
    proof_status = str(reachability_summary.get("proof_status", "unknown"))
    provider_health_status = str(compare_summary.get("provider_health_status", "unknown"))

    admission_fail_reasons = admission_snapshot.get("admission_fail_reasons", [])
    if not isinstance(admission_fail_reasons, list):
        admission_fail_reasons = []
    provider_semantics_gap = (
        any("provider_availability" in str(item) for item in admission_fail_reasons)
        and provider_health_status == "ok"
        and proof_status == "ok"
        and normalization_status in {"ok", "partial"}
    )

    sample_contract_status = str(sample_contract_summary.get("sample_contract_status", "insufficient"))
    coverage_contract_status = str(sample_contract_summary.get("coverage_contract_status", "insufficient"))
    freeze_contract_status = str(sample_contract_summary.get("freeze_contract_status", "insufficient"))
    if (
        sample_contract_status == "adequate"
        and coverage_contract_status == "adequate"
        and freeze_contract_status == "adequate"
        and provider_health_status == "ok"
        and proof_status == "ok"
        and normalization_status == "ok"
    ):
        evidence_hardness_status = "hard"
    elif (
        sample_contract_status in {"adequate", "borderline"}
        and coverage_contract_status == "adequate"
        and freeze_contract_status in {"adequate", "borderline"}
        and provider_health_status == "ok"
        and proof_status == "ok"
        and normalization_status in {"ok", "partial"}
    ):
        evidence_hardness_status = "supported_with_caveat"
    else:
        evidence_hardness_status = "soft"

    if promotion_ready and admission_status_suite == "ok":
        consistency = "aligned_ready"
        cleanup_status = "explained"
        recommendation = "mainline_ready"
    elif promotion_ready and admission_status_suite == "partial":
        if sample_contract_status in {"borderline", "adequate"} or provider_semantics_gap:
            consistency = "consistent_different_layers"
            cleanup_status = "explained"
            recommendation = (
                "promotion_valid_admission_partial"
                if sample_contract_status == "adequate"
                else "need_sample_contract_hardening"
            )
        else:
            consistency = "promotion_ahead_of_evidence"
            cleanup_status = "unresolved"
            recommendation = "need_cleaner_run_contract"
    elif promotion_ready:
        consistency = "promotion_blocked_by_run_cleanliness"
        cleanup_status = "blocked"
        recommendation = "need_cleaner_run_contract"
    else:
        consistency = "no_promotion_claim"
        cleanup_status = "explained"
        recommendation = "need_cleaner_run_contract"

    if provider_semantics_gap and recommendation == "promotion_valid_admission_partial":
        recommendation = "promotion_valid_admission_partial"
    elif cleanup_status != "explained" and provider_health_status != "ok":
        recommendation = "need_cleaner_run_contract"
    elif cleanup_status == "explained" and sample_contract_status == "borderline":
        recommendation = "need_sample_contract_hardening"

    missing_evidence: list[str] = []
    if sample_contract_status != "adequate":
        missing_evidence.append("sample_coverage")
    if freeze_contract_status != "adequate":
        missing_evidence.append("freeze_provenance")
    if provider_health_status != "ok" or proof_status != "ok" or normalization_status not in {"ok", "partial"}:
        missing_evidence.append("run_cleanliness")
    if provider_semantics_gap:
        missing_evidence.append("provider_availability_semantics")

    summary = {
        "promotion_decision": promotion_decision,
        "promotion_ready": promotion_ready,
        "decision_confidence": decision_confidence,
        "persistent_memory_main_status": persistent_memory_main_status,
        "main_gain_state": main_gain_state,
        "admission_status_suite": admission_status_suite,
        "admission_status_compare_side": admission_status_compare_side,
        "calibration_status": calibration_status,
        "normalization_status": normalization_status,
        "proof_status": proof_status,
        "provider_health_status": provider_health_status,
        "sample_contract_status": sample_contract_status,
        "coverage_contract_status": coverage_contract_status,
        "freeze_contract_status": freeze_contract_status,
        "evidence_hardness_status": evidence_hardness_status,
        "promotion_vs_admission_consistency_status": consistency,
        "mainline_admission_cleanup_status": cleanup_status,
        "next_action_recommendation": recommendation,
        "missing_evidence": missing_evidence,
        "provider_availability_semantics_gap": provider_semantics_gap,
        "admission_fail_reasons": admission_fail_reasons,
        "large_sample_claim_status": sample_contract_summary.get("large_sample_claim_status"),
        "wording_recommendation": sample_contract_summary.get("wording_recommendation"),
        "compare_dir": _stable_relpath(compare_root),
        "suite_dir": _stable_relpath(suite_root),
    }
    snapshot = {
        "mainline_admission_cleanup_summary": summary,
        "sample_contract_summary": sample_contract_summary,
        "compare_summary": compare_summary,
        "promotion_decision_summary": promotion_summary,
        "admission_control_snapshot": {
            "admission_status": admission_snapshot.get("admission_status"),
            "admission_fail_reasons": admission_fail_reasons,
        },
    }
    return {
        "summary": summary,
        "snapshot": snapshot,
    }


def write_sample_contract_outputs(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = build_sample_contract_summary(
        suite_dir=suite_dir,
        compare_dir=compare_dir,
        manifest_path=manifest_path,
    )
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"freeze_details", "thresholds"}
    }
    rows = [row]
    csv_path = tables_dir / "table_sample_contract_summary.csv"
    md_path = tables_dir / "table_sample_contract_summary.md"
    report_path = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv_rows(csv_path, rows)
    _write_text(md_path, "# Sample Contract Summary\n\n" + _rows_to_markdown(rows))
    report_lines = [
        "# Sample Contract",
        "",
        f"- sample_contract_status: `{summary.get('sample_contract_status', '')}`",
        f"- large_sample_claim_status: `{summary.get('large_sample_claim_status', '')}`",
        f"- wording_recommendation: `{summary.get('wording_recommendation', '')}`",
        f"- selected_uids_count: `{summary.get('selected_uids_count', 0)}`",
        f"- paired_sample_count: `{summary.get('paired_sample_count', 0)}`",
        f"- coverage_mean: `{summary.get('coverage_mean', None)}`",
        f"- coverage_min: `{summary.get('coverage_min', None)}`",
        f"- provider_signature_match: `{summary.get('provider_signature_match', False)}`",
        f"- provider_noise_summary_hash_match: `{summary.get('provider_noise_summary_hash_match', False)}`",
        f"- perception_signature_match: `{summary.get('perception_signature_match', False)}`",
        f"- freeze_contract_status: `{summary.get('freeze_contract_status', '')}`",
        f"- budget_keys: `{summary.get('budget_keys', [])}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    _write_json(snapshot_path, {"sample_contract_summary": summary})
    return {
        "summary": summary,
        "csv_path": csv_path,
        "md_path": md_path,
        "report_path": report_path,
        "snapshot_path": snapshot_path,
    }


def write_mainline_admission_cleanup_outputs(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    artifacts = build_mainline_admission_cleanup(
        suite_dir=suite_dir,
        compare_dir=compare_dir,
        manifest_path=manifest_path,
    )
    summary = artifacts["summary"]
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"missing_evidence", "admission_fail_reasons"}
    }
    row["missing_evidence"] = json.dumps(summary.get("missing_evidence", []), ensure_ascii=False)
    row["admission_fail_reasons"] = json.dumps(summary.get("admission_fail_reasons", []), ensure_ascii=False)
    rows = [row]
    csv_path = tables_dir / "table_mainline_admission_cleanup.csv"
    md_path = tables_dir / "table_mainline_admission_cleanup.md"
    report_path = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv_rows(csv_path, rows)
    _write_text(md_path, "# Mainline Admission Cleanup\n\n" + _rows_to_markdown(rows))
    report_lines = [
        "# Mainline Admission Cleanup",
        "",
        f"- promotion_decision: `{summary.get('promotion_decision', '')}`",
        f"- promotion_ready: `{summary.get('promotion_ready', False)}`",
        f"- admission_status_suite: `{summary.get('admission_status_suite', '')}`",
        f"- admission_status_compare_side: `{summary.get('admission_status_compare_side', '')}`",
        f"- promotion_vs_admission_consistency_status: `{summary.get('promotion_vs_admission_consistency_status', '')}`",
        f"- evidence_hardness_status: `{summary.get('evidence_hardness_status', '')}`",
        f"- sample_contract_status: `{summary.get('sample_contract_status', '')}`",
        f"- large_sample_claim_status: `{summary.get('large_sample_claim_status', '')}`",
        f"- provider_availability_semantics_gap: `{summary.get('provider_availability_semantics_gap', False)}`",
        f"- missing_evidence: `{summary.get('missing_evidence', [])}`",
        f"- next_action_recommendation: `{summary.get('next_action_recommendation', '')}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    _write_json(snapshot_path, artifacts["snapshot"])
    return {
        "summary": summary,
        "csv_path": csv_path,
        "md_path": md_path,
        "report_path": report_path,
        "snapshot_path": snapshot_path,
    }
