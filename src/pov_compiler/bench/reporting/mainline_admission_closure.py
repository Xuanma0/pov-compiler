from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pov_compiler.bench.reporting.mainline_admission_cleanup import (
    ROOT,
    _read_json,
    _rows_to_markdown,
    _stable_relpath,
    _to_float,
    _write_csv_rows,
    _write_json,
    _write_text,
    build_mainline_admission_cleanup,
    build_sample_contract_summary,
)


def _resolve_manifest_path(suite_dir: str | Path, manifest_path: str | Path | None) -> Path | None:
    if manifest_path is not None:
        return Path(manifest_path).resolve()
    candidate = Path(suite_dir).resolve() / "manifest" / "experiment_manifest.yaml"
    if candidate.exists():
        return candidate
    return None


def _load_compare_summary(compare_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    compare_root = Path(compare_dir).resolve()
    if (compare_root / "compare").exists():
        compare_root = compare_root / "compare"
    return compare_root, _read_json(compare_root / "compare_summary.json")


def _load_decision_summary(compare_dir: str | Path) -> dict[str, Any]:
    compare_root = Path(compare_dir).resolve()
    decision_root = compare_root / "promotion_decision"
    if not decision_root.exists():
        decision_root = compare_root.parent / "promotion_decision"
    snapshot = _read_json(decision_root / "snapshot.json")
    summary = snapshot.get("promotion_decision_summary", {})
    return summary if isinstance(summary, dict) else {}


def _provider_semantics_gap(
    *,
    suite_dir: Path,
    compare_summary: dict[str, Any],
) -> tuple[bool, list[str], str]:
    admission_snapshot = _read_json(suite_dir / "admission_control" / "snapshot.json")
    normalization_snapshot = _read_json(suite_dir / "provider_normalization" / "snapshot.json")
    reachability_summary = _read_json(suite_dir / "provider_reachability" / "summary.json")
    admission_fail_reasons = admission_snapshot.get("admission_fail_reasons", [])
    if not isinstance(admission_fail_reasons, list):
        admission_fail_reasons = []
    provider_health_status = str(compare_summary.get("provider_health_status", "unknown"))
    proof_status = str(reachability_summary.get("proof_status", "unknown"))
    normalization_status = str(normalization_snapshot.get("normalization_status", "unknown"))
    semantics_gap = (
        any("provider_availability" in str(item) for item in admission_fail_reasons)
        and provider_health_status == "ok"
        and proof_status == "ok"
        and normalization_status in {"ok", "partial"}
    )
    effective_reasons = [
        str(item)
        for item in admission_fail_reasons
        if not semantics_gap or "provider_availability" not in str(item)
    ]
    if semantics_gap and not effective_reasons:
        cleanliness = "semantic_noise_only"
    elif effective_reasons:
        cleanliness = "noisy"
    elif provider_health_status == "ok" and proof_status == "ok" and normalization_status in {"ok", "partial"}:
        cleanliness = "clean"
    else:
        cleanliness = "unknown"
    return semantics_gap, effective_reasons, cleanliness


def build_harder_sample_contract_summary(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_manifest = _resolve_manifest_path(suite_dir, manifest_path)
    summary = build_sample_contract_summary(
        suite_dir=suite_dir,
        compare_dir=compare_dir,
        manifest_path=resolved_manifest,
    )
    thresholds = dict(summary.get("thresholds", {}))
    selected_uids = int(summary.get("selected_uids_count", 0) or 0)
    paired_sample_count = int(summary.get("paired_sample_count", 0) or 0)
    coverage_mean = _to_float(summary.get("coverage_mean"))
    coverage_min = _to_float(summary.get("coverage_min"))
    selected_gap = max(0, int(thresholds.get("selected_uids_count_adequate_min", 0)) - selected_uids)
    paired_gap = max(0, int(thresholds.get("paired_sample_count_adequate_min", 0)) - paired_sample_count)
    coverage_mean_gap = max(
        0.0,
        float(thresholds.get("coverage_mean_min", 0.0)) - float(coverage_mean if coverage_mean is not None else 0.0),
    )
    coverage_min_gap = max(
        0.0,
        float(thresholds.get("coverage_min_min", 0.0)) - float(coverage_min if coverage_min is not None else 0.0),
    )
    summary["harder_thresholds"] = thresholds
    summary["selected_uids_gap_to_adequate"] = selected_gap
    summary["paired_sample_gap_to_adequate"] = paired_gap
    summary["coverage_mean_gap_to_adequate"] = round(coverage_mean_gap, 6)
    summary["coverage_min_gap_to_adequate"] = round(coverage_min_gap, 6)
    return summary


def write_harder_sample_contract_outputs(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = build_harder_sample_contract_summary(
        suite_dir=suite_dir,
        compare_dir=compare_dir,
        manifest_path=manifest_path,
    )
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"freeze_details", "thresholds", "harder_thresholds"}
    }
    csv_path = tables_dir / "table_harder_sample_contract_summary.csv"
    md_path = tables_dir / "table_harder_sample_contract_summary.md"
    report_path = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv_rows(csv_path, [row])
    _write_text(md_path, "# Harder Sample Contract Summary\n\n" + _rows_to_markdown([row]))
    report_lines = [
        "# Harder Sample Contract",
        "",
        f"- sample_contract_status: `{summary.get('sample_contract_status', '')}`",
        f"- large_sample_claim_status: `{summary.get('large_sample_claim_status', '')}`",
        f"- wording_recommendation: `{summary.get('wording_recommendation', '')}`",
        f"- selected_uids_count: `{summary.get('selected_uids_count', 0)}`",
        f"- paired_sample_count: `{summary.get('paired_sample_count', 0)}`",
        f"- selected_uids_gap_to_adequate: `{summary.get('selected_uids_gap_to_adequate', 0)}`",
        f"- paired_sample_gap_to_adequate: `{summary.get('paired_sample_gap_to_adequate', 0)}`",
        f"- coverage_mean_gap_to_adequate: `{summary.get('coverage_mean_gap_to_adequate', 0.0)}`",
        f"- coverage_min_gap_to_adequate: `{summary.get('coverage_min_gap_to_adequate', 0.0)}`",
        f"- provider_signature_match: `{summary.get('provider_signature_match', False)}`",
        f"- provider_noise_summary_hash_match: `{summary.get('provider_noise_summary_hash_match', False)}`",
        f"- perception_signature_match: `{summary.get('perception_signature_match', False)}`",
        f"- freeze_contract_status: `{summary.get('freeze_contract_status', '')}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    _write_json(snapshot_path, {"harder_sample_contract_summary": summary})
    return {
        "summary": summary,
        "csv_path": csv_path,
        "md_path": md_path,
        "report_path": report_path,
        "snapshot_path": snapshot_path,
    }


def refresh_persistent_memory_main_compare_outputs(
    *,
    run_a: str | Path,
    run_b: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    compare_root = Path(out_dir).resolve() / "compare"
    summary_path = compare_root / "compare_summary.json"
    snapshot_path = compare_root / "snapshot.json"
    summary = _read_json(summary_path)
    suite_root = Path(run_b).resolve()
    harder_summary = build_harder_sample_contract_summary(
        suite_dir=suite_root,
        compare_dir=compare_root,
        manifest_path=manifest_path,
    )
    semantics_gap, effective_reasons, provider_cleanliness_status = _provider_semantics_gap(
        suite_dir=suite_root,
        compare_summary=summary,
    )
    promotion_ready = bool(
        summary.get("alignment_ok", False)
        and summary.get("persistent_memory_main_status") == "improved"
        and (_to_float(summary.get("mean_delta_mrr_strict")) or 0.0) > 0.0
    )
    mainline_admission_ready = bool(
        promotion_ready
        and harder_summary.get("sample_contract_status") == "adequate"
        and harder_summary.get("large_sample_claim_status") == "supported"
        and not semantics_gap
        and provider_cleanliness_status in {"clean", "semantic_noise_only"}
    )
    summary["uid_set_id"] = summary.get("uid_set_id") or harder_summary.get("uid_set_id", "")
    summary["sample_signature_hash"] = summary.get("sample_signature_hash") or harder_summary.get("sample_signature_hash", "")
    summary["selected_uids_count"] = int(summary.get("selected_uids_count", harder_summary.get("selected_uids_count", 0)) or 0)
    summary["paired_sample_count"] = int(summary.get("paired_sample_count", harder_summary.get("paired_sample_count", 0)) or 0)
    summary["sample_contract_status"] = harder_summary.get("sample_contract_status")
    summary["large_sample_claim_status"] = harder_summary.get("large_sample_claim_status")
    summary["promotion_ready"] = promotion_ready
    summary["mainline_admission_ready"] = mainline_admission_ready
    summary["provider_semantics_gap"] = semantics_gap
    summary["provider_cleanliness_status"] = provider_cleanliness_status
    summary["admission_noise_reasons_effective"] = effective_reasons
    summary["coverage_mean"] = harder_summary.get("coverage_mean")
    summary["coverage_min"] = harder_summary.get("coverage_min")
    summary["harder_thresholds"] = harder_summary.get("harder_thresholds", {})
    _write_json(summary_path, summary)

    snapshot = _read_json(snapshot_path)
    snapshot["persistent_memory_main_compare_summary"] = summary
    snapshot["harder_sample_contract_summary"] = harder_summary
    snapshot["provider_semantics_gap"] = semantics_gap
    snapshot["provider_cleanliness_status"] = provider_cleanliness_status
    snapshot["admission_noise_reasons_effective"] = effective_reasons
    _write_json(snapshot_path, snapshot)
    return summary


def write_refreshed_persistent_memory_main_decision_outputs(
    *,
    compare_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    compare_root, compare_summary = _load_compare_summary(compare_dir)
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    promotion_ready = bool(compare_summary.get("promotion_ready", False))
    mainline_admission_ready = bool(compare_summary.get("mainline_admission_ready", False))
    strict_gain = (_to_float(compare_summary.get("mean_delta_mrr_strict")) or 0.0) > 0.0
    memory_gain = all(
        (_to_float(compare_summary.get(key)) or 0.0)
        > (_to_float(compare_summary.get(key.replace("_b", "_a"))) or 0.0)
        for key in (
            "lost_object_query_support_rate_b",
            "chain_object_grounding_support_rate_b",
            "reappearance_support_rate_b",
            "query_strength_coverage_b",
        )
    )
    if promotion_ready:
        promotion_decision = "promote_persistent_memory_to_mainline"
    elif strict_gain or memory_gain:
        promotion_decision = "need_cleaner_large_sample_run"
    else:
        promotion_decision = "persistent_memory_gain_insufficient"
    if mainline_admission_ready:
        admission_decision = "mainline_ready"
    elif promotion_ready:
        admission_decision = "cleanup_needed"
    else:
        admission_decision = "not_ready"
    if promotion_ready and mainline_admission_ready:
        decision_confidence = "high"
    elif promotion_ready or strict_gain:
        decision_confidence = "medium"
    else:
        decision_confidence = "low"
    if strict_gain:
        strict_gain_status = "strict_main_gain_improved"
    elif (_to_float(compare_summary.get("mean_delta_mrr_relaxed")) or 0.0) > 0.0:
        strict_gain_status = "relaxed_only_gain"
    else:
        strict_gain_status = "no_strict_gain"
    if memory_gain:
        memory_signal_status = "memory_signals_improved"
    elif compare_summary.get("persistent_memory_main_status") == "improved":
        memory_signal_status = "mixed_memory_signals"
    else:
        memory_signal_status = "memory_signals_not_improved"
    if mainline_admission_ready:
        recommended_next_step = "promote_persistent_memory_to_mainline"
    elif promotion_ready:
        recommended_next_step = "need_cleaner_large_sample_run"
    elif strict_gain:
        recommended_next_step = "need_retrieval_side_fix"
    else:
        recommended_next_step = "persistent_memory_gain_insufficient"

    summary = {
        "promotion_decision": promotion_decision,
        "admission_decision": admission_decision,
        "promotion_ready": promotion_ready,
        "mainline_admission_ready": mainline_admission_ready,
        "decision_confidence": decision_confidence,
        "strict_gain_status": strict_gain_status,
        "memory_signal_status": memory_signal_status,
        "large_sample_claim_status": compare_summary.get("large_sample_claim_status", "unsupported"),
        "recommended_next_step": recommended_next_step,
        "mean_delta_mrr_strict": compare_summary.get("mean_delta_mrr_strict"),
        "mean_delta_mrr_relaxed": compare_summary.get("mean_delta_mrr_relaxed"),
        "query_strength_delta": (
            (_to_float(compare_summary.get("query_strength_coverage_b")) or 0.0)
            - (_to_float(compare_summary.get("query_strength_coverage_a")) or 0.0)
        ),
        "weak_query_groups_delta": (
            int(compare_summary.get("weak_query_groups_count_b", 0) or 0)
            - int(compare_summary.get("weak_query_groups_count_a", 0) or 0)
        ),
        "lost_object_delta": (
            (_to_float(compare_summary.get("lost_object_query_support_rate_b")) or 0.0)
            - (_to_float(compare_summary.get("lost_object_query_support_rate_a")) or 0.0)
        ),
        "chain_object_grounding_delta": (
            (_to_float(compare_summary.get("chain_object_grounding_support_rate_b")) or 0.0)
            - (_to_float(compare_summary.get("chain_object_grounding_support_rate_a")) or 0.0)
        ),
        "reappearance_delta": (
            (_to_float(compare_summary.get("reappearance_support_rate_b")) or 0.0)
            - (_to_float(compare_summary.get("reappearance_support_rate_a")) or 0.0)
        ),
        "object_persistence_delta": (
            (_to_float(compare_summary.get("object_persistence_support_rate_b")) or 0.0)
            - (_to_float(compare_summary.get("object_persistence_support_rate_a")) or 0.0)
        ),
        "paired_sample_count": compare_summary.get("paired_sample_count", 0),
        "provider_health_status": compare_summary.get("provider_health_status", "unknown"),
        "provider_semantics_gap": compare_summary.get("provider_semantics_gap", False),
        "admission_status_b": compare_summary.get("admission_status_b", "unknown"),
        "calibration_status_b": compare_summary.get("calibration_status_b", "unknown"),
    }
    rows = [summary]
    csv_path = tables_dir / "table_persistent_memory_main_decision.csv"
    md_path = tables_dir / "table_persistent_memory_main_decision.md"
    report_path = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv_rows(csv_path, rows)
    _write_text(md_path, "# Persistent Memory Main Decision\n\n" + _rows_to_markdown(rows))
    report_lines = [
        "# Persistent Memory Main Decision",
        "",
        f"- promotion_decision: `{summary['promotion_decision']}`",
        f"- admission_decision: `{summary['admission_decision']}`",
        f"- promotion_ready: `{summary['promotion_ready']}`",
        f"- mainline_admission_ready: `{summary['mainline_admission_ready']}`",
        f"- decision_confidence: `{summary['decision_confidence']}`",
        f"- strict_gain_status: `{summary['strict_gain_status']}`",
        f"- memory_signal_status: `{summary['memory_signal_status']}`",
        f"- large_sample_claim_status: `{summary['large_sample_claim_status']}`",
        f"- recommended_next_step: `{summary['recommended_next_step']}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    _write_json(snapshot_path, {"promotion_decision_summary": summary})
    return {
        "promotion_decision_summary": summary,
        "table_csv": csv_path,
        "table_md": md_path,
        "report_md": report_path,
        "snapshot_json": snapshot_path,
    }


def write_refreshed_mainline_admission_cleanup_outputs(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest = _resolve_manifest_path(suite_dir, manifest_path)
    artifacts = build_mainline_admission_cleanup(
        suite_dir=suite_dir,
        compare_dir=compare_dir,
        manifest_path=resolved_manifest,
    )
    compare_root, compare_summary = _load_compare_summary(compare_dir)
    harder_summary = build_harder_sample_contract_summary(
        suite_dir=suite_dir,
        compare_dir=compare_root,
        manifest_path=resolved_manifest,
    )
    semantics_gap, effective_reasons, provider_cleanliness_status = _provider_semantics_gap(
        suite_dir=Path(suite_dir).resolve(),
        compare_summary=compare_summary,
    )
    summary = dict(artifacts["summary"])
    sample_contract_status = str(harder_summary.get("sample_contract_status", "insufficient"))
    large_sample_claim_status = str(harder_summary.get("large_sample_claim_status", "unsupported"))
    evidence_hardness_status = (
        "supported"
        if sample_contract_status == "adequate" and not semantics_gap and provider_cleanliness_status == "clean"
        else summary.get("evidence_hardness_status", "soft")
    )
    mainline_admission_cleanup_status = str(summary.get("mainline_admission_cleanup_status", "unresolved"))
    if sample_contract_status == "adequate" and provider_cleanliness_status in {"clean", "semantic_noise_only"}:
        mainline_admission_cleanup_status = "improved"
    elif not compare_summary.get("alignment_ok", False):
        mainline_admission_cleanup_status = "blocked"
    summary.update(
        {
            "sample_contract_status": sample_contract_status,
            "coverage_contract_status": harder_summary.get("coverage_contract_status", summary.get("coverage_contract_status")),
            "freeze_contract_status": harder_summary.get("freeze_contract_status", summary.get("freeze_contract_status")),
            "provider_semantics_gap": semantics_gap,
            "provider_availability_semantics_gap": semantics_gap,
            "provider_cleanliness_status": provider_cleanliness_status,
            "evidence_hardness_status": evidence_hardness_status,
            "large_sample_claim_status": large_sample_claim_status,
            "mainline_admission_cleanup_status": mainline_admission_cleanup_status,
            "admission_noise_reasons_effective": effective_reasons,
        }
    )
    if mainline_admission_cleanup_status == "improved" and large_sample_claim_status == "supported":
        summary["next_action_recommendation"] = "mainline_ready"
    elif sample_contract_status != "adequate":
        summary["next_action_recommendation"] = "need_sample_contract_hardening"
    elif semantics_gap:
        summary["next_action_recommendation"] = "promotion_valid_admission_partial"
    else:
        summary["next_action_recommendation"] = "need_cleaner_run_contract"

    row = {
        key: value
        for key, value in summary.items()
        if key not in {"missing_evidence", "admission_fail_reasons", "admission_noise_reasons_effective"}
    }
    row["missing_evidence"] = json.dumps(summary.get("missing_evidence", []), ensure_ascii=False)
    row["admission_fail_reasons"] = json.dumps(summary.get("admission_fail_reasons", []), ensure_ascii=False)
    row["admission_noise_reasons_effective"] = json.dumps(effective_reasons, ensure_ascii=False)
    csv_path = tables_dir / "table_mainline_admission_cleanup.csv"
    md_path = tables_dir / "table_mainline_admission_cleanup.md"
    report_path = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv_rows(csv_path, [row])
    _write_text(md_path, "# Mainline Admission Cleanup\n\n" + _rows_to_markdown([row]))
    report_lines = [
        "# Mainline Admission Cleanup",
        "",
        f"- promotion_decision: `{summary.get('promotion_decision', '')}`",
        f"- promotion_ready: `{summary.get('promotion_ready', False)}`",
        f"- admission_status_suite: `{summary.get('admission_status_suite', '')}`",
        f"- admission_status_compare_side: `{summary.get('admission_status_compare_side', '')}`",
        f"- provider_cleanliness_status: `{summary.get('provider_cleanliness_status', '')}`",
        f"- provider_semantics_gap: `{summary.get('provider_semantics_gap', False)}`",
        f"- sample_contract_status: `{summary.get('sample_contract_status', '')}`",
        f"- large_sample_claim_status: `{summary.get('large_sample_claim_status', '')}`",
        f"- evidence_hardness_status: `{summary.get('evidence_hardness_status', '')}`",
        f"- mainline_admission_cleanup_status: `{summary.get('mainline_admission_cleanup_status', '')}`",
        f"- next_action_recommendation: `{summary.get('next_action_recommendation', '')}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    _write_json(
        snapshot_path,
        {
            "mainline_admission_cleanup_summary": summary,
            "harder_sample_contract_summary": harder_summary,
            "compare_summary": compare_summary,
        },
    )
    return {
        "summary": summary,
        "csv_path": csv_path,
        "md_path": md_path,
        "report_path": report_path,
        "snapshot_path": snapshot_path,
    }


def build_mainline_admission_closure(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    decision_dir: str | Path,
    cleanup_dir: str | Path,
    sample_contract_dir: str | Path,
) -> dict[str, Any]:
    compare_root, compare_summary = _load_compare_summary(compare_dir)
    decision_snapshot = _read_json(Path(decision_dir).resolve() / "snapshot.json")
    cleanup_snapshot = _read_json(Path(cleanup_dir).resolve() / "snapshot.json")
    sample_snapshot = _read_json(Path(sample_contract_dir).resolve() / "snapshot.json")
    decision_summary = decision_snapshot.get("promotion_decision_summary", {})
    if not isinstance(decision_summary, dict):
        decision_summary = {}
    cleanup_summary = cleanup_snapshot.get("mainline_admission_cleanup_summary", {})
    if not isinstance(cleanup_summary, dict):
        cleanup_summary = {}
    sample_summary = sample_snapshot.get("harder_sample_contract_summary", {})
    if not isinstance(sample_summary, dict):
        sample_summary = {}

    promotion_ready = bool(decision_summary.get("promotion_ready", False))
    mainline_admission_ready = bool(decision_summary.get("mainline_admission_ready", False))
    large_sample_claim_status = str(sample_summary.get("large_sample_claim_status", compare_summary.get("large_sample_claim_status", "unsupported")))
    evidence_hardness_status = str(cleanup_summary.get("evidence_hardness_status", "soft"))
    provider_cleanliness_status = str(cleanup_summary.get("provider_cleanliness_status", "unknown"))
    provider_semantics_gap = bool(cleanup_summary.get("provider_semantics_gap", False))
    if promotion_ready and mainline_admission_ready and large_sample_claim_status == "supported":
        closure_status = "closed"
        recommended_wording = "large-sample real main experiment"
        recommended_next_step = "mainline_admission_closed"
        closure_basis = "promotion_ready_and_harder_sample_adequate"
    elif promotion_ready and large_sample_claim_status in {"supported", "supported_with_caveat"}:
        closure_status = "partial_but_harder"
        recommended_wording = str(sample_summary.get("wording_recommendation", "large-sample real main experiment (supported with caveat)"))
        recommended_next_step = (
            "need_provider_semantics_cleanup" if provider_semantics_gap else "keep_mainline_with_caveat"
        )
        closure_basis = "promotion_valid_but_admission_cleanup_pending"
    elif promotion_ready:
        closure_status = "still_partial"
        recommended_wording = str(sample_summary.get("wording_recommendation", "real main experiment with moderate paired sample coverage"))
        recommended_next_step = "need_more_harder_sample"
        closure_basis = "promotion_valid_but_harder_sample_insufficient"
    else:
        closure_status = "blocked"
        recommended_wording = "real experiment preview only"
        recommended_next_step = "soften_large_sample_wording"
        closure_basis = "promotion_not_ready"

    summary = {
        "promotion_decision": decision_summary.get("promotion_decision", ""),
        "promotion_ready": promotion_ready,
        "mainline_admission_ready": mainline_admission_ready,
        "persistent_memory_main_status": compare_summary.get("persistent_memory_main_status", "unknown"),
        "mainline_admission_cleanup_status": cleanup_summary.get("mainline_admission_cleanup_status", "unresolved"),
        "sample_contract_status": sample_summary.get("sample_contract_status", "insufficient"),
        "large_sample_claim_status": large_sample_claim_status,
        "evidence_hardness_status": evidence_hardness_status,
        "provider_cleanliness_status": provider_cleanliness_status,
        "provider_semantics_gap": provider_semantics_gap,
        "closure_basis": closure_basis,
        "mainline_admission_closure_status": closure_status,
        "recommended_wording": recommended_wording,
        "recommended_next_step": recommended_next_step,
        "compare_dir": _stable_relpath(compare_root),
        "suite_dir": _stable_relpath(Path(suite_dir).resolve()),
    }
    return {
        "summary": summary,
        "compare_summary": compare_summary,
        "decision_summary": decision_summary,
        "cleanup_summary": cleanup_summary,
        "sample_summary": sample_summary,
    }


def write_mainline_admission_closure_outputs(
    *,
    suite_dir: str | Path,
    compare_dir: str | Path,
    decision_dir: str | Path,
    cleanup_dir: str | Path,
    sample_contract_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    payload = build_mainline_admission_closure(
        suite_dir=suite_dir,
        compare_dir=compare_dir,
        decision_dir=decision_dir,
        cleanup_dir=cleanup_dir,
        sample_contract_dir=sample_contract_dir,
    )
    summary = payload["summary"]
    csv_path = tables_dir / "table_mainline_admission_closure.csv"
    md_path = tables_dir / "table_mainline_admission_closure.md"
    report_path = out_root / "report.md"
    snapshot_path = out_root / "snapshot.json"
    _write_csv_rows(csv_path, [summary])
    _write_text(md_path, "# Mainline Admission Closure\n\n" + _rows_to_markdown([summary]))
    report_lines = [
        "# Mainline Admission Closure",
        "",
        f"- promotion_decision: `{summary.get('promotion_decision', '')}`",
        f"- promotion_ready: `{summary.get('promotion_ready', False)}`",
        f"- mainline_admission_ready: `{summary.get('mainline_admission_ready', False)}`",
        f"- mainline_admission_closure_status: `{summary.get('mainline_admission_closure_status', '')}`",
        f"- large_sample_claim_status: `{summary.get('large_sample_claim_status', '')}`",
        f"- recommended_wording: `{summary.get('recommended_wording', '')}`",
        f"- recommended_next_step: `{summary.get('recommended_next_step', '')}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    _write_json(snapshot_path, {"mainline_admission_closure_summary": summary})
    return {
        "summary": summary,
        "csv_path": csv_path,
        "md_path": md_path,
        "report_path": report_path,
        "snapshot_path": snapshot_path,
    }
