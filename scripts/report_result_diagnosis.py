from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.result_diagnosis import write_result_diagnosis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a result-diagnosis report from a benchmark suite output.")
    parser.add_argument("--suite-dir", required=True, help="Benchmark suite root")
    parser.add_argument("--out_dir", required=True, help="Output directory for diagnosis artifacts")
    parser.add_argument("--near-zero-threshold", type=float, default=1e-6)
    parser.add_argument("--effect-size-threshold", type=float, default=1e-9)
    parser.add_argument("--significance-threshold", type=float, default=0.50)
    parser.add_argument("--provider-telemetry-dir", default=None, help="Optional directory with supplemental provider telemetry CSVs")
    parser.add_argument("--provider-normalization-dir", default=None, help="Optional directory with normalized provider telemetry outputs")
    parser.add_argument("--provider-reachability-dir", default=None, help="Optional directory with provider reachability proof outputs")
    parser.add_argument("--repeatability-audit-dir", default=None, help="Optional directory with repeatability audit outputs")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_json_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _provider_result_state(snapshot: dict, reachability: dict) -> str:
    proof_status = str(reachability.get("proof_status", "")).strip()
    real_call_status = str(reachability.get("real_call_status", "")).strip()
    if proof_status in {"fail", ""} or real_call_status in {"missing_key", "server_unavailable", "auth_fail", "unknown"}:
        return "provider_unavailable"
    if float(snapshot.get("effect_size_nonzero_rate", 0.0) or 0.0) > 0.0 or int(snapshot.get("selected_uids_count", 0) or 0) > 0:
        return "provider_reachable_and_results_present"
    return "provider_reachable_but_weak_results"


def _merged_provider_summary(base_summary: dict, reachability: dict) -> dict:
    merged = dict(base_summary or {})
    if not reachability:
        return merged
    if "availability" not in merged or str(merged.get("availability", "")).strip() in {"", "unavailable"}:
        if str(reachability.get("proof_status", "")).strip() == "ok":
            merged["availability"] = "ok"
        elif str(reachability.get("proof_status", "")).strip() == "partial":
            merged["availability"] = "partial"
        else:
            merged["availability"] = "provider_unavailable"
    merged["proof_status"] = str(reachability.get("proof_status", "fail"))
    merged["structured_output_supported"] = reachability.get("structured_output_supported", "unknown")
    merged["usage_present"] = bool(reachability.get("usage_present", False))
    merged["real_call_status"] = str(reachability.get("real_call_status", "missing_or_unavailable"))
    merged["reachability_base_url"] = str(reachability.get("base_url", ""))
    return merged


def _rewrite_csv_with_reachability(table_path: Path, merged_summary: dict) -> None:
    if not table_path.exists():
        return
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    if not rows or not fieldnames:
        return
    for row in rows:
        row["provider_noise_summary"] = json.dumps(merged_summary, ensure_ascii=False, sort_keys=True)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_report_context(report_path: Path, reachability: dict, merged_summary: dict, provider_result_state: str) -> None:
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    lines = [
        "",
        "## Provider Reachability",
        "",
        f"- proof_status: `{reachability.get('proof_status', 'fail')}`",
        f"- real_call_status: `{reachability.get('real_call_status', 'missing_or_unavailable')}`",
        f"- structured_output_supported: `{reachability.get('structured_output_supported', 'unknown')}`",
        f"- usage_present: `{reachability.get('usage_present', False)}`",
        f"- provider_result_state: `{provider_result_state}`",
        f"- provider_noise_summary_merged: `{json.dumps(merged_summary, ensure_ascii=False, sort_keys=True)}`",
    ]
    report_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _stability_interpretation(
    *,
    snapshot: dict,
    repeatability_snapshot: dict,
    query_strength_snapshot: dict,
    delta_audit_snapshot: dict,
) -> str:
    repeatability_status = str(repeatability_snapshot.get("repeatability_status", "")).strip()
    flag_counts = repeatability_snapshot.get("stability_flag_counts", {})
    if not isinstance(flag_counts, dict):
        flag_counts = {}
    provider_noise_flags = int(flag_counts.get("provider_noise_driven", 0) or 0)
    small_sample_flags = int(flag_counts.get("small_sample_driven", 0) or 0) + int(flag_counts.get("weak_evidence", 0) or 0)
    weak_query_groups = int(query_strength_snapshot.get("weak_query_groups_count", 0) or 0)
    query_recommendation = str(query_strength_snapshot.get("main_recommendation", "")).strip()
    delta_recommendation = str(delta_audit_snapshot.get("main_recommendation", "")).strip()
    if provider_noise_flags > 0:
        return "provider_unstable"
    if repeatability_status in {"weak", "partial"} and small_sample_flags > 0:
        return "sample_too_small"
    if weak_query_groups > 0 and query_recommendation in {"increase_sample_size", "strengthen_signal_support", "keep_for_analysis_only"}:
        return "query_too_weak"
    if repeatability_status == "ok" and delta_recommendation == "algorithm_no_effect_detected":
        return "stable_but_algorithm_no_effect"
    if str(snapshot.get("provider_result_state", "")).strip():
        return str(snapshot.get("provider_result_state", "")).strip()
    return "repeatability_not_available"


def _append_repeatability_context(
    report_path: Path,
    repeatability_snapshot: dict,
    query_strength_snapshot: dict,
    stability_interpretation: str,
) -> None:
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    lines = [
        "",
        "## Repeatability",
        "",
        f"- repeatability_status: `{repeatability_snapshot.get('repeatability_status', 'weak')}`",
        f"- stability_flag_counts: `{repeatability_snapshot.get('stability_flag_counts', {})}`",
        f"- provider_noise_rate_mean: `{repeatability_snapshot.get('provider_noise_rate_mean', 0.0)}`",
        f"- stability_interpretation: `{stability_interpretation}`",
        f"- query_strength_main_recommendation: `{query_strength_snapshot.get('main_recommendation', '')}`",
    ]
    report_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    provider_telemetry_dir = args.provider_telemetry_dir
    if not provider_telemetry_dir:
        candidate = Path(args.suite_dir).resolve() / "provider_telemetry"
        if candidate.exists():
            provider_telemetry_dir = str(candidate)
    provider_normalization_dir = args.provider_normalization_dir
    if not provider_normalization_dir:
        candidate = Path(args.suite_dir).resolve() / "provider_normalization"
        if candidate.exists():
            provider_normalization_dir = str(candidate)
    provider_reachability_dir = args.provider_reachability_dir
    if not provider_reachability_dir:
        candidate = Path(args.suite_dir).resolve() / "provider_reachability"
        if candidate.exists():
            provider_reachability_dir = str(candidate)
    repeatability_audit_dir = args.repeatability_audit_dir
    if not repeatability_audit_dir:
        candidate = Path(args.suite_dir).resolve() / "repeatability_audit"
        if candidate.exists():
            repeatability_audit_dir = str(candidate)
    outputs = write_result_diagnosis_outputs(
        suite_dir=args.suite_dir,
        out_dir=args.out_dir,
        near_zero_threshold=float(args.near_zero_threshold),
        effect_size_threshold=float(args.effect_size_threshold),
        significance_threshold=float(args.significance_threshold),
        provider_telemetry_dir=provider_telemetry_dir,
        provider_normalization_dir=provider_normalization_dir,
    )
    snapshot_path = Path(outputs["snapshot_json"]).resolve()
    snapshot = _read_json(snapshot_path)
    reachability_summary = _read_json(Path(provider_reachability_dir).resolve() / "summary.json") if provider_reachability_dir else {}
    repeatability_snapshot = _read_json(Path(repeatability_audit_dir).resolve() / "snapshot.json") if repeatability_audit_dir else {}
    query_strength_snapshot = _read_json(Path(args.suite_dir).resolve() / "query_strength_audit" / "snapshot.json")
    delta_audit_snapshot = _read_json(Path(args.suite_dir).resolve() / "delta_audit" / "snapshot.json")
    if reachability_summary:
        merged_summary = _merged_provider_summary(snapshot.get("provider_noise_summary", {}), reachability_summary)
        snapshot["provider_noise_summary"] = merged_summary
        snapshot["provider_reachability_dir"] = str(Path(provider_reachability_dir).resolve())
        snapshot["provider_result_state"] = _provider_result_state(snapshot, reachability_summary)
        snapshot["normalization_status"] = (
            str(snapshot.get("normalization_status", "")).strip()
            or str(merged_summary.get("normalization_status", "")).strip()
            or "fallback"
        )
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        _rewrite_csv_with_reachability(Path(outputs["table_csv"]).resolve(), merged_summary)
        _append_report_context(Path(outputs["report_md"]).resolve(), reachability_summary, merged_summary, snapshot["provider_result_state"])
        outputs["provider_noise_summary"] = merged_summary
    if repeatability_snapshot:
        provider_noise_summary = dict(snapshot.get("provider_noise_summary", {}))
        provider_noise_summary["repeatability_status"] = str(repeatability_snapshot.get("repeatability_status", "weak"))
        provider_noise_summary["repeatability_provider_noise_rate_mean"] = repeatability_snapshot.get("provider_noise_rate_mean", 0.0)
        provider_noise_summary["repeatability_stability_flag_counts"] = repeatability_snapshot.get("stability_flag_counts", {})
        snapshot["provider_noise_summary"] = provider_noise_summary
        snapshot["repeatability_audit_dir"] = str(Path(repeatability_audit_dir).resolve())
        snapshot["repeatability_status"] = str(repeatability_snapshot.get("repeatability_status", "weak"))
        snapshot["repeatability_summary"] = {
            "stability_flag_counts": repeatability_snapshot.get("stability_flag_counts", {}),
            "provider_noise_rate_mean": repeatability_snapshot.get("provider_noise_rate_mean", 0.0),
            "repeat_runs_total": repeatability_snapshot.get("repeat_runs_total", 0),
        }
        snapshot["stability_interpretation"] = _stability_interpretation(
            snapshot=snapshot,
            repeatability_snapshot=repeatability_snapshot,
            query_strength_snapshot=query_strength_snapshot,
            delta_audit_snapshot=delta_audit_snapshot,
        )
        if snapshot["stability_interpretation"] not in snapshot.get("diagnosis_recommendations", []):
            snapshot["diagnosis_recommendations"] = list(snapshot.get("diagnosis_recommendations", [])) + [snapshot["stability_interpretation"]]
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        _append_repeatability_context(
            Path(outputs["report_md"]).resolve(),
            repeatability_snapshot,
            query_strength_snapshot,
            str(snapshot.get("stability_interpretation", "")),
        )
        outputs["provider_noise_summary"] = snapshot.get("provider_noise_summary", outputs["provider_noise_summary"])
    print(f"rows_total={outputs['rows_total']}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_report={outputs['report_md']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"no_data_reason_counts={outputs['overall_no_data_reason_counts']}")
    print(f"provider_noise_summary={outputs['provider_noise_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
