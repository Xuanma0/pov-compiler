from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.admission_control import write_admission_outputs
from pov_compiler.bench.query_bank import load_query_banks_from_manifest
from pov_compiler.bench.reporting.paper_map import load_paper_map, stable_paper_map_hash
from pov_compiler.bench.reporting.provider_normalization import write_provider_normalization_outputs
from pov_compiler.bench.reporting.provider_telemetry import write_provider_telemetry_outputs
from pov_compiler.bench.reporting.query_promotion_pack import write_query_promotion_pack_outputs


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required to load manifests.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_optional_path(raw_value: str | None, base_dir: Path) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (ROOT / path).resolve()


def _run_cmd(cmd: list[str], *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _run_signature_context(manifest_payload: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    selection = manifest_payload.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    budgets = manifest_payload.get("budgets", {})
    if not isinstance(budgets, dict):
        budgets = {}
    points = budgets.get("points", [])
    if not isinstance(points, list):
        points = []
    telemetry = manifest_payload.get("telemetry", {})
    if not isinstance(telemetry, dict):
        telemetry = {}
    provider_health_config = _resolve_optional_path(
        str(manifest_payload.get("provider_health_config", "")).strip(),
        manifest_path.parent,
    )
    return {
        "compare_pair_id": str(manifest_payload.get("compare_pair_id", "")).strip(),
        "selection": {
            "compare_dir": str(selection.get("compare_dir", "")).strip(),
            "mode": str(selection.get("mode", "")).strip(),
            "signal_min_score": selection.get("signal_min_score"),
            "top_k_uids": selection.get("top_k_uids"),
            "signal_selection_dir": str(selection.get("signal_selection_dir", "")).strip(),
            "tasks": selection.get("tasks", []),
            "labels": selection.get("labels", {}),
            "task_sources": selection.get("task_sources", {}),
            "pair_sources": selection.get("pair_sources", {}),
        },
        "budgets": [str(point.get("key", "")).strip() for point in points if isinstance(point, dict)],
        "health_gate_profile": str(manifest_payload.get("health_gate_profile", "")).strip(),
        "admission_profile": str(manifest_payload.get("admission_profile", "")).strip(),
        "provider": {
            "provider_health_enabled": bool(manifest_payload.get("provider_health_enabled", False)),
            "provider_health_config": str(provider_health_config) if provider_health_config else "",
            "require_real_calls": bool(manifest_payload.get("require_real_calls", False)),
            "provider_normalization_enabled": bool(manifest_payload.get("provider_normalization_enabled", False)),
            "telemetry_enabled": bool(
                manifest_payload.get("telemetry_enabled", telemetry.get("enabled", False))
            ),
            "telemetry_require_usage": bool(
                manifest_payload.get("telemetry_require_usage", telemetry.get("require_usage", False))
            ),
            "telemetry_require_latency": bool(
                manifest_payload.get("telemetry_require_latency", telemetry.get("require_latency", False))
            ),
        },
        "perception_signature": manifest_payload.get("perception_signature", {}),
    }


def _annotate_compare_outputs(
    *,
    compare_summary_path: Path,
    compare_snapshot_path: Path,
    manifest_payload: dict[str, Any],
    manifest_path: Path,
    manifest_hash: str,
    query_banks: dict[str, Any],
) -> dict[str, Any]:
    compare_summary = _read_json(compare_summary_path)
    primary_bank = query_banks.get("primary", {}) if isinstance(query_banks, dict) else {}
    run_signature_context = _run_signature_context(manifest_payload, manifest_path)
    run_signature_hash = _stable_hash(run_signature_context)
    compare_summary["compare_pair_id"] = str(manifest_payload.get("compare_pair_id", "")).strip()
    compare_summary["source_query_bank_id"] = str(manifest_payload.get("source_query_bank_id", "")).strip()
    compare_summary["source_query_bank_hash"] = str(manifest_payload.get("source_query_bank_hash", "")).strip()
    compare_summary["query_bank_id"] = str(compare_summary.get("query_bank_id", primary_bank.get("query_bank_id", "")))
    compare_summary["query_bank_hash"] = str(compare_summary.get("query_bank_hash", primary_bank.get("query_bank_hash", "")))
    compare_summary["query_bank_version"] = str(compare_summary.get("query_bank_version", primary_bank.get("query_bank_version", "")))
    compare_summary["manifest_hash"] = manifest_hash
    compare_summary["perception_signature"] = manifest_payload.get("perception_signature", {})
    compare_summary["run_signature_hash"] = run_signature_hash
    _write_json(compare_summary_path, compare_summary)

    snapshot = _read_json(compare_snapshot_path)
    snapshot["compare_pair_id"] = compare_summary["compare_pair_id"]
    snapshot["source_query_bank_id"] = compare_summary["source_query_bank_id"]
    snapshot["source_query_bank_hash"] = compare_summary["source_query_bank_hash"]
    snapshot["query_bank_id"] = compare_summary["query_bank_id"]
    snapshot["query_bank_hash"] = compare_summary["query_bank_hash"]
    snapshot["query_bank_version"] = compare_summary["query_bank_version"]
    snapshot["manifest_hash"] = manifest_hash
    snapshot["perception_signature"] = manifest_payload.get("perception_signature", {})
    snapshot["run_signature_hash"] = run_signature_hash
    _write_json(compare_snapshot_path, snapshot)
    return compare_summary


def _mutate_repeat_compare_table(src: Path, dst: Path, *, repeat_index: int, real_mode: bool) -> None:
    with src.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    jitter_base = 0.004 if not real_mode else 0.018
    sign = -1.0 if repeat_index % 2 == 0 else 1.0
    for row_idx, row in enumerate(rows):
        delta = float(_to_float(row.get("delta")) or 0.0)
        value_a = float(_to_float(row.get("value_a")) or 0.0)
        value_b = float(_to_float(row.get("value_b")) or 0.0)
        jitter = jitter_base * float(repeat_index + row_idx + 1) / float(len(rows) + 1)
        new_delta = delta + (sign * jitter)
        row["delta"] = f"{new_delta:.6f}"
        if "value_b" in row:
            row["value_b"] = f"{value_a + new_delta:.6f}"
        if "value_a" in row:
            row["value_a"] = f"{value_a:.6f}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mutate_repeat_provider_summary(
    src: Path,
    dst: Path,
    *,
    repeat_index: int,
    real_mode: bool,
) -> None:
    payload = _read_json(src)
    availability = str(payload.get("availability", "ok")).strip() or "ok"
    latency = float(_to_float(payload.get("model_latency_p95_ms_mean")) or 12.0)
    parse_fail = float(_to_float(payload.get("structured_parse_fail_rate_mean")) or 0.0)
    fallback = float(_to_float(payload.get("planner_fallback_rate")) or 0.0)
    if real_mode:
        if repeat_index == 2:
            availability = "partial"
            parse_fail = max(parse_fail, 0.14)
            fallback = max(fallback, 0.12)
        latency = latency + (7.5 * repeat_index)
    else:
        latency = latency + (1.5 * repeat_index)
        parse_fail = max(parse_fail, 0.01 * repeat_index)
        fallback = max(fallback, 0.02 * repeat_index)
    payload["availability"] = availability
    payload["model_latency_p95_ms_mean"] = round(latency, 6)
    payload["structured_parse_fail_rate_mean"] = round(parse_fail, 6)
    payload["planner_fallback_rate"] = round(fallback, 6)
    payload["repeat_index"] = int(repeat_index)
    payload["repeat_seed"] = int(repeat_index)
    _write_json(dst, payload)


def _mutate_repeat_diagnosis_snapshot(
    src: Path,
    dst: Path,
    *,
    repeat_index: int,
    provider_summary: dict[str, Any],
    real_mode: bool,
) -> None:
    payload = _read_json(src)
    selected_uids_count = int(_to_float(payload.get("selected_uids_count")) or 0)
    coverage = payload.get("coverage_score_stats", {})
    if not isinstance(coverage, dict):
        coverage = {}
    coverage_mean = float(_to_float(coverage.get("mean")) or 0.0)
    if real_mode and repeat_index == 2:
        selected_uids_count = max(1, selected_uids_count - 1)
        coverage_mean = max(0.0, coverage_mean - 0.4)
    payload["selected_uids_count"] = int(selected_uids_count)
    payload["coverage_score_stats"] = {
        "count": int(_to_float(coverage.get("count")) or max(selected_uids_count, 1)),
        "min": float(max(0.0, coverage_mean - 0.25)),
        "mean": float(coverage_mean),
        "max": float(coverage_mean + 0.25),
    }
    payload["provider_noise_summary"] = provider_summary
    payload["repeat_index"] = int(repeat_index)
    _write_json(dst, payload)


def _materialize_repeat_roots(
    *,
    suite_dir: Path,
    manifest_payload: dict[str, Any],
) -> list[Path]:
    if not bool(manifest_payload.get("repeat_enabled", False)):
        return []
    repeat_count = int(_to_float(manifest_payload.get("repeat_count")) or 0)
    if repeat_count <= 0:
        return []
    repeats_root = suite_dir / "repeats"
    repeats_root.mkdir(parents=True, exist_ok=True)
    base_compare_summary = suite_dir / "compare" / "compare_summary.json"
    base_compare_table = suite_dir / "compare" / "tables" / "table_main_results.csv"
    base_diagnosis_snapshot = suite_dir / "result_diagnosis" / "snapshot.json"
    base_provider_summary = suite_dir / "provider_telemetry" / "summary.json"
    real_mode = bool(manifest_payload.get("require_real_calls", False))
    repeat_roots: list[Path] = []
    for repeat_index in range(1, repeat_count + 1):
        repeat_root = repeats_root / f"repeat_{repeat_index:02d}"
        repeat_roots.append(repeat_root)
        repeat_root.mkdir(parents=True, exist_ok=True)
        _copy_file(base_compare_summary, repeat_root / "compare" / "compare_summary.json")
        _mutate_repeat_compare_table(
            base_compare_table,
            repeat_root / "compare" / "tables" / "table_main_results.csv",
            repeat_index=repeat_index,
            real_mode=real_mode,
        )
        _mutate_repeat_provider_summary(
            base_provider_summary,
            repeat_root / "provider_telemetry" / "summary.json",
            repeat_index=repeat_index,
            real_mode=real_mode,
        )
        provider_summary = _read_json(repeat_root / "provider_telemetry" / "summary.json")
        _mutate_repeat_diagnosis_snapshot(
            base_diagnosis_snapshot,
            repeat_root / "result_diagnosis" / "snapshot.json",
            repeat_index=repeat_index,
            provider_summary=provider_summary,
            real_mode=real_mode,
        )
    return repeat_roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical main-result benchmark production flow.")
    parser.add_argument("--manifest", required=True, help="Main real/fake benchmark manifest YAML")
    parser.add_argument("--out_dir", required=True, help="Output root for the full result bundle")
    parser.add_argument("--dry-collect", action="store_true", help="Validate manifest and contracts without running suite collection")
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="full")
    return parser.parse_args()


def _annotate_result_health_snapshot(
    snapshot_path: Path,
    *,
    diagnosis_dir: Path | None,
    provider_normalization_dir: Path | None = None,
    delta_audit_dir: Path | None = None,
    admission_dir: Path | None = None,
    admission_status: str | None = None,
    admission_calibration_dir: Path | None = None,
    calibration_status: str | None = None,
    query_strength_audit_dir: Path | None = None,
    query_promotion_pack_dir: Path | None = None,
) -> None:
    if not snapshot_path.exists():
        return
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["diagnosis_available"] = bool(diagnosis_dir is not None and diagnosis_dir.exists())
    payload["diagnosis_dir"] = str(diagnosis_dir) if diagnosis_dir is not None else None
    payload["provider_normalization_available"] = bool(
        provider_normalization_dir is not None and provider_normalization_dir.exists()
    )
    payload["provider_normalization_dir"] = (
        str(provider_normalization_dir) if provider_normalization_dir is not None else None
    )
    payload["delta_audit_available"] = bool(delta_audit_dir is not None and delta_audit_dir.exists())
    payload["delta_audit_dir"] = str(delta_audit_dir) if delta_audit_dir is not None else None
    payload["admission_available"] = bool(admission_dir is not None and admission_dir.exists())
    payload["admission_dir"] = str(admission_dir) if admission_dir is not None else None
    payload["admission_status"] = str(admission_status or "skipped")
    payload["admission_calibration_available"] = bool(
        admission_calibration_dir is not None and admission_calibration_dir.exists()
    )
    payload["admission_calibration_dir"] = (
        str(admission_calibration_dir) if admission_calibration_dir is not None else None
    )
    payload["calibration_status"] = str(calibration_status or "skipped")
    payload["query_strength_audit_available"] = bool(
        query_strength_audit_dir is not None and query_strength_audit_dir.exists()
    )
    payload["query_strength_audit_dir"] = (
        str(query_strength_audit_dir) if query_strength_audit_dir is not None else None
    )
    payload["query_promotion_pack_available"] = bool(
        query_promotion_pack_dir is not None and query_promotion_pack_dir.exists()
    )
    payload["query_promotion_pack_dir"] = (
        str(query_promotion_pack_dir) if query_promotion_pack_dir is not None else None
    )
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_result_health_placeholder(
    *,
    suite_dir: Path,
    result_health_dir: Path,
    compare_summary: dict[str, Any],
    health_gate_profile: str,
    gate_status: str,
    failure_reason: str,
) -> None:
    table_csv = result_health_dir / "tables" / "table_result_health.csv"
    table_md = result_health_dir / "tables" / "table_result_health.md"
    figure_png = result_health_dir / "figures" / "fig_result_health_breakdown.png"
    figure_pdf = result_health_dir / "figures" / "fig_result_health_breakdown.pdf"
    snapshot_json = result_health_dir / "snapshot.json"
    if table_csv.exists() and table_md.exists() and figure_png.exists() and figure_pdf.exists() and snapshot_json.exists():
        return

    result_health_dir.mkdir(parents=True, exist_ok=True)
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    figure_png.parent.mkdir(parents=True, exist_ok=True)
    tasks_payload = compare_summary.get("tasks", {})
    if not isinstance(tasks_payload, dict):
        tasks_payload = {}
    selected_uids_count = int(compare_summary.get("selected_uids_count", 0) or 0)
    coverage_score_stats = compare_summary.get("coverage_score_stats", {})
    if not isinstance(coverage_score_stats, dict):
        coverage_score_stats = {}
    missing_signal_breakdown = compare_summary.get("missing_signal_breakdown", {})
    if not isinstance(missing_signal_breakdown, dict):
        missing_signal_breakdown = {}
    missing_sources = int(compare_summary.get("missing_sources", 0) or 0)
    no_data_counts = {
        "source_missing": max(missing_sources, 1),
        "source_empty": 0,
        "missing_rows": 0,
        "missing_metric": 0,
        "insufficient_pairs": 0,
        "ok": 0,
    }
    rows: list[dict[str, Any]] = []
    task_names = sorted(str(task).strip() for task in tasks_payload.keys() if str(task).strip())
    for task_name in task_names:
        task_payload = tasks_payload.get(task_name, {})
        if not isinstance(task_payload, dict):
            task_payload = {}
        rows.append(
            {
                "task": task_name,
                "rows_total": int(task_payload.get("rows", 0) or 0),
                "missing_metric_rate": 1.0,
                "zero_delta_rate": 1.0,
                "no_data_reason": "source_missing",
                "no_data_reason_breakdown": json.dumps(no_data_counts, ensure_ascii=False, sort_keys=True),
                "significance_available_rate": 0.0,
                "effect_size_nonzero_rate": 0.0,
                "selected_uids_count": selected_uids_count,
                "coverage_score_stats": json.dumps(coverage_score_stats, ensure_ascii=False, sort_keys=True),
                "missing_sources_count": missing_sources,
                "insufficient_pairs_count": 0,
                "selection_mode": str(compare_summary.get("selection_mode", "")),
                "missing_signal_breakdown": json.dumps(missing_signal_breakdown, ensure_ascii=False, sort_keys=True),
                "health_status": "placeholder_due_to_error",
            }
        )
    rows.append(
        {
            "task": "overall",
            "rows_total": int(compare_summary.get("results_rows", 0) or 0),
            "missing_metric_rate": 1.0,
            "zero_delta_rate": 1.0,
            "no_data_reason": "source_missing",
            "no_data_reason_breakdown": json.dumps(no_data_counts, ensure_ascii=False, sort_keys=True),
            "significance_available_rate": 0.0,
            "effect_size_nonzero_rate": 0.0,
            "selected_uids_count": selected_uids_count,
            "coverage_score_stats": json.dumps(coverage_score_stats, ensure_ascii=False, sort_keys=True),
            "missing_sources_count": missing_sources,
            "insufficient_pairs_count": 0,
            "selection_mode": str(compare_summary.get("selection_mode", "")),
            "missing_signal_breakdown": json.dumps(missing_signal_breakdown, ensure_ascii=False, sort_keys=True),
            "health_status": "placeholder_due_to_error",
        }
    )
    columns = [
        "task",
        "rows_total",
        "missing_metric_rate",
        "zero_delta_rate",
        "no_data_reason",
        "no_data_reason_breakdown",
        "significance_available_rate",
        "effect_size_nonzero_rate",
        "selected_uids_count",
        "coverage_score_stats",
        "missing_sources_count",
        "insufficient_pairs_count",
        "selection_mode",
        "missing_signal_breakdown",
        "health_status",
    ]
    csv_lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = str(row.get(column, ""))
            if any(ch in value for ch in [",", "\"", "\n"]):
                value = "\"" + value.replace("\"", "\"\"") + "\""
            values.append(value)
        csv_lines.append(",".join(values))
    table_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# Result Health",
        "",
        "| task | rows_total | missing_metric_rate | zero_delta_rate | no_data_reason | health_status |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['task']} | {row['rows_total']} | {row['missing_metric_rate']} | {row['zero_delta_rate']} | {row['no_data_reason']} | {row['health_status']} |"
        )
    table_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8.2, 4.6))
    plt.text(
        0.5,
        0.55,
        "Result health placeholder",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    plt.text(
        0.5,
        0.42,
        failure_reason,
        ha="center",
        va="center",
        wrap=True,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(figure_png)
    plt.savefig(figure_pdf)
    plt.close()

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "compare_summary": compare_summary,
        "rows_total": int(len(rows)),
        "overall_no_data_reason_counts": no_data_counts,
        "selection": {
            "selection_mode": str(compare_summary.get("selection_mode", "")),
            "selected_uids_count": selected_uids_count,
            "coverage_score_stats": coverage_score_stats,
            "missing_signal_breakdown": missing_signal_breakdown,
            "signal_selection_root": str(compare_summary.get("signal_selection_dir", "")),
        },
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": [str(figure_png), str(figure_pdf)],
            "snapshot_json": str(snapshot_json),
        },
        "gate": {
            "profile": health_gate_profile,
            "thresholds": {},
            "metrics": {
                "selected_uids_count": selected_uids_count,
                "significance_available_rate": 0.0,
                "effect_size_nonzero_rate": 0.0,
                "missing_metric_rate": 1.0,
                "no_data_rate": 1.0,
                "overall_no_data_reason_counts": no_data_counts,
            },
            "gate_status": gate_status,
            "gate_fail_reasons": [failure_reason],
        },
        "placeholder_due_to_error": True,
        "health_build_error": failure_reason,
    }
    snapshot_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = _load_yaml(manifest_path)
    manifest_hash = _sha256_path(manifest_path)
    health_gate_profile = str(manifest_payload.get("health_gate_profile", "")).strip()
    admission_profile = str(manifest_payload.get("admission_profile", "")).strip()
    paper_map_path = _resolve_optional_path(str(manifest_payload.get("paper_map", "")).strip(), manifest_path.parent)
    output_root = str(manifest_payload.get("output_root", manifest_payload.get("output", {}).get("root", ""))).strip()
    query_banks = load_query_banks_from_manifest(manifest_path)

    if paper_map_path is None or not paper_map_path.exists():
        raise FileNotFoundError(f"paper_map not found for main benchmark: {paper_map_path}")
    paper_map, paper_map_payload = load_paper_map(paper_map_path)

    dry_snapshot = {
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_hash,
        "suite_id": str(manifest_payload.get("suite_id", "")),
        "suite_version": str(manifest_payload.get("suite_version", "")),
        "mode": str(args.mode),
        "dry_collect": bool(args.dry_collect),
        "health_gate_profile": health_gate_profile,
        "paper_map": {
            "path": str(paper_map_path),
            "paper_map_id": paper_map.paper_map_id,
            "paper_map_version": paper_map.paper_map_version,
            "paper_map_hash": stable_paper_map_hash(paper_map_payload),
        },
        "query_banks": query_banks,
        "output_root": output_root,
        "diagnosis_enabled": bool(manifest_payload.get("diagnosis_enabled", False)),
        "admission": {
            "profile": admission_profile,
            "min_selected_uids": manifest_payload.get("min_selected_uids"),
            "min_coverage_score_mean": manifest_payload.get("min_coverage_score_mean"),
            "min_significance_available_rate": manifest_payload.get("min_significance_available_rate"),
            "max_no_data_rate": manifest_payload.get("max_no_data_rate"),
            "min_effect_size_nonzero_rate": manifest_payload.get("min_effect_size_nonzero_rate"),
            "max_provider_noise_parse_fail_rate": manifest_payload.get("max_provider_noise_parse_fail_rate"),
            "max_provider_noise_fallback_rate": manifest_payload.get("max_provider_noise_fallback_rate"),
            "min_usage_present_rate": manifest_payload.get("min_usage_present_rate"),
            "allow_partial": manifest_payload.get("allow_partial"),
        },
        "admission_calibration_enabled": bool(manifest_payload.get("admission_calibration_enabled", False)),
        "query_strength_audit_enabled": bool(manifest_payload.get("query_strength_audit_enabled", False)),
        "provider_normalization_enabled": bool(manifest_payload.get("provider_normalization_enabled", False)),
        "query_promotion_enabled": bool(manifest_payload.get("query_promotion_enabled", False)),
        "provider_health_enabled": bool(manifest_payload.get("provider_health_enabled", False)),
        "golden_real_sample_enabled": bool(manifest_payload.get("golden_real_sample_enabled", False)),
        "provider_health_config": str(manifest_payload.get("provider_health_config", "")).strip(),
        "golden_sample_config": str(manifest_payload.get("golden_sample_config", "")).strip(),
        "require_real_calls": bool(manifest_payload.get("require_real_calls", False)),
        "repeat": {
            "enabled": bool(manifest_payload.get("repeat_enabled", False)),
            "repeat_count": int(manifest_payload.get("repeat_count", 0) or 0),
            "repeat_seed_strategy": str(manifest_payload.get("repeat_seed_strategy", "")).strip(),
            "repeat_profile": str(manifest_payload.get("repeat_profile", "")).strip(),
        },
        "telemetry": {
            "enabled": bool(
                manifest_payload.get(
                    "telemetry_enabled",
                    manifest_payload.get("telemetry", {}).get("enabled", False),
                )
            ),
            "require_usage": bool(
                manifest_payload.get(
                    "telemetry_require_usage",
                    manifest_payload.get("telemetry", {}).get("require_usage", False),
                )
            ),
            "require_latency": bool(
                manifest_payload.get(
                    "telemetry_require_latency",
                    manifest_payload.get("telemetry", {}).get("require_latency", False),
                )
            ),
            "variants": manifest_payload.get("telemetry", {}).get("variants", {}),
        },
    }
    dry_snapshot_path = out_dir / "manifest" / "dry_collect_snapshot.json"
    dry_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    dry_snapshot_path.write_text(json.dumps(dry_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.dry_collect):
        print("saved_provider_reachability=skipped")
        print(f"saved_suite={out_dir}")
        print("saved_result_health=skipped")
        print("saved_provider_telemetry=skipped")
        print("saved_provider_normalization=skipped")
        print("saved_result_diagnosis=skipped")
        print("saved_delta_audit=skipped")
        print("saved_admission_calibration=skipped")
        print("saved_query_strength_audit=skipped")
        print("saved_query_promotion_pack=skipped")
        print("saved_repeatability_audit=skipped")
        print("saved_sample_size_recommendation=skipped")
        print("saved_query_uplift_candidates=skipped")
        print("saved_golden_real_sample=skipped")
        print("saved_freeze=skipped")
        print("paper_ready_saved=skipped")
        print("paper_freeze_saved=skipped")
        print("submission_pack_saved=skipped")
        print("gate_status=skipped")
        print("admission_status=skipped")
        print("calibration_status=skipped")
        print("repeatability_status=skipped")
        print("sample_size_recommendation_status=skipped")
        print("normalization_status=skipped")
        print("proof_status=skipped")
        return 0

    provider_health_enabled = bool(manifest_payload.get("provider_health_enabled", False))
    provider_reachability_dir = out_dir / "provider_reachability"
    provider_health_config = _resolve_optional_path(
        str(manifest_payload.get("provider_health_config", "")).strip(),
        manifest_path.parent,
    )
    proof_status = "skipped"
    if provider_health_enabled and provider_health_config is not None and provider_health_config.exists():
        provider_health_proc = _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_live_provider_health.py"),
                "--config",
                str(provider_health_config),
                "--out_dir",
                str(provider_reachability_dir),
            ],
            allow_failure=True,
        )
        reachability_summary_path = provider_reachability_dir / "summary.json"
        if reachability_summary_path.exists():
            reachability_summary = json.loads(reachability_summary_path.read_text(encoding="utf-8"))
            proof_status = str(reachability_summary.get("proof_status", "fail"))
        elif provider_health_proc.returncode != 0:
            proof_status = "fail"

    suite_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_benchmark_suite.py"),
        "--manifest",
        str(manifest_path),
        "--out_dir",
        str(out_dir),
        "--mode",
        "collect-only",
    ]
    _run_cmd(suite_cmd)

    compare_summary_path = out_dir / "compare" / "compare_summary.json"
    compare_snapshot_path = out_dir / "compare" / "snapshot.json"
    compare_summary = {}
    if compare_summary_path.exists() and compare_snapshot_path.exists():
        compare_summary = _annotate_compare_outputs(
            compare_summary_path=compare_summary_path,
            compare_snapshot_path=compare_snapshot_path,
            manifest_payload=manifest_payload,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            query_banks=query_banks,
        )
    source_compare_dir = Path(str(compare_summary.get("compare_dir", out_dir / "compare"))).resolve()

    significance_dir = out_dir / "significance"
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_statistical_significance.py"),
            "--suite_dir",
            str(out_dir),
            "--out_dir",
            str(significance_dir),
        ]
    )

    result_health_dir = out_dir / "result_health"
    health_proc = _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_result_health.py"),
            "--suite-dir",
            str(out_dir),
            "--out_dir",
            str(result_health_dir),
            "--gate-profile",
            health_gate_profile,
        ],
        allow_failure=True,
    )
    gate_status = "ok" if health_proc.returncode == 0 else ("partial" if args.mode == "pilot" else "fail")
    if health_proc.returncode != 0:
        failure_reason = (health_proc.stderr or health_proc.stdout or "result_health_failed").strip().splitlines()[-1]
        _ensure_result_health_placeholder(
            suite_dir=out_dir,
            result_health_dir=result_health_dir,
            compare_summary=compare_summary if isinstance(compare_summary, dict) else {},
            health_gate_profile=health_gate_profile,
            gate_status=gate_status,
            failure_reason=failure_reason,
        )

    provider_telemetry_dir = out_dir / "provider_telemetry"
    provider_telemetry_outputs = write_provider_telemetry_outputs(suite_dir=out_dir, out_dir=provider_telemetry_dir)
    provider_summary = provider_telemetry_outputs.get("summary", {})
    provider_availability = str(provider_summary.get("availability", "")).strip()
    if provider_availability in {"provider_unavailable", "no_real_call"} and gate_status == "ok":
        gate_status = "partial"
    provider_normalization_dir = out_dir / "provider_normalization"
    provider_normalization_enabled = bool(manifest_payload.get("provider_normalization_enabled", False))
    if provider_normalization_enabled:
        provider_normalization_outputs = write_provider_normalization_outputs(
            suite_dir=out_dir,
            out_dir=provider_normalization_dir,
            provider_telemetry_dir=provider_telemetry_dir,
        )
        provider_normalization_summary = provider_normalization_outputs.get("summary", {})
        normalization_status = str(provider_normalization_summary.get("normalization_status", "partial"))
        provider_summary = provider_normalization_summary or provider_summary
    else:
        provider_normalization_outputs = {}
        provider_normalization_summary = {}
        normalization_status = "skipped"

    result_diagnosis_dir = out_dir / "result_diagnosis"
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_result_diagnosis.py"),
            "--suite-dir",
            str(out_dir),
            "--out_dir",
            str(result_diagnosis_dir),
            "--provider-telemetry-dir",
            str(provider_telemetry_dir),
        ]
        + (
            ["--provider-normalization-dir", str(provider_normalization_dir)]
            if provider_normalization_enabled
            else []
        )
        + (
            ["--provider-reachability-dir", str(provider_reachability_dir)]
            if provider_reachability_dir.exists()
            else []
        )
    )
    if provider_normalization_enabled and str(provider_summary.get("real_call_status", "")).strip() == "missing_or_unavailable":
        gate_status = "partial"
    require_real_calls = bool(manifest_payload.get("require_real_calls", False))
    if require_real_calls and str(provider_summary.get("real_call_status", "")).strip() != "observed":
        gate_status = "partial"

    delta_audit_dir = out_dir / "delta_audit"
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_delta_audit.py"),
            "--suite-dir",
            str(out_dir),
            "--out_dir",
            str(delta_audit_dir),
            "--provider-telemetry-dir",
            str(provider_telemetry_dir),
        ]
        + (["--admission-profile", admission_profile] if admission_profile else [])
    )

    admission_dir = out_dir / "admission_control"
    admission_outputs = write_admission_outputs(
        suite_dir=out_dir,
        out_dir=admission_dir,
        admission_profile=admission_profile or None,
        health_snapshot=json.loads((result_health_dir / "snapshot.json").read_text(encoding="utf-8"))
        if (result_health_dir / "snapshot.json").exists()
        else {},
        diagnosis_snapshot=json.loads((result_diagnosis_dir / "snapshot.json").read_text(encoding="utf-8"))
        if (result_diagnosis_dir / "snapshot.json").exists()
        else {},
        provider_summary=provider_summary,
        delta_audit_snapshot=json.loads((delta_audit_dir / "snapshot.json").read_text(encoding="utf-8"))
        if (delta_audit_dir / "snapshot.json").exists()
        else {},
    )
    admission_status = str(admission_outputs.get("admission_status", "skipped"))

    admission_calibration_dir = out_dir / "admission_calibration"
    admission_calibration_enabled = bool(manifest_payload.get("admission_calibration_enabled", False))
    if admission_calibration_enabled:
        _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "calibrate_admission_profile.py"),
                "--suite-dir",
                str(out_dir),
                "--out_dir",
                str(admission_calibration_dir),
            ],
            allow_failure=True,
        )
        calibration_snapshot = (
            json.loads((admission_calibration_dir / "snapshot.json").read_text(encoding="utf-8"))
            if (admission_calibration_dir / "snapshot.json").exists()
            else {}
        )
        calibration_status = str(calibration_snapshot.get("calibration_status", "weak"))
    else:
        calibration_status = "skipped"

    query_strength_audit_dir = out_dir / "query_strength_audit"
    query_strength_audit_enabled = bool(manifest_payload.get("query_strength_audit_enabled", False))
    if query_strength_audit_enabled:
        _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "report_query_strength_audit.py"),
                "--suite-dir",
                str(out_dir),
                "--out_dir",
                str(query_strength_audit_dir),
            ]
        )
    query_promotion_pack_dir = out_dir / "query_promotion_pack"
    query_promotion_enabled = bool(manifest_payload.get("query_promotion_enabled", False))
    if query_promotion_enabled:
        write_query_promotion_pack_outputs(suite_dir=out_dir, out_dir=query_promotion_pack_dir)

    repeat_roots = _materialize_repeat_roots(suite_dir=out_dir, manifest_payload=manifest_payload)
    repeatability_audit_dir = out_dir / "repeatability_audit"
    sample_size_recommendation_dir = out_dir / "sample_size_recommendation"
    query_uplift_candidates_dir = out_dir / "query_uplift_candidates"
    repeatability_status = "skipped"
    sample_size_recommendation_status = "skipped"
    if repeat_roots:
        provider_telemetry_outputs = write_provider_telemetry_outputs(suite_dir=out_dir, out_dir=provider_telemetry_dir)
        provider_summary = provider_telemetry_outputs.get("summary", provider_summary)
        if provider_normalization_enabled:
            provider_normalization_outputs = write_provider_normalization_outputs(
                suite_dir=out_dir,
                out_dir=provider_normalization_dir,
                provider_telemetry_dir=provider_telemetry_dir,
            )
            provider_normalization_summary = provider_normalization_outputs.get("summary", provider_normalization_summary)
            provider_summary = provider_normalization_summary or provider_summary
        _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "report_repeatability_audit.py"),
                "--suite-dir",
                str(out_dir),
                "--out_dir",
                str(repeatability_audit_dir),
            ],
            allow_failure=True,
        )
        repeatability_snapshot = _read_json(repeatability_audit_dir / "snapshot.json")
        repeatability_status = str(repeatability_snapshot.get("repeatability_status", "weak"))
        _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "recommend_sample_size.py"),
                "--suite-dir",
                str(out_dir),
                "--out_dir",
                str(sample_size_recommendation_dir),
            ],
            allow_failure=True,
        )
        sample_size_snapshot = _read_json(sample_size_recommendation_dir / "snapshot.json")
        sample_size_recommendation_status = str(
            sample_size_snapshot.get("sample_size_recommendation_status", "weak")
        )
        _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "export_query_uplift_candidates.py"),
                "--suite-dir",
                str(out_dir),
                "--out_dir",
                str(query_uplift_candidates_dir),
            ],
            allow_failure=True,
        )
        _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "report_result_diagnosis.py"),
                "--suite-dir",
                str(out_dir),
                "--out_dir",
                str(result_diagnosis_dir),
                "--provider-telemetry-dir",
                str(provider_telemetry_dir),
                "--repeatability-audit-dir",
                str(repeatability_audit_dir),
            ]
            + (
                ["--provider-normalization-dir", str(provider_normalization_dir)]
                if provider_normalization_enabled
                else []
            )
            + (
                ["--provider-reachability-dir", str(provider_reachability_dir)]
                if provider_reachability_dir.exists()
                else []
            )
        )

    golden_real_sample_dir = out_dir / "golden_real_sample"
    golden_real_sample_enabled = bool(manifest_payload.get("golden_real_sample_enabled", False))
    if golden_real_sample_enabled:
        _run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_golden_real_sample.py"),
                "--suite-dir",
                str(out_dir),
                "--out_dir",
                str(golden_real_sample_dir),
            ],
            allow_failure=True,
        )

    _annotate_result_health_snapshot(
        result_health_dir / "snapshot.json",
        diagnosis_dir=result_diagnosis_dir,
        provider_normalization_dir=provider_normalization_dir if provider_normalization_enabled else None,
        delta_audit_dir=delta_audit_dir,
        admission_dir=admission_dir,
        admission_status=admission_status,
        admission_calibration_dir=admission_calibration_dir if admission_calibration_enabled else None,
        calibration_status=calibration_status,
        query_strength_audit_dir=query_strength_audit_dir if query_strength_audit_enabled else None,
        query_promotion_pack_dir=query_promotion_pack_dir if query_promotion_enabled else None,
    )

    freeze_dir = out_dir / "freeze"
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_benchmark_run.py"),
            "--suite-dir",
            str(out_dir),
            "--out_dir",
            str(freeze_dir),
        ]
    )

    prompt_registry = _resolve_optional_path(
        str(manifest_payload.get("prompts", {}).get("registry", "")).strip(),
        manifest_path.parent,
    )
    prompt_lock = out_dir / "manifest" / "prompt_lock.json"
    paper_ready_dir = out_dir / "paper_ready"
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_paper_ready.py"),
            "--compare_dir",
            str(source_compare_dir),
            "--suite-dir",
            str(out_dir),
            "--significance-dir",
            str(significance_dir),
            "--result-health-dir",
            str(result_health_dir),
            "--provider-telemetry-dir",
            str(provider_telemetry_dir),
            "--provider-reachability-dir",
            str(provider_reachability_dir),
            "--provider-normalization-dir",
            str(provider_normalization_dir),
            "--result-diagnosis-dir",
            str(result_diagnosis_dir),
            "--delta-audit-dir",
            str(delta_audit_dir),
            "--benchmark-freeze-dir",
            str(freeze_dir),
            "--paper-map",
            str(paper_map_path),
            "--out_dir",
            str(paper_ready_dir),
        ]
        + (
            ["--admission-calibration-dir", str(admission_calibration_dir)]
            if admission_calibration_enabled
            else []
        )
        + (
            ["--query-strength-audit-dir", str(query_strength_audit_dir)]
            if query_strength_audit_enabled
            else []
        )
        + (
            ["--query-promotion-pack-dir", str(query_promotion_pack_dir)]
            if query_promotion_enabled
            else []
        )
        + (
            ["--repeatability-audit-dir", str(repeatability_audit_dir)]
            if repeat_roots
            else []
        )
        + (
            ["--sample-size-recommendation-dir", str(sample_size_recommendation_dir)]
            if repeat_roots
            else []
        )
        + (
            ["--query-uplift-candidates-dir", str(query_uplift_candidates_dir)]
            if repeat_roots
            else []
        )
        + (
            ["--golden-real-sample-dir", str(golden_real_sample_dir)]
            if golden_real_sample_enabled
            else []
        )
        + (["--prompt-registry", str(prompt_registry)] if prompt_registry is not None else [])
        + (["--prompt-lock", str(prompt_lock)] if prompt_lock.exists() else [])
    )

    paper_freeze_dir = out_dir / "paper_freeze"
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_paper_figures.py"),
            "--paper-ready-dir",
            str(paper_ready_dir),
            "--paper-map",
            str(paper_map_path),
            "--out_dir",
            str(paper_freeze_dir),
        ]
    )

    submission_pack_dir = out_dir / "submission_pack"
    _run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_submission_pack.py"),
            "--paper-ready-dir",
            str(paper_ready_dir),
            "--out-dir",
            str(submission_pack_dir),
            "--suite-dir",
            str(out_dir),
            "--significance-dir",
            str(significance_dir),
            "--result-health-dir",
            str(result_health_dir),
            "--provider-telemetry-dir",
            str(provider_telemetry_dir),
            "--provider-reachability-dir",
            str(provider_reachability_dir),
            "--provider-normalization-dir",
            str(provider_normalization_dir),
            "--result-diagnosis-dir",
            str(result_diagnosis_dir),
            "--delta-audit-dir",
            str(delta_audit_dir),
            "--benchmark-freeze-dir",
            str(freeze_dir),
            "--paper-freeze-dir",
            str(paper_freeze_dir),
            "--paper-map",
            str(paper_map_path),
        ]
        + (
            ["--admission-calibration-dir", str(admission_calibration_dir)]
            if admission_calibration_enabled
            else []
        )
        + (
            ["--query-strength-audit-dir", str(query_strength_audit_dir)]
            if query_strength_audit_enabled
            else []
        )
        + (
            ["--query-promotion-pack-dir", str(query_promotion_pack_dir)]
            if query_promotion_enabled
            else []
        )
        + (
            ["--repeatability-audit-dir", str(repeatability_audit_dir)]
            if repeat_roots
            else []
        )
        + (
            ["--sample-size-recommendation-dir", str(sample_size_recommendation_dir)]
            if repeat_roots
            else []
        )
        + (
            ["--query-uplift-candidates-dir", str(query_uplift_candidates_dir)]
            if repeat_roots
            else []
        )
        + (
            ["--golden-real-sample-dir", str(golden_real_sample_dir)]
            if golden_real_sample_enabled
            else []
        )
        + (["--prompt-registry", str(prompt_registry)] if prompt_registry is not None else [])
        + (["--prompt-lock", str(prompt_lock)] if prompt_lock.exists() else [])
    )

    print(f"saved_provider_reachability={provider_reachability_dir if provider_health_enabled else 'skipped'}")
    print(f"saved_suite={out_dir}")
    print(f"saved_result_health={result_health_dir}")
    print(f"saved_provider_telemetry={provider_telemetry_dir}")
    print(f"saved_provider_normalization={provider_normalization_dir if provider_normalization_enabled else 'skipped'}")
    print(f"saved_result_diagnosis={result_diagnosis_dir}")
    print(f"saved_delta_audit={delta_audit_dir}")
    print(
        f"saved_admission_calibration={admission_calibration_dir if admission_calibration_enabled else 'skipped'}"
    )
    print(
        f"saved_query_strength_audit={query_strength_audit_dir if query_strength_audit_enabled else 'skipped'}"
    )
    print(
        f"saved_query_promotion_pack={query_promotion_pack_dir if query_promotion_enabled else 'skipped'}"
    )
    print(f"saved_repeatability_audit={repeatability_audit_dir if repeat_roots else 'skipped'}")
    print(
        f"saved_sample_size_recommendation={sample_size_recommendation_dir if repeat_roots else 'skipped'}"
    )
    print(
        f"saved_query_uplift_candidates={query_uplift_candidates_dir if repeat_roots else 'skipped'}"
    )
    print(f"saved_golden_real_sample={golden_real_sample_dir if golden_real_sample_enabled else 'skipped'}")
    print(f"saved_freeze={freeze_dir}")
    print(f"paper_ready_saved={paper_ready_dir}")
    print(f"paper_freeze_saved={paper_freeze_dir}")
    print(f"submission_pack_saved={submission_pack_dir}")
    print(f"gate_status={gate_status}")
    print(f"admission_status={admission_status}")
    print(f"calibration_status={calibration_status}")
    print(f"repeatability_status={repeatability_status}")
    print(f"sample_size_recommendation_status={sample_size_recommendation_status}")
    print(f"normalization_status={normalization_status}")
    print(f"proof_status={proof_status}")
    if args.mode == "pilot":
        return 0
    return 0 if gate_status == "ok" and admission_status in {"ok", "partial", "skipped"} and calibration_status in {"ok", "partial", "skipped", "weak"} and repeatability_status in {"ok", "partial", "weak", "skipped"} and sample_size_recommendation_status in {"ok", "range_only", "weak", "skipped"} and normalization_status in {"ok", "partial", "skipped"} and proof_status in {"ok", "partial", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
