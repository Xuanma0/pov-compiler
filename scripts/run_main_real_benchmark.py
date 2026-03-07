from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.admission_control import write_admission_outputs
from pov_compiler.bench.query_bank import load_query_banks_from_manifest
from pov_compiler.bench.reporting.paper_map import load_paper_map, stable_paper_map_hash
from pov_compiler.bench.reporting.provider_telemetry import write_provider_telemetry_outputs


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
    delta_audit_dir: Path | None = None,
    admission_dir: Path | None = None,
    admission_status: str | None = None,
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
    payload["delta_audit_available"] = bool(delta_audit_dir is not None and delta_audit_dir.exists())
    payload["delta_audit_dir"] = str(delta_audit_dir) if delta_audit_dir is not None else None
    payload["admission_available"] = bool(admission_dir is not None and admission_dir.exists())
    payload["admission_dir"] = str(admission_dir) if admission_dir is not None else None
    payload["admission_status"] = str(admission_status or "skipped")
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        print(f"saved_suite={out_dir}")
        print("saved_result_health=skipped")
        print("saved_provider_telemetry=skipped")
        print("saved_result_diagnosis=skipped")
        print("saved_delta_audit=skipped")
        print("saved_freeze=skipped")
        print("paper_ready_saved=skipped")
        print("paper_freeze_saved=skipped")
        print("submission_pack_saved=skipped")
        print("gate_status=skipped")
        print("admission_status=skipped")
        return 0

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
    compare_summary = {}
    if compare_summary_path.exists():
        compare_summary = json.loads(compare_summary_path.read_text(encoding="utf-8"))
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

    provider_telemetry_dir = out_dir / "provider_telemetry"
    provider_telemetry_outputs = write_provider_telemetry_outputs(suite_dir=out_dir, out_dir=provider_telemetry_dir)
    provider_summary = provider_telemetry_outputs.get("summary", {})
    provider_availability = str(provider_summary.get("availability", "")).strip()
    if provider_availability in {"provider_unavailable", "no_real_call"} and gate_status == "ok":
        gate_status = "partial"

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
    )
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
    _annotate_result_health_snapshot(
        result_health_dir / "snapshot.json",
        diagnosis_dir=result_diagnosis_dir,
        delta_audit_dir=delta_audit_dir,
        admission_dir=admission_dir,
        admission_status=admission_status,
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
        + (["--prompt-registry", str(prompt_registry)] if prompt_registry is not None else [])
        + (["--prompt-lock", str(prompt_lock)] if prompt_lock.exists() else [])
    )

    print(f"saved_suite={out_dir}")
    print(f"saved_result_health={result_health_dir}")
    print(f"saved_provider_telemetry={provider_telemetry_dir}")
    print(f"saved_result_diagnosis={result_diagnosis_dir}")
    print(f"saved_delta_audit={delta_audit_dir}")
    print(f"saved_freeze={freeze_dir}")
    print(f"paper_ready_saved={paper_ready_dir}")
    print(f"paper_freeze_saved={paper_freeze_dir}")
    print(f"submission_pack_saved={submission_pack_dir}")
    print(f"gate_status={gate_status}")
    print(f"admission_status={admission_status}")
    return 0 if gate_status == "ok" and admission_status in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
