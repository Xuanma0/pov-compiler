from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.query_bank import QueryBank
from pov_compiler.bench.query_bank import copy_query_bank_files, load_query_banks_from_manifest, write_query_bank_lock
from pov_compiler.bench.reporting.object_persistence_uplift import (
    compute_variant_object_persistence_metrics,
    write_object_persistence_compare_outputs,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for object-persistence manifests.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_path(raw_value: str | None, base_dir: Path) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("Expected a non-empty path.")
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (ROOT / path).resolve()


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _provider_probe_note() -> dict[str, Any]:
    required = ["PARATERA_API_KEY", "PARATERA_BASE_URL", "PARATERA_MODEL"]
    present = {key: bool(os.environ.get(key)) for key in required}
    if not all(present.values()):
        return {"checked": False, "status": "skipped_no_env", "env_present": present}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "model_health_check.py"),
        "--provider",
        "openai_compat",
        "--model",
        str(os.environ.get("PARATERA_MODEL", "")),
        "--base-url",
        str(os.environ.get("PARATERA_BASE_URL", "")),
        "--api-key-env",
        "PARATERA_API_KEY",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    return {
        "checked": True,
        "status": "ok" if proc.returncode == 0 else "failed",
        "env_present": present,
        "stdout_lines": [line for line in proc.stdout.splitlines() if "api_key" not in line.lower()],
    }


def _load_query_bank(manifest_payload: dict[str, Any], manifest_path: Path) -> QueryBank:
    query_bank_path = _resolve_path(str(manifest_payload.get("query_bank", "")).strip(), manifest_path.parent)
    return QueryBank.from_path(query_bank_path)


def _query_bank_lock_payload(
    *,
    manifest_payload: dict[str, Any],
    manifest_path: Path,
    query_bank: QueryBank,
) -> dict[str, Any]:
    query_bank_path = _resolve_path(str(manifest_payload.get("query_bank", "")).strip(), manifest_path.parent)
    lock_payload = load_query_banks_from_manifest(manifest_path)
    if lock_payload.get("primary"):
        return lock_payload
    return {
        "primary": query_bank.build_lock(query_bank_path, is_primary=True),
        "banks": [query_bank.build_lock(query_bank_path, is_primary=True)],
        "groups": [],
        "top_k": None,
    }


def _run_fixture_variant(
    *,
    variant_label: str,
    variant_cfg: dict[str, Any],
    query_bank: QueryBank,
    out_dir: Path,
) -> dict[str, Any]:
    metrics = compute_variant_object_persistence_metrics(
        variant_label=variant_label,
        output_payload={},
        query_bank=query_bank,
        source_mode="fixture",
        metrics_override={**dict(variant_cfg), **dict(variant_cfg.get("fixture_metrics", {}))},
    )
    _write_json(out_dir / "metrics.json", metrics)
    return metrics


def _run_offline_variant(
    *,
    manifest_payload: dict[str, Any],
    manifest_path: Path,
    variant_label: str,
    variant_cfg: dict[str, Any],
    query_bank: QueryBank,
    out_dir: Path,
) -> tuple[dict[str, Any], str]:
    base_config_path = _resolve_path(str(variant_cfg.get("config", "configs/default.yaml")), manifest_path.parent)
    video_path = _resolve_path(str(variant_cfg.get("video", manifest_payload.get("video", ""))), manifest_path.parent)
    base_config = _load_yaml(base_config_path)
    resolved_config = json.loads(json.dumps(base_config))
    resolved_config.setdefault("perception", {})
    resolved_config.setdefault("object_memory", {})
    resolved_config["sample_fps"] = float(variant_cfg.get("sample_fps", resolved_config.get("sample_fps", 4.0)))
    resolved_config["perception"]["enabled"] = True
    resolved_config["perception"]["backend"] = str(variant_cfg.get("perception_backend", "stub"))
    resolved_config["perception"]["sample_fps"] = float(
        variant_cfg.get("perception_fps", resolved_config["perception"].get("sample_fps", 4.0))
    )
    resolved_config["perception"]["max_frames"] = int(
        variant_cfg.get("perception_max_frames", resolved_config["perception"].get("max_frames", 24))
    )
    resolved_config["perception"]["cache_dir"] = str(out_dir / "perception_cache")
    resolved_config["perception"]["fallback_to_stub"] = bool(
        variant_cfg.get("fallback_to_stub", resolved_config["perception"].get("fallback_to_stub", True))
    )
    resolved_config["object_memory"]["enabled"] = True

    config_out = out_dir / "resolved_config.yaml"
    json_out = out_dir / "output.json"
    _write_yaml(config_out, resolved_config)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_offline.py"),
        "--video",
        str(video_path),
        "--out",
        str(json_out),
        "--config",
        str(config_out),
        "--run-perception",
        "--perception-backend",
        str(variant_cfg.get("perception_backend", "stub")),
        "--perception-fps",
        str(variant_cfg.get("perception_fps", resolved_config["perception"].get("sample_fps", 4.0))),
        "--perception-max-frames",
        str(variant_cfg.get("perception_max_frames", resolved_config["perception"].get("max_frames", 24))),
    ]
    if bool(resolved_config["perception"].get("fallback_to_stub", True)):
        cmd.append("--perception-fallback-stub")
    else:
        cmd.append("--no-perception-fallback-stub")
    _run_cmd(cmd)
    output_payload = _read_json(json_out)
    metrics = compute_variant_object_persistence_metrics(
        variant_label=variant_label,
        output_payload=output_payload,
        query_bank=query_bank,
        source_mode="offline",
        metrics_override={"perception_backend": str(variant_cfg.get("perception_backend", ""))},
    )
    _write_json(out_dir / "metrics.json", metrics)
    return metrics, " ".join(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small object-persistence uplift pilot for YOLO26n vs YOLO26n+SAM3.")
    parser.add_argument("--manifest", required=True, help="Object-persistence manifest YAML")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest_payload = _load_yaml(manifest_path)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = out_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, manifest_dir / "experiment_manifest.yaml")
    query_bank = _load_query_bank(manifest_payload, manifest_path)
    query_bank_lock_payload = _query_bank_lock_payload(
        manifest_payload=manifest_payload,
        manifest_path=manifest_path,
        query_bank=query_bank,
    )
    write_query_bank_lock(query_bank_lock_payload, manifest_dir / "query_bank_lock.json")
    copy_query_bank_files(query_bank_lock_payload, manifest_dir / "query_banks")
    _write_json(
        manifest_dir / "manifest_resolved.json",
        {
            "manifest_path": str(manifest_path),
            "suite_id": str(manifest_payload.get("suite_id", out_dir.name)),
            "query_bank_id": query_bank.query_bank_id,
            "query_bank_version": query_bank.query_bank_version,
            "query_bank_hash": query_bank.query_bank_hash,
        },
    )

    commands: list[str] = []
    variant_results: dict[str, dict[str, Any]] = {}
    for key in ("baseline", "uplift"):
        variant_cfg = manifest_payload.get(key, {})
        if not isinstance(variant_cfg, dict):
            raise ValueError(f"Manifest field `{key}` must be a mapping.")
        variant_label = str(variant_cfg.get("label", key)).strip() or key
        variant_dir = out_dir / "runs" / variant_label
        variant_dir.mkdir(parents=True, exist_ok=True)
        source_mode = str(variant_cfg.get("source", "fixture")).strip().lower()
        if source_mode == "fixture":
            variant_results[key] = _run_fixture_variant(
                variant_label=variant_label,
                variant_cfg=variant_cfg,
                query_bank=query_bank,
                out_dir=variant_dir,
            )
            commands.append(f"# fixture:{variant_label}")
        elif source_mode == "offline":
            metrics, command_text = _run_offline_variant(
                manifest_payload=manifest_payload,
                manifest_path=manifest_path,
                variant_label=variant_label,
                variant_cfg=variant_cfg,
                query_bank=query_bank,
                out_dir=variant_dir,
            )
            variant_results[key] = metrics
            commands.append(command_text)
        else:
            raise ValueError(f"Unsupported object-persistence variant source: {source_mode}")

    provider_probe = _provider_probe_note()
    outputs = write_object_persistence_compare_outputs(
        out_dir=out_dir,
        suite_id=str(manifest_payload.get("suite_id", out_dir.name)),
        query_bank=query_bank,
        baseline_metrics=variant_results["baseline"],
        uplift_metrics=variant_results["uplift"],
        commands=commands,
        thresholds=manifest_payload.get("report", {}).get("thresholds", {}),
        extra_summary={"provider_probe": provider_probe},
    )
    print(f"saved_suite={out_dir}")
    print(f"saved_compare={out_dir / 'compare'}")
    print(f"saved_table={outputs['table_csv']}")
    print(f"saved_snapshot={outputs['snapshot_json']}")
    print(f"object_persistence_status={outputs['object_persistence_status']}")
    print(f"next_action_recommendation={outputs['next_action_recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
