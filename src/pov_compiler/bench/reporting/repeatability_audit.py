from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

from pov_compiler.bench.reporting.latex import df_to_markdown_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for repeatability-audit reporting.")
    return pd


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> Any:
    lib = _require_pandas()
    if not path.exists():
        return lib.DataFrame()
    try:
        return lib.read_csv(path)
    except Exception:
        return lib.DataFrame()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _safe_mean(values)
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return float(math.sqrt(variance))


def _load_repeat_profile(suite_root: Path) -> tuple[Path | None, dict[str, Any]]:
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    raw_path = str(manifest_payload.get("repeat_profile", "")).strip()
    if not raw_path:
        return None, {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[4] / path).resolve()
    return path, _load_yaml(path)


def _repeat_roots(suite_root: Path) -> list[Path]:
    roots = [suite_root]
    repeats_root = suite_root / "repeats"
    if repeats_root.exists():
        roots.extend(sorted(path for path in repeats_root.iterdir() if path.is_dir() and path.name.startswith("repeat_")))
    return roots


def _provider_noise_flag(summary: dict[str, Any], profile: dict[str, Any]) -> bool:
    availability = str(summary.get("availability", "unavailable")).strip()
    parse_fail = float(_to_float(summary.get("structured_parse_fail_rate_mean")) or 0.0)
    fallback = float(_to_float(summary.get("planner_fallback_rate")) or 0.0)
    parse_limit = float(_to_float(profile.get("max_parse_fail_rate")) or 0.10)
    fallback_limit = float(_to_float(profile.get("max_fallback_rate")) or 0.10)
    return bool(
        availability in {"provider_unavailable", "no_real_call", "partial", "unavailable"}
        or parse_fail > parse_limit
        or fallback > fallback_limit
    )


def _suite_variant(compare_summary: dict[str, Any]) -> str:
    label_a = str(compare_summary.get("label_a", "stub")).strip() or "stub"
    label_b = str(compare_summary.get("label_b", "real")).strip() or "real"
    return f"{label_a}->{label_b}"


def _mean_coverage(diag_snapshot: dict[str, Any]) -> float:
    coverage = diag_snapshot.get("coverage_score_stats", {})
    if isinstance(coverage, dict):
        return float(_to_float(coverage.get("mean")) or 0.0)
    return 0.0


def _stability_flag(
    *,
    runs_count: int,
    selected_uids_count: int,
    coefficient_of_variation: float | None,
    provider_noise_rate: float,
    delta_mean: float,
    delta_std: float,
    profile: dict[str, Any],
) -> str:
    min_runs = int(_to_float(profile.get("min_runs")) or 3)
    small_sample_max = int(_to_float(profile.get("small_sample_selected_uids_max")) or 2)
    provider_noise_cv_max = float(_to_float(profile.get("provider_noise_cv_max")) or 0.25)
    stable_no_effect_abs_mean_max = float(_to_float(profile.get("stable_no_effect_abs_mean_max")) or 0.05)
    stable_no_effect_std_max = float(_to_float(profile.get("stable_no_effect_std_max")) or 0.02)
    if runs_count < min_runs:
        return "weak_evidence"
    if provider_noise_rate > 0.0 and coefficient_of_variation is not None and coefficient_of_variation > provider_noise_cv_max:
        return "provider_noise_driven"
    if selected_uids_count <= small_sample_max:
        return "small_sample_driven"
    if abs(delta_mean) <= stable_no_effect_abs_mean_max and delta_std <= stable_no_effect_std_max:
        return "stable_but_no_effect"
    return "stable"


def _repeatability_status(rows: list[dict[str, Any]]) -> str:
    flags = [str(row.get("stability_flag", "")).strip() for row in rows if str(row.get("stability_flag", "")).strip()]
    if not flags:
        return "weak"
    if any(flag == "weak_evidence" for flag in flags):
        return "weak"
    if any(flag in {"provider_noise_driven", "small_sample_driven"} for flag in flags):
        return "partial"
    return "ok"


def build_repeatability_audit_table(
    *,
    suite_dir: str | Path,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    profile_path, profile = _load_repeat_profile(suite_root)
    if not profile:
        profile = {
            "min_runs": 3,
            "small_sample_selected_uids_max": 2,
            "provider_noise_cv_max": 0.25,
            "max_parse_fail_rate": 0.10,
            "max_fallback_rate": 0.10,
            "stable_no_effect_abs_mean_max": 0.05,
            "stable_no_effect_std_max": 0.02,
        }

    grouped_values: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    run_roots = _repeat_roots(suite_root)
    selected_uids_count = 0
    coverage_score_mean = 0.0
    provider_noise_rates: list[float] = []
    repeat_latency_values: list[float] = []

    for run_root in run_roots:
        compare_summary = _read_json(run_root / "compare" / "compare_summary.json")
        main_df = _read_csv(run_root / "compare" / "tables" / "table_main_results.csv")
        if main_df.empty:
            continue
        variant = _suite_variant(compare_summary)
        diag_snapshot = _read_json(run_root / "result_diagnosis" / "snapshot.json")
        selected_uids_count = max(selected_uids_count, int(_to_float(diag_snapshot.get("selected_uids_count")) or 0))
        coverage_score_mean = max(coverage_score_mean, _mean_coverage(diag_snapshot))
        provider_summary = _read_json(run_root / "provider_telemetry" / "summary.json")
        noise_flag = _provider_noise_flag(provider_summary, profile)
        provider_noise_rates.append(1.0 if noise_flag else 0.0)
        latency_p95 = _to_float(provider_summary.get("model_latency_p95_ms_mean"))
        if latency_p95 is not None:
            repeat_latency_values.append(latency_p95)
        for _, row in main_df.iterrows():
            task = str(row.get("task", "")).strip() or "overall"
            metric = str(row.get("primary_metric", "")).strip() or "primary_metric"
            budget = str(row.get("budget_key", "")).strip() or "n/a"
            delta_value = float(_to_float(row.get("delta")) or 0.0)
            grouped_values[(variant, f"{task}:{metric}", budget)].append(
                {
                    "run_root": str(run_root),
                    "delta_value": delta_value,
                    "provider_noise_flag": noise_flag,
                }
            )

    rows: list[dict[str, Any]] = []
    for (variant, metric, budget), samples in sorted(grouped_values.items()):
        values = [float(sample["delta_value"]) for sample in samples]
        runs_count = len(values)
        mean = _safe_mean(values)
        std = _safe_std(values)
        coefficient_of_variation = abs(std / mean) if abs(mean) > 1e-12 else None
        provider_noise_rate = _safe_mean([1.0 if bool(sample["provider_noise_flag"]) else 0.0 for sample in samples])
        stability_flag = _stability_flag(
            runs_count=runs_count,
            selected_uids_count=selected_uids_count,
            coefficient_of_variation=coefficient_of_variation,
            provider_noise_rate=provider_noise_rate,
            delta_mean=mean,
            delta_std=std,
            profile=profile,
        )
        rows.append(
            {
                "variant": variant,
                "metric": metric,
                "budget": budget,
                "runs_count": int(runs_count),
                "mean": float(mean),
                "std": float(std),
                "min": float(min(values)) if values else 0.0,
                "max": float(max(values)) if values else 0.0,
                "coefficient_of_variation": coefficient_of_variation,
                "stability_flag": stability_flag,
                "provider_noise_flag": bool(provider_noise_rate > 0.0),
                "provider_noise_rate": float(provider_noise_rate),
                "selected_uids_count": int(selected_uids_count),
                "coverage_score_mean": float(coverage_score_mean),
            }
        )

    out_df = lib.DataFrame(rows)
    flag_counter = Counter(str(row.get("stability_flag", "")) for row in rows)
    repeatability_status = _repeatability_status(rows)
    latency_mean = _safe_mean(repeat_latency_values)
    latency_std = _safe_std(repeat_latency_values)
    latency_cv = abs(latency_std / latency_mean) if abs(latency_mean) > 1e-12 else 0.0
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "repeat_profile": {
            "path": str(profile_path) if profile_path is not None else None,
            "profile": profile,
        },
        "repeat_roots": [str(path) for path in run_roots],
        "rows_total": int(len(out_df)),
        "repeat_runs_total": int(len(run_roots)),
        "repeatability_status": repeatability_status,
        "stability_flag_counts": {key: int(value) for key, value in sorted(flag_counter.items())},
        "selected_uids_count": int(selected_uids_count),
        "coverage_score_mean": float(coverage_score_mean),
        "provider_noise_rate_mean": _safe_mean(provider_noise_rates),
        "provider_latency_p95_mean": latency_mean,
        "provider_latency_p95_std": latency_std,
        "provider_latency_p95_cv": latency_cv,
    }
    return out_df, snapshot


def write_repeatability_audit_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
    figure_formats: list[str] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    formats = figure_formats or ["png", "pdf"]
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    out_df, snapshot = build_repeatability_audit_table(suite_dir=suite_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_csv = tables_dir / "table_repeatability_audit.csv"
    table_md = tables_dir / "table_repeatability_audit.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"

    out_df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Repeatability Audit\n\n" + df_to_markdown_table(out_df))

    figure_paths: list[str] = []
    fig_base = figures_dir / "fig_repeatability_variance"
    plt.figure(figsize=(9.2, 4.8))
    if len(out_df) > 0:
        labels = [f"{str(row.get('metric', ''))}\n{str(row.get('budget', ''))}" for _, row in out_df.iterrows()]
        x = list(range(len(labels)))
        means = [float(_to_float(value) or 0.0) for value in out_df["mean"].tolist()]
        stds = [float(_to_float(value) or 0.0) for value in out_df["std"].tolist()]
        plt.bar(x, means, yerr=stds, capsize=4)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel("Delta mean +/- std")
        plt.title("Repeatability Variance")
        plt.grid(True, axis="y", alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No repeatability rows available", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    for ext in formats:
        target = fig_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    report_lines = [
        "# Repeatability Audit",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- repeat_runs_total: `{snapshot.get('repeat_runs_total', 0)}`",
        f"- repeatability_status: `{snapshot.get('repeatability_status', 'weak')}`",
        f"- stability_flag_counts: `{snapshot.get('stability_flag_counts', {})}`",
        f"- selected_uids_count: `{snapshot.get('selected_uids_count', 0)}`",
        f"- provider_noise_rate_mean: `{snapshot.get('provider_noise_rate_mean', 0.0)}`",
        f"- provider_latency_p95_cv: `{snapshot.get('provider_latency_p95_cv', 0.0)}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- figures: `{figure_paths}`",
        f"- report_md: `{report_md}`",
        f"- snapshot_json: `{snapshot_json}`",
    ]
    _write_text(report_md, "\n".join(report_lines))
    snapshot["outputs"] = {
        "table_csv": str(table_csv),
        "table_md": str(table_md),
        "figures": figure_paths,
        "report_md": str(report_md),
        "snapshot_json": str(snapshot_json),
    }
    _write_text(snapshot_json, json.dumps(snapshot, ensure_ascii=False, indent=2))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "report_md": report_md,
        "snapshot_json": snapshot_json,
        "rows_total": int(len(out_df)),
        "repeatability_status": snapshot.get("repeatability_status", "weak"),
        "summary": {
            "stability_flag_counts": snapshot.get("stability_flag_counts", {}),
            "provider_noise_rate_mean": snapshot.get("provider_noise_rate_mean", 0.0),
        },
    }
