from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency should exist in runtime env.
    pd = None

from pov_compiler.bench.query_bank import QueryBank
from pov_compiler.bench.reporting.latex import df_to_markdown_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency should exist in runtime env.
        raise ImportError("pandas is required for query-strength audit reporting.")
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


def _read_csv(path: Path) -> Any:
    lib = _require_pandas()
    if not path.exists():
        return lib.DataFrame()
    try:
        return lib.read_csv(path)
    except Exception:
        return lib.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _parse_json_dict(value: Any) -> dict[str, Any]:
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


def _load_primary_query_bank(suite_root: Path) -> QueryBank:
    lock_payload = _read_json(suite_root / "manifest" / "query_bank_lock.json")
    primary = lock_payload.get("primary", {})
    preferred_name = ""
    if isinstance(primary, dict):
        preferred_name = Path(str(primary.get("path", "")).strip()).name
    banks_root = suite_root / "manifest" / "query_banks"
    if banks_root.exists():
        candidates = sorted(banks_root.glob("*.yaml"))
        if preferred_name:
            for candidate in candidates:
                if candidate.name == preferred_name:
                    return QueryBank.from_path(candidate)
        if candidates:
            return QueryBank.from_path(candidates[0])
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    query_bank_path = str(manifest_payload.get("queries", {}).get("query_bank", "")).strip()
    if not query_bank_path:
        raise FileNotFoundError("No query_bank available for query-strength audit.")
    resolved = Path(query_bank_path)
    if not resolved.is_absolute():
        resolved = (Path(__file__).resolve().parents[4] / query_bank_path).resolve()
    return QueryBank.from_path(resolved)


def _relevant_missing_keys(signal_tags: list[str], sensitive_to: list[str]) -> list[str]:
    keys: list[str] = []
    all_tags = [str(item).strip() for item in [*signal_tags, *sensitive_to] if str(item).strip()]
    if any(tag in {"place"} for tag in all_tags):
        keys.append("missing_place")
    if any(tag in {"lost_object", "object_memory"} for tag in all_tags):
        keys.append("missing_lost_object")
    if any(tag in {"chain", "streaming", "decision"} for tag in all_tags):
        keys.append("missing_interaction")
    return sorted(dict.fromkeys(keys))


def _provider_noise_flag(provider_summary: dict[str, Any], manifest_payload: dict[str, Any]) -> bool:
    parse_fail = _to_float(provider_summary.get("structured_parse_fail_rate_mean"))
    fallback = _to_float(provider_summary.get("planner_fallback_rate"))
    parse_limit = float(_to_float(manifest_payload.get("max_provider_noise_parse_fail_rate")) or 0.10)
    fallback_limit = float(_to_float(manifest_payload.get("max_provider_noise_fallback_rate")) or 0.10)
    availability = str(provider_summary.get("availability", "unavailable")).strip()
    return bool(
        availability in {"provider_unavailable", "no_real_call", "partial", "unavailable"}
        or (parse_fail is not None and parse_fail > parse_limit)
        or (fallback is not None and fallback > fallback_limit)
    )


def _recommended_action(
    *,
    matched_tasks: list[str],
    coverage_rate: float,
    signal_support_rate: float,
    nonzero_delta_rate: float,
    significance_available_rate: float,
    provider_noise_flag: bool,
) -> str:
    if not matched_tasks:
        return "keep_for_analysis_only"
    if signal_support_rate < 0.35:
        return "drop_from_main_real"
    if coverage_rate < 0.50:
        return "increase_sample_size"
    if provider_noise_flag and nonzero_delta_rate < 0.50:
        return "keep_for_analysis_only"
    if nonzero_delta_rate >= 0.67 and significance_available_rate >= 0.50 and signal_support_rate >= 0.60:
        return "promote_to_core_query_bank"
    if nonzero_delta_rate < 0.34 and significance_available_rate >= 0.50 and signal_support_rate >= 0.60:
        return "algorithm_no_effect_detected"
    if nonzero_delta_rate < 0.34:
        return "increase_sample_size"
    if signal_support_rate < 0.60:
        return "strengthen_signal_support"
    return "keep_for_analysis_only"


def _promotion_reason(
    *,
    matched_tasks: list[str],
    coverage_rate: float,
    signal_support_rate: float,
    nonzero_delta_rate: float,
    significance_available_rate: float,
    provider_noise_flag: bool,
    recommended_action: str,
) -> str:
    if recommended_action == "promote_to_core_query_bank":
        return "strong_signal_and_nonzero_delta"
    if not matched_tasks:
        return "no_matched_tasks"
    if signal_support_rate < 0.35:
        return "weak_signal_support"
    if coverage_rate < 0.50:
        return "insufficient_sample"
    if provider_noise_flag and nonzero_delta_rate < 0.50:
        return "provider_noise_dominates"
    if significance_available_rate < 0.50:
        return "insufficient_significance"
    if nonzero_delta_rate < 0.34:
        return "algorithm_no_effect_detected"
    return recommended_action


def _uplift_candidate(
    *,
    recommended_action: str,
    provider_noise_flag: bool,
    weak_query_flag: bool,
) -> bool:
    if recommended_action in {"increase_sample_size", "strengthen_signal_support"}:
        return True
    if provider_noise_flag and weak_query_flag and recommended_action not in {"drop_from_main_real", "algorithm_no_effect_detected"}:
        return True
    return False


def _uplift_reason(
    *,
    recommended_action: str,
    provider_noise_flag: bool,
    weak_query_flag: bool,
    signal_support_rate: float,
    coverage_rate: float,
) -> str:
    if recommended_action == "strengthen_signal_support":
        return "signal_support_too_low"
    if recommended_action == "increase_sample_size":
        return "sample_too_small"
    if provider_noise_flag and weak_query_flag:
        return "provider_noise_masks_signal"
    if weak_query_flag and signal_support_rate < 0.55:
        return "weak_query_signal"
    if weak_query_flag and coverage_rate < 0.50:
        return "weak_query_coverage"
    return "not_uplift_candidate"


def build_query_strength_audit_table(
    *,
    suite_dir: str | Path,
    effect_size_threshold: float = 1e-9,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    manifest_payload = _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")
    bank = _load_primary_query_bank(suite_root)
    main_df = _read_csv(suite_root / "compare" / "tables" / "table_main_results.csv")
    sig_df = _read_csv(suite_root / "significance" / "tables" / "table_significance_main.csv")
    health_snapshot = _read_json(suite_root / "result_health" / "snapshot.json")
    diagnosis_snapshot = _read_json(suite_root / "result_diagnosis" / "snapshot.json")
    provider_summary = _read_json(suite_root / "provider_telemetry" / "summary.json")
    selection_payload = health_snapshot.get("selection", {})
    if not isinstance(selection_payload, dict):
        selection_payload = {}
    coverage_stats = selection_payload.get("coverage_score_stats", {})
    if not isinstance(coverage_stats, dict):
        coverage_stats = _parse_json_dict(coverage_stats)
    missing_signal_breakdown = selection_payload.get("missing_signal_breakdown", {})
    if not isinstance(missing_signal_breakdown, dict):
        missing_signal_breakdown = _parse_json_dict(missing_signal_breakdown)

    selected_uids_count = int(
        _to_float(selection_payload.get("selected_uids_count"))
        or _to_float(diagnosis_snapshot.get("selected_uids_count"))
        or 0
    )
    requested_top_k = int(_to_float(manifest_payload.get("selection", {}).get("top_k_uids")) or 0)
    coverage_rate = 0.0
    if requested_top_k > 0:
        coverage_rate = max(0.0, min(1.0, selected_uids_count / float(requested_top_k)))
    coverage_mean = float(_to_float(coverage_stats.get("mean")) or 0.0)
    coverage_count = max(int(_to_float(coverage_stats.get("count")) or 0), 1)
    available_tasks = sorted(
        {str(value).strip() for value in main_df.get("task", lib.Series(dtype=str)).astype(str).tolist() if str(value).strip()}
    )
    provider_noise_flag = _provider_noise_flag(provider_summary, manifest_payload)

    rows: list[dict[str, Any]] = []
    for group in bank.groups:
        group_queries = [query for query in bank.queries if query.group == group.group_id and bool(query.enabled)]
        task_set = sorted({str(query.task).strip() for query in group_queries if str(query.task).strip()})
        mode_set = sorted({str(query.mode).strip() for query in group_queries if str(query.mode).strip()})
        signal_tags = sorted({tag for query in group_queries for tag in query.signal_tags})
        sensitive_to = sorted({tag for query in group_queries for tag in query.sensitive_to})
        matched_tasks = [task for task in task_set if task in available_tasks]

        task_main = main_df.loc[main_df["task"].astype(str).isin(matched_tasks)].copy() if matched_tasks and not main_df.empty else lib.DataFrame()
        delta_series = lib.to_numeric(task_main.get("delta", lib.Series(dtype=float)), errors="coerce").dropna()
        nonzero_delta_rate = float((delta_series.abs() > float(effect_size_threshold)).mean()) if len(delta_series) > 0 else 0.0

        task_sig = sig_df.loc[sig_df["task"].astype(str).isin(matched_tasks)].copy() if matched_tasks and not sig_df.empty else lib.DataFrame()
        if len(task_sig) > 0:
            sig_status = task_sig.get("status", lib.Series([""] * len(task_sig))).astype(str)
            significance_available_rate = float((~sig_status.isin(["task_missing", "insufficient_pairs"])).mean())
        else:
            significance_available_rate = 0.0

        relevant_missing = _relevant_missing_keys(signal_tags, sensitive_to)
        if relevant_missing:
            missing_total = sum(int(_to_float(missing_signal_breakdown.get(key)) or 0) for key in relevant_missing)
            missing_penalty = min(1.0, missing_total / float(coverage_count * len(relevant_missing)))
            base_support = max(0.0, min(1.0, coverage_mean / 3.0)) if coverage_mean > 0.0 else coverage_rate
            signal_support_rate = max(0.0, min(1.0, base_support * (1.0 - missing_penalty)))
        else:
            signal_support_rate = max(0.0, min(1.0, coverage_mean / 3.0)) if coverage_mean > 0.0 else coverage_rate

        weak_query_flag = bool(
            not matched_tasks
            or signal_support_rate < 0.55
            or (nonzero_delta_rate < 0.34 and significance_available_rate < 0.50)
        )
        recommended_action = _recommended_action(
            matched_tasks=matched_tasks,
            coverage_rate=coverage_rate,
            signal_support_rate=signal_support_rate,
            nonzero_delta_rate=nonzero_delta_rate,
            significance_available_rate=significance_available_rate,
            provider_noise_flag=provider_noise_flag,
        )
        promotion_candidate = bool(recommended_action == "promote_to_core_query_bank")
        promotion_reason = _promotion_reason(
            matched_tasks=matched_tasks,
            coverage_rate=coverage_rate,
            signal_support_rate=signal_support_rate,
            nonzero_delta_rate=nonzero_delta_rate,
            significance_available_rate=significance_available_rate,
            provider_noise_flag=provider_noise_flag,
            recommended_action=recommended_action,
        )
        uplift_candidate = _uplift_candidate(
            recommended_action=recommended_action,
            provider_noise_flag=provider_noise_flag,
            weak_query_flag=weak_query_flag,
        )
        uplift_reason = _uplift_reason(
            recommended_action=recommended_action,
            provider_noise_flag=provider_noise_flag,
            weak_query_flag=weak_query_flag,
            signal_support_rate=signal_support_rate,
            coverage_rate=coverage_rate,
        )

        rows.append(
            {
                "query_group": str(group.group_id),
                "query_type": "|".join(mode_set) if mode_set else "|".join(task_set),
                "query_bank_id": bank.query_bank_id,
                "query_ids": json.dumps([query.query_id for query in group_queries], ensure_ascii=False),
                "queries_total": len(group_queries),
                "matched_tasks": json.dumps(matched_tasks, ensure_ascii=False),
                "coverage_rate": float(coverage_rate),
                "signal_support_rate": float(signal_support_rate),
                "nonzero_delta_rate": float(nonzero_delta_rate),
                "significance_available_rate": float(significance_available_rate),
                "weak_query_flag": bool(weak_query_flag),
                "provider_noise_flag": bool(provider_noise_flag),
                "promotion_candidate": promotion_candidate,
                "promotion_reason": promotion_reason,
                "uplift_candidate": uplift_candidate,
                "uplift_reason": uplift_reason,
                "recommended_action": recommended_action,
            }
        )

    out_df = lib.DataFrame(rows)
    if out_df.empty:
        out_df = lib.DataFrame(
            [
                {
                    "query_group": "unknown",
                    "query_type": "",
                    "query_bank_id": bank.query_bank_id,
                    "query_ids": "[]",
                    "queries_total": 0,
                    "matched_tasks": "[]",
                    "coverage_rate": 0.0,
                    "signal_support_rate": 0.0,
                    "nonzero_delta_rate": 0.0,
                    "significance_available_rate": 0.0,
                    "weak_query_flag": True,
                    "provider_noise_flag": provider_noise_flag,
                    "promotion_candidate": False,
                    "promotion_reason": "no_query_groups",
                    "uplift_candidate": True,
                    "uplift_reason": "no_query_groups",
                    "recommended_action": "keep_for_analysis_only",
                }
            ]
        )

    action_counter = Counter(str(value) for value in out_df["recommended_action"].tolist())
    weak_count = int(out_df["weak_query_flag"].astype(bool).sum())
    promoted_groups = sorted(
        out_df.loc[out_df["promotion_candidate"].astype(bool), "query_group"].astype(str).tolist()
    )
    uplift_groups = sorted(
        out_df.loc[out_df["uplift_candidate"].astype(bool), "query_group"].astype(str).tolist()
    )
    analysis_only_groups = sorted(
        out_df.loc[out_df["recommended_action"].astype(str) == "keep_for_analysis_only", "query_group"].astype(str).tolist()
    )
    main_recommendation = ""
    if action_counter:
        main_recommendation = sorted(action_counter.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "query_bank_id": bank.query_bank_id,
        "query_bank_version": bank.query_bank_version,
        "rows_total": int(len(out_df)),
        "selected_uids_count": int(selected_uids_count),
        "coverage_rate": float(coverage_rate),
        "provider_noise_summary": provider_summary,
        "weak_query_groups_count": weak_count,
        "recommended_action_counts": {key: int(value) for key, value in sorted(action_counter.items())},
        "promoted_query_groups": promoted_groups,
        "uplift_candidate_count": int(len(uplift_groups)),
        "uplift_candidate_groups": uplift_groups,
        "analysis_only_query_groups": analysis_only_groups,
        "main_recommendation": main_recommendation,
        "available_tasks": available_tasks,
    }
    return out_df, snapshot


def write_query_strength_audit_outputs(
    *,
    suite_dir: str | Path,
    out_dir: str | Path,
    figure_formats: list[str] | None = None,
    effect_size_threshold: float = 1e-9,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    formats = figure_formats or ["png", "pdf"]
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_df, snapshot = build_query_strength_audit_table(
        suite_dir=suite_dir,
        effect_size_threshold=effect_size_threshold,
    )

    table_csv = tables_dir / "table_query_strength_audit.csv"
    table_md = tables_dir / "table_query_strength_audit.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Query Strength Audit\n\n" + df_to_markdown_table(out_df))

    figure_paths: list[str] = []
    fig_base = figures_dir / "fig_query_strength_breakdown"
    labels = out_df["query_group"].astype(str).tolist()
    x = list(range(len(out_df)))
    width = 0.25
    plt.figure(figsize=(9.4, 4.8))
    plt.bar([item - width for item in x], out_df["coverage_rate"], width=width, label="coverage_rate")
    plt.bar(x, out_df["signal_support_rate"], width=width, label="signal_support_rate")
    plt.bar([item + width for item in x], out_df["nonzero_delta_rate"], width=width, label="nonzero_delta_rate")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Query Strength Breakdown")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        target = fig_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    report_lines = [
        "# Query Strength Audit",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- query_bank_id: `{snapshot.get('query_bank_id', '')}`",
        f"- query_bank_version: `{snapshot.get('query_bank_version', '')}`",
        f"- rows_total: `{snapshot.get('rows_total', 0)}`",
        f"- weak_query_groups_count: `{snapshot.get('weak_query_groups_count', 0)}`",
        f"- uplift_candidate_count: `{snapshot.get('uplift_candidate_count', 0)}`",
        f"- main_recommendation: `{snapshot.get('main_recommendation', '')}`",
        f"- recommended_action_counts: `{snapshot.get('recommended_action_counts', {})}`",
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
        "query_strength_summary": {
            "main_recommendation": snapshot.get("main_recommendation", ""),
            "recommended_action_counts": snapshot.get("recommended_action_counts", {}),
        },
    }
