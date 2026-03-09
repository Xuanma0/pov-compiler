from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency expected in runtime env.
    pd = None

from pov_compiler.bench.query_bank import QueryBank, selection_artifact_paths
from pov_compiler.bench.reporting.latex import df_to_markdown_table
from pov_compiler.bench.reporting.significance import paired_bootstrap_ci, wilcoxon_result


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency expected in runtime env.
        raise ImportError("pandas is required for persistent-memory main reporting.")
    return pd


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


def _read_csv(path: Path) -> Any:
    lib = _require_pandas()
    if not path.exists():
        return lib.DataFrame()
    try:
        return lib.read_csv(path)
    except Exception:
        return lib.DataFrame()


def _load_manifest(run_root: Path) -> dict[str, Any]:
    return _load_yaml(run_root / "manifest" / "experiment_manifest.yaml")


def _load_primary_query_bank(run_root: Path, manifest_payload: dict[str, Any]) -> QueryBank:
    banks_root = run_root / "manifest" / "query_banks"
    compare_summary = _read_json(run_root / "compare" / "compare_summary.json")
    preferred_hash = str(compare_summary.get("query_bank_hash", "")).strip()
    preferred_id = str(compare_summary.get("query_bank_id", "")).strip()
    if banks_root.exists():
        for candidate in sorted(banks_root.glob("*.yaml")):
            bank = QueryBank.from_path(candidate)
            if preferred_hash and bank.query_bank_hash == preferred_hash:
                return bank
            if preferred_id and bank.query_bank_id == preferred_id:
                return bank
        candidates = sorted(banks_root.glob("*.yaml"))
        if candidates:
            return QueryBank.from_path(candidates[0])
    query_bank_path = str(manifest_payload.get("queries", {}).get("query_bank", "")).strip()
    if not query_bank_path:
        raise FileNotFoundError(f"No primary query bank found for run: {run_root}")
    resolved = Path(query_bank_path)
    if not resolved.is_absolute():
        resolved = (Path(__file__).resolve().parents[4] / query_bank_path).resolve()
    return QueryBank.from_path(resolved)


def _selected_uids(run_root: Path, manifest_payload: dict[str, Any], compare_summary: dict[str, Any]) -> list[str]:
    compare_dir_text = str(
        compare_summary.get("compare_dir", manifest_payload.get("selection", {}).get("compare_dir", ""))
    ).strip()
    compare_dir = Path(compare_dir_text) if compare_dir_text else run_root / "compare"
    if not compare_dir.is_absolute():
        compare_dir = (Path(__file__).resolve().parents[4] / compare_dir).resolve()
    manifest_path = run_root / "manifest" / "experiment_manifest.yaml"
    selection_paths = selection_artifact_paths(compare_dir, manifest_path)
    path = selection_paths["selected_uids"]
    if not path.exists():
        return []
    return sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _budget_keys(manifest_payload: dict[str, Any]) -> list[str]:
    budgets = manifest_payload.get("budgets", {})
    if not isinstance(budgets, dict):
        budgets = {}
    points = budgets.get("points", [])
    if not isinstance(points, list):
        return []
    return [
        str(point.get("key", "")).strip()
        for point in points
        if isinstance(point, dict) and str(point.get("key", "")).strip()
    ]


def _provider_signature(run_root: Path) -> tuple[dict[str, Any], str]:
    reachability = _read_json(run_root / "provider_reachability" / "summary.json")
    normalization = _read_json(run_root / "provider_normalization" / "snapshot.json")
    diagnosis = _read_json(run_root / "result_diagnosis" / "snapshot.json")
    provider_noise = (
        diagnosis.get("provider_noise_summary", {})
        if isinstance(diagnosis.get("provider_noise_summary"), dict)
        else {}
    )
    signature = {
        "proof_status": str(reachability.get("proof_status", provider_noise.get("proof_status", "unknown"))),
        "real_call_status": str(
            reachability.get("real_call_status", provider_noise.get("real_call_status", "unknown"))
        ),
        "provider": str(
            reachability.get("provider", provider_noise.get("provider", normalization.get("provider", "")))
        ),
        "model": str(reachability.get("model", provider_noise.get("model", normalization.get("model", "")))),
        "api_mode": str(
            reachability.get(
                "api_mode_tested",
                provider_noise.get("api_mode_used", normalization.get("api_mode_used", "")),
            )
        ),
        "normalization_status": str(
            normalization.get("normalization_status", provider_noise.get("normalization_status", "unknown"))
        ),
        "structured_output_supported": reachability.get(
            "structured_output_supported",
            provider_noise.get("structured_output_supported", "unknown"),
        ),
    }
    return signature, _stable_hash(signature)


def _provider_health_status(signature: dict[str, Any]) -> str:
    proof_status = str(signature.get("proof_status", "unknown"))
    normalization_status = str(signature.get("normalization_status", "unknown"))
    real_call_status = str(signature.get("real_call_status", "unknown"))
    if proof_status == "ok" and normalization_status in {"ok", "partial"} and real_call_status in {"ok", "observed"}:
        return "ok"
    if proof_status in {"ok", "partial"} or normalization_status == "partial":
        return "partial"
    return "fail"


def _perception_signature(manifest_payload: dict[str, Any], compare_summary: dict[str, Any]) -> dict[str, Any]:
    signature = manifest_payload.get("perception_signature", {})
    if isinstance(signature, dict) and signature:
        return signature
    summary_signature = compare_summary.get("perception_signature", {})
    return summary_signature if isinstance(summary_signature, dict) else {}


def _compare_sources(run_root: Path, manifest_payload: dict[str, Any], key: str) -> dict[str, Path]:
    selection = manifest_payload.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    compare_dir = Path(str(selection.get("compare_dir", "")).strip())
    if not compare_dir.is_absolute():
        compare_dir = (Path(__file__).resolve().parents[4] / compare_dir).resolve()
    labels = selection.get("labels", {})
    if not isinstance(labels, dict):
        labels = {}
    variants = manifest_payload.get("variants", {})
    if not isinstance(variants, dict):
        variants = {}
    treatment_key = str(variants.get("treatment", "b")).strip() or "b"
    treatment_label = str(labels.get(treatment_key, treatment_key)).strip() or treatment_key
    sources = selection.get(key, {})
    if not isinstance(sources, dict):
        sources = {}
    resolved: dict[str, Path] = {}
    for task, pattern in sources.items():
        template = str(pattern or "").strip()
        if not template:
            continue
        resolved[str(task)] = (compare_dir / template.replace("{label}", treatment_label)).resolve()
    return resolved


def _load_main_metric_rows(run_root: Path) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    manifest_payload = _load_manifest(run_root)
    source_paths = _compare_sources(run_root, manifest_payload, "task_sources")
    nlq_df = _read_csv(source_paths.get("nlq", Path())) if "nlq" in source_paths else lib.DataFrame()
    if nlq_df.empty:
        return lib.DataFrame(), {"source_paths": {task: str(path) for task, path in source_paths.items()}}
    rows: list[dict[str, Any]] = []
    for _, row in nlq_df.iterrows():
        rows.append(
            {
                "budget_key": str(row.get("budget_key", "")).strip(),
                "budget_seconds": float(_to_float(row.get("budget_seconds")) or 0.0),
                "mrr_strict": _to_float(row.get("mrr_strict", row.get("nlq_full_hit_at_k_strict"))),
                "mrr_relaxed": _to_float(
                    row.get("mrr_relaxed", row.get("mrr_strict", row.get("nlq_full_hit_at_k_strict")))
                ),
                "distractor_rate": _to_float(row.get("top1_in_distractor_rate")),
                "critical_fn_rate": _to_float(row.get("critical_fn_rate")),
            }
        )
    return lib.DataFrame(rows), {"source_paths": {task: str(path) for task, path in source_paths.items()}}


def _load_pair_metrics(run_root: Path) -> Any:
    lib = _require_pandas()
    manifest_payload = _load_manifest(run_root)
    source_paths = _compare_sources(run_root, manifest_payload, "pair_sources")
    nlq_pairs = _read_csv(source_paths.get("nlq", Path())) if "nlq" in source_paths else lib.DataFrame()
    if nlq_pairs.empty:
        return lib.DataFrame()
    out = nlq_pairs.copy()
    if "mrr_strict" not in out.columns and "nlq_full_hit_at_k_strict" in out.columns:
        out["mrr_strict"] = out["nlq_full_hit_at_k_strict"]
    if "mrr_relaxed" not in out.columns:
        out["mrr_relaxed"] = out["mrr_strict"]
    out["budget_key"] = out["budget_key"].astype(str)
    out["sample_unit"] = out["sample_unit"].astype(str)
    return out


def _load_memory_metrics(run_root: Path) -> dict[str, Any]:
    compare_metrics = _read_json(run_root / "compare" / "persistent_memory_main_metrics.json")
    query_strength_snapshot = _read_json(run_root / "query_strength_audit" / "snapshot.json")
    out = dict(compare_metrics)
    query_strength_block = out.get("query_strength", {})
    if not isinstance(query_strength_block, dict):
        query_strength_block = {}
    out.setdefault("query_strength_coverage_rate", _to_float(query_strength_block.get("coverage_rate")))
    out.setdefault("weak_query_groups_count", query_strength_block.get("weak_query_groups_count"))
    if out.get("query_strength_coverage_rate") is None:
        out["query_strength_coverage_rate"] = _to_float(query_strength_snapshot.get("coverage_rate"))
    if out.get("weak_query_groups_count") is None:
        out["weak_query_groups_count"] = int(query_strength_snapshot.get("weak_query_groups_count", 0) or 0)
    return out


def _run_bundle(run_root: Path) -> dict[str, Any]:
    manifest_payload = _load_manifest(run_root)
    compare_summary = _read_json(run_root / "compare" / "compare_summary.json")
    query_bank = _load_primary_query_bank(run_root, manifest_payload)
    provider_signature, provider_signature_hash = _provider_signature(run_root)
    result_health_snapshot = _read_json(run_root / "result_health" / "snapshot.json")
    compare_df, compare_meta = _load_main_metric_rows(run_root)
    pair_df = _load_pair_metrics(run_root)
    memory_metrics = _load_memory_metrics(run_root)
    return {
        "run_root": str(run_root),
        "manifest_payload": manifest_payload,
        "compare_summary": compare_summary,
        "query_bank": query_bank,
        "selected_uids": _selected_uids(run_root, manifest_payload, compare_summary),
        "budget_keys": _budget_keys(manifest_payload),
        "compare_pair_id": str(compare_summary.get("compare_pair_id", manifest_payload.get("compare_pair_id", ""))).strip(),
        "run_signature_hash": str(compare_summary.get("run_signature_hash", "")).strip(),
        "paired_contract_hash": str(compare_summary.get("paired_contract_hash", "")).strip(),
        "uid_set_id": str(compare_summary.get("uid_set_id", manifest_payload.get("uid_set_id", ""))).strip(),
        "query_bank_id": str(compare_summary.get("query_bank_id", query_bank.query_bank_id)).strip(),
        "query_bank_hash": str(compare_summary.get("query_bank_hash", query_bank.query_bank_hash)).strip(),
        "object_memory_logic_variant": str(
            compare_summary.get("object_memory_logic_variant", memory_metrics.get("object_memory_logic_variant", ""))
        ).strip(),
        "provider_label": str(compare_summary.get("provider_label", memory_metrics.get("provider_label", ""))).strip(),
        "model_route": str(compare_summary.get("model_route", memory_metrics.get("model_route", ""))).strip(),
        "provider_signature": provider_signature,
        "provider_signature_hash": provider_signature_hash,
        "provider_health_status": _provider_health_status(provider_signature),
        "perception_signature": _perception_signature(manifest_payload, compare_summary),
        "admission_status": str(result_health_snapshot.get("admission_status", "unknown")),
        "calibration_status": str(result_health_snapshot.get("calibration_status", "unknown")),
        "compare_df": compare_df,
        "pair_df": pair_df,
        "compare_meta": compare_meta,
        "memory_metrics": memory_metrics,
    }


def _alignment_reasons(run_a: dict[str, Any], run_b: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if run_a["compare_pair_id"] != run_b["compare_pair_id"]:
        reasons.append("compare_pair_id_mismatch")
    if run_a["uid_set_id"] != run_b["uid_set_id"]:
        reasons.append("uid_set_id_mismatch")
    if run_a["selected_uids"] != run_b["selected_uids"]:
        reasons.append("selected_uids_mismatch")
    if run_a["budget_keys"] != run_b["budget_keys"]:
        reasons.append("budget_keys_mismatch")
    if run_a["query_bank_id"] != run_b["query_bank_id"]:
        reasons.append("query_bank_id_mismatch")
    if run_a["query_bank_hash"] != run_b["query_bank_hash"]:
        reasons.append("query_bank_hash_mismatch")
    if run_a["provider_signature"] != run_b["provider_signature"]:
        reasons.append("provider_signature_mismatch")
    if run_a["perception_signature"] != run_b["perception_signature"]:
        reasons.append("perception_signature_mismatch")
    if (
        run_a["paired_contract_hash"]
        and run_b["paired_contract_hash"]
        and run_a["paired_contract_hash"] != run_b["paired_contract_hash"]
    ):
        reasons.append("paired_contract_hash_mismatch")
    if run_a["object_memory_logic_variant"] == run_b["object_memory_logic_variant"]:
        reasons.append("object_memory_logic_variant_same")
    return reasons


def _metric_value(metrics: dict[str, Any], key: str, *, default: float = 0.0) -> float:
    value = _to_float(metrics.get(key))
    return float(value) if value is not None else float(default)


def _int_metric(metrics: dict[str, Any], key: str, *, default: int = 0) -> int:
    value = _to_float(metrics.get(key))
    return int(round(value)) if value is not None else int(default)


def _paired_significance_rows(run_a: dict[str, Any], run_b: dict[str, Any]) -> tuple[Any, int, float, float]:
    lib = _require_pandas()
    pair_a = run_a["pair_df"].copy()
    pair_b = run_b["pair_df"].copy()
    if pair_a.empty or pair_b.empty:
        return (
            lib.DataFrame(
                [
                    {
                        "budget_key": "n/a",
                        "metric": "mrr_strict",
                        "paired_sample_count": 0,
                        "mean_delta": 0.0,
                        "win_rate": 0.0,
                        "status": "insufficient_pairs",
                        "wilcoxon_p_value": None,
                        "ci_low": None,
                        "ci_high": None,
                    }
                ]
            ),
            0,
            0.0,
            0.0,
        )

    merged = lib.merge(
        pair_a,
        pair_b,
        on=["budget_key", "sample_unit"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    rows: list[dict[str, Any]] = []
    deltas_strict: list[float] = []
    deltas_relaxed: list[float] = []
    wins_strict = 0
    wins_relaxed = 0
    paired_sample_count = 0
    for budget_key in sorted(merged["budget_key"].astype(str).unique().tolist()):
        budget_df = merged.loc[merged["budget_key"].astype(str) == budget_key].copy()
        paired_sample_count += int(len(budget_df))
        for metric in ("mrr_strict", "mrr_relaxed"):
            col_a = f"{metric}_a"
            col_b = f"{metric}_b"
            if col_a not in budget_df.columns or col_b not in budget_df.columns:
                continue
            baseline = budget_df[col_a].astype(float).tolist()
            treatment = budget_df[col_b].astype(float).tolist()
            n_pairs = len(baseline)
            if n_pairs < 2:
                mean_delta = 0.0
                win_rate = 0.0
                wilcoxon_payload = {"status": "insufficient_pairs", "p_value": None}
                ci_payload = {"status": "insufficient_pairs", "ci_low": None, "ci_high": None}
            else:
                deltas = [float(t) - float(b) for b, t in zip(baseline, treatment)]
                mean_delta = float(sum(deltas) / float(len(deltas)))
                win_rate = float(sum(1 for delta in deltas if delta > 0.0) / float(len(deltas)))
                wilcoxon_payload = wilcoxon_result(baseline, treatment)
                ci_payload = paired_bootstrap_ci(baseline, treatment, seed=0)
                if metric == "mrr_strict":
                    deltas_strict.extend(deltas)
                    wins_strict += int(sum(1 for delta in deltas if delta > 0.0))
                else:
                    deltas_relaxed.extend(deltas)
                    wins_relaxed += int(sum(1 for delta in deltas if delta > 0.0))
            rows.append(
                {
                    "budget_key": budget_key,
                    "metric": metric,
                    "paired_sample_count": int(n_pairs),
                    "mean_delta": round(mean_delta, 6),
                    "win_rate": round(win_rate, 6),
                    "status": str(wilcoxon_payload.get("status", "unknown")),
                    "wilcoxon_p_value": wilcoxon_payload.get("p_value"),
                    "ci_low": ci_payload.get("ci_low"),
                    "ci_high": ci_payload.get("ci_high"),
                }
            )
    win_rate_strict = float(wins_strict / len(deltas_strict)) if deltas_strict else 0.0
    win_rate_relaxed = float(wins_relaxed / len(deltas_relaxed)) if deltas_relaxed else 0.0
    return lib.DataFrame(rows), paired_sample_count, win_rate_strict, win_rate_relaxed


def build_persistent_memory_main_compare_artifacts(*, run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    lib = _require_pandas()
    run_a_root = Path(run_a).resolve()
    run_b_root = Path(run_b).resolve()
    bundle_a = _run_bundle(run_a_root)
    bundle_b = _run_bundle(run_b_root)
    mismatch_reasons = _alignment_reasons(bundle_a, bundle_b)
    alignment_ok = not mismatch_reasons

    df_a = bundle_a["compare_df"].copy()
    df_b = bundle_b["compare_df"].copy()
    merged = lib.merge(
        df_a,
        df_b,
        on=["budget_key", "budget_seconds"],
        how="outer",
        suffixes=("_a", "_b"),
    ).sort_values(by=["budget_seconds", "budget_key"])

    metrics_a = bundle_a["memory_metrics"]
    metrics_b = bundle_b["memory_metrics"]
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        mrr_strict_a = _to_float(row.get("mrr_strict_a"))
        mrr_strict_b = _to_float(row.get("mrr_strict_b"))
        mrr_relaxed_a = _to_float(row.get("mrr_relaxed_a"))
        mrr_relaxed_b = _to_float(row.get("mrr_relaxed_b"))
        rows.append(
            {
                "query_bank_a_id": bundle_a["query_bank_id"],
                "query_bank_b_id": bundle_b["query_bank_id"],
                "query_bank_a_hash": bundle_a["query_bank_hash"],
                "query_bank_b_hash": bundle_b["query_bank_hash"],
                "object_memory_logic_variant_a": bundle_a["object_memory_logic_variant"],
                "object_memory_logic_variant_b": bundle_b["object_memory_logic_variant"],
                "selected_uids_count": int(len(bundle_a["selected_uids"])),
                "budgets_matched": bundle_a["budget_keys"] == bundle_b["budget_keys"],
                "budget_key": str(row.get("budget_key", "")),
                "budget_seconds": float(_to_float(row.get("budget_seconds")) or 0.0),
                "mrr_strict_a": mrr_strict_a,
                "mrr_strict_b": mrr_strict_b,
                "delta_mrr_strict": None
                if mrr_strict_a is None or mrr_strict_b is None
                else round(mrr_strict_b - mrr_strict_a, 6),
                "mrr_relaxed_a": mrr_relaxed_a,
                "mrr_relaxed_b": mrr_relaxed_b,
                "delta_mrr_relaxed": None
                if mrr_relaxed_a is None or mrr_relaxed_b is None
                else round(mrr_relaxed_b - mrr_relaxed_a, 6),
                "distractor_rate_a": _to_float(row.get("distractor_rate_a")),
                "distractor_rate_b": _to_float(row.get("distractor_rate_b")),
                "critical_fn_rate_a": _to_float(row.get("critical_fn_rate_a")),
                "critical_fn_rate_b": _to_float(row.get("critical_fn_rate_b")),
                "query_strength_coverage_a": _metric_value(metrics_a, "query_strength_coverage_rate"),
                "query_strength_coverage_b": _metric_value(metrics_b, "query_strength_coverage_rate"),
                "weak_query_groups_count_a": _int_metric(metrics_a, "weak_query_groups_count"),
                "weak_query_groups_count_b": _int_metric(metrics_b, "weak_query_groups_count"),
                "lost_object_query_support_rate_a": _metric_value(metrics_a, "lost_object_query_support_rate"),
                "lost_object_query_support_rate_b": _metric_value(metrics_b, "lost_object_query_support_rate"),
                "chain_object_grounding_support_rate_a": _metric_value(
                    metrics_a, "chain_object_grounding_support_rate"
                ),
                "chain_object_grounding_support_rate_b": _metric_value(
                    metrics_b, "chain_object_grounding_support_rate"
                ),
                "reappearance_support_rate_a": _metric_value(metrics_a, "reappearance_support_rate"),
                "reappearance_support_rate_b": _metric_value(metrics_b, "reappearance_support_rate"),
                "object_persistence_support_rate_a": _metric_value(metrics_a, "object_persistence_support_rate"),
                "object_persistence_support_rate_b": _metric_value(metrics_b, "object_persistence_support_rate"),
                "object_memory_items_total_a": _int_metric(metrics_a, "object_memory_items_total"),
                "object_memory_items_total_b": _int_metric(metrics_b, "object_memory_items_total"),
                "object_memory_persistence_items_total_a": _int_metric(
                    metrics_a, "object_memory_persistence_items_total"
                ),
                "object_memory_persistence_items_total_b": _int_metric(
                    metrics_b, "object_memory_persistence_items_total"
                ),
                "admission_status_a": bundle_a["admission_status"],
                "admission_status_b": bundle_b["admission_status"],
                "calibration_status_a": bundle_a["calibration_status"],
                "calibration_status_b": bundle_b["calibration_status"],
                "provider_health_status_a": bundle_a["provider_health_status"],
                "provider_health_status_b": bundle_b["provider_health_status"],
                "provider_noise_summary_hash_a": bundle_a["provider_signature_hash"],
                "provider_noise_summary_hash_b": bundle_b["provider_signature_hash"],
                "delta_audit_main_recommendation_a": str(metrics_a.get("delta_audit_main_recommendation", "")),
                "delta_audit_main_recommendation_b": str(metrics_b.get("delta_audit_main_recommendation", "")),
            }
        )
    table_df = lib.DataFrame(rows)
    significance_df, paired_sample_count, win_rate_strict, win_rate_relaxed = _paired_significance_rows(bundle_a, bundle_b)

    mean_delta_mrr_strict = (
        float(table_df["delta_mrr_strict"].dropna().mean())
        if not table_df.empty and table_df["delta_mrr_strict"].notna().any()
        else 0.0
    )
    mean_delta_mrr_relaxed = (
        float(table_df["delta_mrr_relaxed"].dropna().mean())
        if not table_df.empty and table_df["delta_mrr_relaxed"].notna().any()
        else 0.0
    )
    query_strength_delta = _metric_value(metrics_b, "query_strength_coverage_rate") - _metric_value(
        metrics_a, "query_strength_coverage_rate"
    )
    weak_query_groups_delta = _int_metric(metrics_b, "weak_query_groups_count") - _int_metric(
        metrics_a, "weak_query_groups_count"
    )
    lost_object_delta = _metric_value(metrics_b, "lost_object_query_support_rate") - _metric_value(
        metrics_a, "lost_object_query_support_rate"
    )
    chain_delta = _metric_value(metrics_b, "chain_object_grounding_support_rate") - _metric_value(
        metrics_a, "chain_object_grounding_support_rate"
    )
    reappearance_delta = _metric_value(metrics_b, "reappearance_support_rate") - _metric_value(
        metrics_a, "reappearance_support_rate"
    )
    persistence_delta = _metric_value(metrics_b, "object_persistence_support_rate") - _metric_value(
        metrics_a, "object_persistence_support_rate"
    )
    provider_signature_match = bundle_a["provider_signature"] == bundle_b["provider_signature"]
    perception_signature_match = bundle_a["perception_signature"] == bundle_b["perception_signature"]
    if bundle_a["provider_health_status"] != bundle_b["provider_health_status"]:
        provider_health_status = "partial"
    else:
        provider_health_status = bundle_a["provider_health_status"]

    if not alignment_ok:
        persistent_memory_main_status = "regressed"
        main_gain_state = "blocked_alignment_mismatch"
    elif mean_delta_mrr_strict < -0.01:
        persistent_memory_main_status = "regressed"
        main_gain_state = "strict_main_regressed"
    elif (
        mean_delta_mrr_strict > 0.01
        and query_strength_delta > 0.05
        and weak_query_groups_delta <= 0
        and lost_object_delta >= 0.0
        and chain_delta >= 0.0
        and reappearance_delta >= 0.0
        and provider_health_status == "ok"
    ):
        persistent_memory_main_status = "improved"
        main_gain_state = "strict_main_gain_improved"
    elif (
        query_strength_delta > 0.0
        or lost_object_delta > 0.0
        or chain_delta > 0.0
        or reappearance_delta > 0.0
        or persistence_delta > 0.0
    ):
        persistent_memory_main_status = "improved"
        main_gain_state = "memory_side_signals_improved_but_strict_gain_weak"
    else:
        persistent_memory_main_status = "no_change"
        main_gain_state = "no_clear_main_or_memory_gain"

    compare_summary = {
        "run_a": str(run_a_root),
        "run_b": str(run_b_root),
        "query_bank_a_id": bundle_a["query_bank_id"],
        "query_bank_a_hash": bundle_a["query_bank_hash"],
        "query_bank_b_id": bundle_b["query_bank_id"],
        "query_bank_b_hash": bundle_b["query_bank_hash"],
        "object_memory_logic_variant_a": bundle_a["object_memory_logic_variant"],
        "object_memory_logic_variant_b": bundle_b["object_memory_logic_variant"],
        "compare_pair_id": bundle_a["compare_pair_id"],
        "alignment_ok": alignment_ok,
        "mismatch_reasons": mismatch_reasons,
        "selected_uids_count": int(len(bundle_a["selected_uids"])),
        "paired_sample_count": int(paired_sample_count),
        "budget_keys": bundle_a["budget_keys"],
        "provider_signature_match": provider_signature_match,
        "perception_signature_match": perception_signature_match,
        "provider_health_status": provider_health_status,
        "provider_noise_summary_hash_a": bundle_a["provider_signature_hash"],
        "provider_noise_summary_hash_b": bundle_b["provider_signature_hash"],
        "query_strength_coverage_a": _metric_value(metrics_a, "query_strength_coverage_rate"),
        "query_strength_coverage_b": _metric_value(metrics_b, "query_strength_coverage_rate"),
        "weak_query_groups_count_a": _int_metric(metrics_a, "weak_query_groups_count"),
        "weak_query_groups_count_b": _int_metric(metrics_b, "weak_query_groups_count"),
        "lost_object_query_support_rate_a": _metric_value(metrics_a, "lost_object_query_support_rate"),
        "lost_object_query_support_rate_b": _metric_value(metrics_b, "lost_object_query_support_rate"),
        "chain_object_grounding_support_rate_a": _metric_value(metrics_a, "chain_object_grounding_support_rate"),
        "chain_object_grounding_support_rate_b": _metric_value(metrics_b, "chain_object_grounding_support_rate"),
        "reappearance_support_rate_a": _metric_value(metrics_a, "reappearance_support_rate"),
        "reappearance_support_rate_b": _metric_value(metrics_b, "reappearance_support_rate"),
        "object_persistence_support_rate_a": _metric_value(metrics_a, "object_persistence_support_rate"),
        "object_persistence_support_rate_b": _metric_value(metrics_b, "object_persistence_support_rate"),
        "mean_delta_mrr_strict": round(mean_delta_mrr_strict, 6),
        "mean_delta_mrr_relaxed": round(mean_delta_mrr_relaxed, 6),
        "win_rate_strict": round(win_rate_strict, 6),
        "win_rate_relaxed": round(win_rate_relaxed, 6),
        "admission_status_a": bundle_a["admission_status"],
        "admission_status_b": bundle_b["admission_status"],
        "calibration_status_a": bundle_a["calibration_status"],
        "calibration_status_b": bundle_b["calibration_status"],
        "persistent_memory_main_status": persistent_memory_main_status,
        "main_gain_state": main_gain_state,
    }
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alignment": {
            "alignment_ok": alignment_ok,
            "mismatch_reasons": mismatch_reasons,
            "compare_pair_id": bundle_a["compare_pair_id"],
            "uid_set_id_a": bundle_a["uid_set_id"],
            "uid_set_id_b": bundle_b["uid_set_id"],
            "selected_uids": bundle_a["selected_uids"],
            "budget_keys": bundle_a["budget_keys"],
            "provider_signature_a": bundle_a["provider_signature"],
            "provider_signature_b": bundle_b["provider_signature"],
            "perception_signature_a": bundle_a["perception_signature"],
            "perception_signature_b": bundle_b["perception_signature"],
            "run_signature_hash_a": bundle_a["run_signature_hash"],
            "run_signature_hash_b": bundle_b["run_signature_hash"],
            "paired_contract_hash_a": bundle_a["paired_contract_hash"],
            "paired_contract_hash_b": bundle_b["paired_contract_hash"],
        },
        "compare_summary": compare_summary,
        "sources": {
            "run_a_compare_sources": bundle_a["compare_meta"].get("source_paths", {}),
            "run_b_compare_sources": bundle_b["compare_meta"].get("source_paths", {}),
            "run_a_root": str(run_a_root),
            "run_b_root": str(run_b_root),
        },
    }
    return {
        "table_df": table_df,
        "significance_df": significance_df,
        "compare_summary": compare_summary,
        "snapshot": snapshot,
        "alignment_ok": alignment_ok,
        "mismatch_reasons": mismatch_reasons,
    }


def write_persistent_memory_main_compare_outputs(
    *,
    run_a: str | Path,
    run_b: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    out_root = Path(out_dir).resolve()
    compare_dir = out_root / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts = build_persistent_memory_main_compare_artifacts(run_a=run_a, run_b=run_b)
    table_df = artifacts["table_df"]
    significance_df = artifacts["significance_df"]
    table_csv = tables_dir / "table_persistent_memory_main_compare.csv"
    table_md = tables_dir / "table_persistent_memory_main_compare.md"
    significance_csv = tables_dir / "table_persistent_memory_main_significance.csv"
    significance_md = tables_dir / "table_persistent_memory_main_significance.md"
    compare_summary_json = compare_dir / "compare_summary.json"
    snapshot_json = compare_dir / "snapshot.json"
    commands_sh = compare_dir / "commands.sh"
    readme_md = compare_dir / "README.md"
    table_df.to_csv(table_csv, index=False)
    significance_df.to_csv(significance_csv, index=False)
    _write_text(table_md, "# Persistent Memory Main Compare\n\n" + df_to_markdown_table(table_df))
    _write_text(significance_md, "# Persistent Memory Main Significance\n\n" + df_to_markdown_table(significance_df))
    _write_json(compare_summary_json, artifacts["compare_summary"])
    _write_json(snapshot_json, artifacts["snapshot"])
    _write_text(
        commands_sh,
        "\n".join(
            [
                (
                    f"python scripts/compare_persistent_memory_main.py --run_a {Path(run_a).resolve()} "
                    f"--run_b {Path(run_b).resolve()} --out_dir {out_root}"
                )
            ]
        ),
    )

    figure_paths: list[str] = []
    fig_delta_base = figures_dir / "fig_persistent_memory_main_delta"
    plt.figure(figsize=(8.4, 4.8))
    if not table_df.empty:
        xs = table_df["budget_key"].astype(str).tolist()
        ys_strict = [float(_to_float(value) or 0.0) for value in table_df["delta_mrr_strict"].tolist()]
        ys_relaxed = [float(_to_float(value) or 0.0) for value in table_df["delta_mrr_relaxed"].tolist()]
        plt.plot(xs, ys_strict, marker="o", label="delta_mrr_strict")
        plt.plot(xs, ys_relaxed, marker="s", label="delta_mrr_relaxed")
        plt.axhline(y=0.0, linewidth=1.0)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "no rows", ha="center", va="center")
    plt.title("Persistent Memory Main Delta")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_delta_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    fig_health_base = figures_dir / "fig_persistent_memory_main_health"
    plt.figure(figsize=(8.4, 4.8))
    health_labels = ["lost_object", "chain_grounding", "reappearance", "query_strength"]
    health_values = [
        artifacts["compare_summary"]["lost_object_query_support_rate_b"]
        - artifacts["compare_summary"]["lost_object_query_support_rate_a"],
        artifacts["compare_summary"]["chain_object_grounding_support_rate_b"]
        - artifacts["compare_summary"]["chain_object_grounding_support_rate_a"],
        artifacts["compare_summary"]["reappearance_support_rate_b"]
        - artifacts["compare_summary"]["reappearance_support_rate_a"],
        artifacts["compare_summary"]["query_strength_coverage_b"]
        - artifacts["compare_summary"]["query_strength_coverage_a"],
    ]
    plt.bar(health_labels, health_values)
    plt.axhline(y=0.0, linewidth=1.0)
    plt.title("Persistent Memory Main Health Deltas")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_health_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    fig_query_strength_base = figures_dir / "fig_persistent_memory_main_query_strength"
    plt.figure(figsize=(8.4, 4.8))
    labels = ["baseline", "persistent"]
    coverage = [
        artifacts["compare_summary"]["query_strength_coverage_a"],
        artifacts["compare_summary"]["query_strength_coverage_b"],
    ]
    weak_groups = [
        artifacts["compare_summary"]["weak_query_groups_count_a"],
        artifacts["compare_summary"]["weak_query_groups_count_b"],
    ]
    xs = range(len(labels))
    plt.bar([x - 0.15 for x in xs], coverage, width=0.3, label="coverage")
    plt.bar([x + 0.15 for x in xs], weak_groups, width=0.3, label="weak_groups")
    plt.xticks(list(xs), labels)
    plt.title("Persistent Memory Main Query Strength")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_query_strength_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    readme_lines = [
        "# Persistent Memory Main Compare",
        "",
        f"- query_bank_id: `{artifacts['compare_summary'].get('query_bank_a_id', '')}`",
        f"- object_memory_logic_variant_a: `{artifacts['compare_summary'].get('object_memory_logic_variant_a', '')}`",
        f"- object_memory_logic_variant_b: `{artifacts['compare_summary'].get('object_memory_logic_variant_b', '')}`",
        f"- alignment_ok: `{artifacts['compare_summary'].get('alignment_ok', False)}`",
        f"- mismatch_reasons: `{artifacts['compare_summary'].get('mismatch_reasons', [])}`",
        f"- persistent_memory_main_status: `{artifacts['compare_summary'].get('persistent_memory_main_status', 'no_change')}`",
        f"- main_gain_state: `{artifacts['compare_summary'].get('main_gain_state', '')}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- significance_csv: `{significance_csv}`",
        f"- significance_md: `{significance_md}`",
        f"- compare_summary_json: `{compare_summary_json}`",
        f"- snapshot_json: `{snapshot_json}`",
        f"- commands_sh: `{commands_sh}`",
        f"- figures: `{figure_paths}`",
    ]
    _write_text(readme_md, "\n".join(readme_lines))
    if not artifacts["alignment_ok"]:
        raise ValueError("persistent-memory main compare alignment failed: " + ", ".join(artifacts["mismatch_reasons"]))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "significance_csv": significance_csv,
        "significance_md": significance_md,
        "compare_summary_json": compare_summary_json,
        "snapshot_json": snapshot_json,
        "commands_sh": commands_sh,
        "readme_md": readme_md,
        "figure_paths": figure_paths,
        "compare_summary": artifacts["compare_summary"],
    }


def build_persistent_memory_main_decision(*, compare_dir: str | Path) -> dict[str, Any]:
    requested_root = Path(compare_dir).resolve()
    compare_root = requested_root
    summary = _read_json(compare_root / "compare_summary.json")
    if not summary and (requested_root / "compare").exists():
        compare_root = requested_root / "compare"
        summary = _read_json(compare_root / "compare_summary.json")
    if not summary:
        raise FileNotFoundError(f"compare summary not found under {requested_root}")
    if not bool(summary.get("alignment_ok", False)):
        raise ValueError("Cannot build persistent-memory main decision from a misaligned compare.")

    mean_delta_mrr_strict = float(_to_float(summary.get("mean_delta_mrr_strict")) or 0.0)
    mean_delta_mrr_relaxed = float(_to_float(summary.get("mean_delta_mrr_relaxed")) or 0.0)
    query_strength_delta = float(_to_float(summary.get("query_strength_coverage_b")) or 0.0) - float(
        _to_float(summary.get("query_strength_coverage_a")) or 0.0
    )
    weak_query_groups_delta = int(_to_float(summary.get("weak_query_groups_count_b")) or 0) - int(
        _to_float(summary.get("weak_query_groups_count_a")) or 0
    )
    lost_object_delta = float(_to_float(summary.get("lost_object_query_support_rate_b")) or 0.0) - float(
        _to_float(summary.get("lost_object_query_support_rate_a")) or 0.0
    )
    chain_delta = float(_to_float(summary.get("chain_object_grounding_support_rate_b")) or 0.0) - float(
        _to_float(summary.get("chain_object_grounding_support_rate_a")) or 0.0
    )
    reappearance_delta = float(_to_float(summary.get("reappearance_support_rate_b")) or 0.0) - float(
        _to_float(summary.get("reappearance_support_rate_a")) or 0.0
    )
    persistence_delta = float(_to_float(summary.get("object_persistence_support_rate_b")) or 0.0) - float(
        _to_float(summary.get("object_persistence_support_rate_a")) or 0.0
    )
    paired_sample_count = int(_to_float(summary.get("paired_sample_count")) or 0)
    provider_health_status = str(summary.get("provider_health_status", "unknown"))
    admission_status_b = str(summary.get("admission_status_b", "unknown"))
    calibration_status_b = str(summary.get("calibration_status_b", "unknown"))

    promotion_decision = "keep_current_mainline"
    decision_confidence = "low"
    primary_basis = "no_clear_improvement"
    secondary_basis = "baseline_more_conservative"
    recommended_next_step = "keep_current_mainline"

    signal_gains = all(
        delta >= 0.0 for delta in (lost_object_delta, chain_delta, reappearance_delta, persistence_delta)
    ) and query_strength_delta >= 0.0 and weak_query_groups_delta <= 0
    cleaner_run_needed = provider_health_status != "ok" or paired_sample_count < 8

    if cleaner_run_needed:
        promotion_decision = "need_cleaner_large_sample_run"
        decision_confidence = "medium"
        primary_basis = "provider_or_sample_contract_not_clean_enough"
        secondary_basis = "large_sample_run_needs_more_stability"
        recommended_next_step = "need_cleaner_large_sample_run"
    elif mean_delta_mrr_strict > 0.02 and signal_gains and admission_status_b in {"ok", "partial"} and calibration_status_b in {"ok", "partial"}:
        promotion_decision = "promote_persistent_memory_to_mainline"
        decision_confidence = "high" if mean_delta_mrr_strict > 0.04 else "medium"
        primary_basis = "strict_and_memory_signals_improved"
        secondary_basis = "weak_query_groups_nonincreasing"
        recommended_next_step = "promote_persistent_memory_to_mainline"
    elif signal_gains and mean_delta_mrr_strict > 0.0:
        promotion_decision = "promote_persistent_memory_to_mainline"
        decision_confidence = "medium"
        primary_basis = "directionally_positive_with_clean_large_sample"
        secondary_basis = "memory_side_signals_consistently_improved"
        recommended_next_step = "promote_persistent_memory_to_mainline"
    elif signal_gains and mean_delta_mrr_strict >= -0.005:
        promotion_decision = "need_retrieval_side_fix"
        decision_confidence = "medium"
        primary_basis = "memory_signals_improved_but_strict_gain_weak"
        secondary_basis = "query_strength_gain_not_translated"
        recommended_next_step = "need_retrieval_side_fix"
    elif signal_gains:
        promotion_decision = "need_decision_side_fix"
        decision_confidence = "medium"
        primary_basis = "memory_signals_up_but_decision_path_not_improving"
        secondary_basis = "strict_metrics_still_flat"
        recommended_next_step = "need_decision_side_fix"
    else:
        promotion_decision = "persistent_memory_gain_insufficient"
        decision_confidence = "low"
        primary_basis = "memory_side_gain_insufficient"
        secondary_basis = "strict_metrics_not_improved"
        recommended_next_step = "keep_current_mainline"

    row = {
        "promotion_decision": promotion_decision,
        "decision_confidence": decision_confidence,
        "primary_basis": primary_basis,
        "secondary_basis": secondary_basis,
        "promotion_ready": bool(promotion_decision == "promote_persistent_memory_to_mainline"),
        "recommended_next_step": recommended_next_step,
        "mean_delta_mrr_strict": round(mean_delta_mrr_strict, 6),
        "mean_delta_mrr_relaxed": round(mean_delta_mrr_relaxed, 6),
        "query_strength_delta": round(query_strength_delta, 6),
        "weak_query_groups_delta": int(weak_query_groups_delta),
        "lost_object_delta": round(lost_object_delta, 6),
        "chain_object_grounding_delta": round(chain_delta, 6),
        "reappearance_delta": round(reappearance_delta, 6),
        "object_persistence_delta": round(persistence_delta, 6),
        "paired_sample_count": int(paired_sample_count),
        "provider_health_status": provider_health_status,
        "admission_status_b": admission_status_b,
        "calibration_status_b": calibration_status_b,
    }
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "compare_dir": str(compare_root),
        "compare_dir_requested": str(requested_root),
        "compare_summary": summary,
        "promotion_decision_summary": row,
    }
    return {"row": row, "snapshot": snapshot}


def write_persistent_memory_main_decision_outputs(
    *,
    compare_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    lib = _require_pandas()
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_persistent_memory_main_decision(compare_dir=compare_dir)
    table_csv = tables_dir / "table_persistent_memory_main_decision.csv"
    table_md = tables_dir / "table_persistent_memory_main_decision.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    df = lib.DataFrame([artifacts["row"]])
    df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Persistent Memory Main Decision\n\n" + df_to_markdown_table(df))
    _write_json(snapshot_json, artifacts["snapshot"])
    report_lines = [
        "# Persistent Memory Main Decision",
        "",
        f"- promotion_decision: `{artifacts['row']['promotion_decision']}`",
        f"- decision_confidence: `{artifacts['row']['decision_confidence']}`",
        f"- primary_basis: `{artifacts['row']['primary_basis']}`",
        f"- secondary_basis: `{artifacts['row']['secondary_basis']}`",
        f"- recommended_next_step: `{artifacts['row']['recommended_next_step']}`",
        f"- promotion_ready: `{artifacts['row']['promotion_ready']}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- report_md: `{report_md}`",
        f"- snapshot_json: `{snapshot_json}`",
    ]
    _write_text(report_md, "\n".join(report_lines))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "report_md": report_md,
        "snapshot_json": snapshot_json,
        "promotion_decision_summary": artifacts["row"],
    }
