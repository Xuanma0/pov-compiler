from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency expected in runtime env.
    pd = None

from pov_compiler.bench.query_bank import QueryBank, selection_artifact_paths
from pov_compiler.bench.reporting.latex import df_to_markdown_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency expected in runtime env.
        raise ImportError("pandas is required for query-bank comparison reporting.")
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
    compare_dir_text = str(compare_summary.get("compare_dir", manifest_payload.get("selection", {}).get("compare_dir", ""))).strip()
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
    return [str(point.get("key", "")).strip() for point in points if isinstance(point, dict) and str(point.get("key", "")).strip()]


def _provider_signature(run_root: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    reachability = _read_json(run_root / "provider_reachability" / "summary.json")
    normalization = _read_json(run_root / "provider_normalization" / "snapshot.json")
    diagnosis = _read_json(run_root / "result_diagnosis" / "snapshot.json")
    provider_noise = diagnosis.get("provider_noise_summary", {}) if isinstance(diagnosis.get("provider_noise_summary"), dict) else {}
    signature = {
        "proof_status": str(reachability.get("proof_status", provider_noise.get("proof_status", "unknown"))),
        "real_call_status": str(reachability.get("real_call_status", provider_noise.get("real_call_status", "unknown"))),
        "provider": str(reachability.get("provider", provider_noise.get("provider", normalization.get("provider", "")))),
        "model": str(reachability.get("model", provider_noise.get("model", normalization.get("model", "")))),
        "api_mode": str(reachability.get("api_mode_tested", provider_noise.get("api_mode_used", normalization.get("api_mode_used", "")))),
        "structured_output_supported": reachability.get(
            "structured_output_supported",
            provider_noise.get("structured_output_supported", "unknown"),
        ),
        "normalization_status": str(
            normalization.get("normalization_status", provider_noise.get("normalization_status", "unknown"))
        ),
    }
    return signature, _stable_hash(provider_noise or normalization), provider_noise or normalization


def _perception_signature(manifest_payload: dict[str, Any], compare_summary: dict[str, Any]) -> dict[str, Any]:
    signature = manifest_payload.get("perception_signature", {})
    if isinstance(signature, dict) and signature:
        return signature
    summary_signature = compare_summary.get("perception_signature", {})
    return summary_signature if isinstance(summary_signature, dict) else {}


def _compare_sources(run_root: Path, manifest_payload: dict[str, Any]) -> dict[str, Path]:
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
    task_sources = selection.get("task_sources", {})
    if not isinstance(task_sources, dict):
        task_sources = {}
    resolved: dict[str, Path] = {}
    for task, pattern in task_sources.items():
        template = str(pattern or "").strip()
        if not template:
            continue
        resolved[str(task)] = (compare_dir / template.replace("{label}", treatment_label)).resolve()
    return resolved


def _load_compare_metric_rows(run_root: Path) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    manifest_payload = _load_manifest(run_root)
    compare_summary = _read_json(run_root / "compare" / "compare_summary.json")
    source_paths = _compare_sources(run_root, manifest_payload)
    nlq_df = _read_csv(source_paths.get("nlq", Path())) if "nlq" in source_paths else lib.DataFrame()
    if nlq_df.empty:
        return lib.DataFrame(), {}
    rows: list[dict[str, Any]] = []
    for _, row in nlq_df.iterrows():
        rows.append(
            {
                "budget_key": str(row.get("budget_key", "")).strip(),
                "budget_seconds": float(_to_float(row.get("budget_seconds")) or 0.0),
                "mrr_strict": _to_float(row.get("mrr_strict")),
                "distractor_rate": _to_float(row.get("top1_in_distractor_rate")),
                "critical_fn_rate": _to_float(row.get("critical_fn_rate")),
            }
        )
    return lib.DataFrame(rows), {"source_paths": {task: str(path) for task, path in source_paths.items()}}


def _run_alignment(run_root: Path) -> dict[str, Any]:
    manifest_payload = _load_manifest(run_root)
    compare_summary = _read_json(run_root / "compare" / "compare_summary.json")
    query_bank = _load_primary_query_bank(run_root, manifest_payload)
    provider_signature, provider_noise_hash, provider_noise_summary = _provider_signature(run_root)
    query_strength_snapshot = _read_json(run_root / "query_strength_audit" / "snapshot.json")
    admission_snapshot = _read_json(run_root / "admission_control" / "snapshot.json")
    signal_uplift_snapshot = _read_json(run_root / "signal_uplift" / "snapshot.json")
    selected_uids = _selected_uids(run_root, manifest_payload, compare_summary)
    compare_df, compare_meta = _load_compare_metric_rows(run_root)
    return {
        "run_root": str(run_root),
        "manifest_payload": manifest_payload,
        "compare_summary": compare_summary,
        "query_bank": query_bank,
        "selected_uids": selected_uids,
        "budget_keys": _budget_keys(manifest_payload),
        "compare_pair_id": str(
            compare_summary.get("compare_pair_id", manifest_payload.get("compare_pair_id", ""))
        ).strip(),
        "run_signature_hash": str(compare_summary.get("run_signature_hash", "")).strip(),
        "provider_signature": provider_signature,
        "provider_noise_summary_hash": provider_noise_hash,
        "provider_noise_summary": provider_noise_summary,
        "perception_signature": _perception_signature(manifest_payload, compare_summary),
        "query_strength_snapshot": query_strength_snapshot,
        "admission_snapshot": admission_snapshot,
        "signal_uplift_snapshot": signal_uplift_snapshot,
        "compare_df": compare_df,
        "compare_meta": compare_meta,
    }


def _alignment_reasons(run_a: dict[str, Any], run_b: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if run_a["compare_pair_id"] != run_b["compare_pair_id"]:
        reasons.append("compare_pair_id_mismatch")
    if run_a["run_signature_hash"] and run_b["run_signature_hash"] and run_a["run_signature_hash"] != run_b["run_signature_hash"]:
        reasons.append("run_signature_hash_mismatch")
    if run_a["selected_uids"] != run_b["selected_uids"]:
        reasons.append("selected_uids_mismatch")
    if run_a["budget_keys"] != run_b["budget_keys"]:
        reasons.append("budget_keys_mismatch")
    if run_a["provider_signature"] != run_b["provider_signature"]:
        reasons.append("provider_signature_mismatch")
    if run_a["perception_signature"] != run_b["perception_signature"]:
        reasons.append("perception_signature_mismatch")
    return reasons


def build_query_bank_compare_artifacts(*, run_a: str | Path, run_b: str | Path) -> dict[str, Any]:
    lib = _require_pandas()
    run_a_root = Path(run_a).resolve()
    run_b_root = Path(run_b).resolve()
    bundle_a = _run_alignment(run_a_root)
    bundle_b = _run_alignment(run_b_root)
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

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        mrr_a = _to_float(row.get("mrr_strict_a"))
        mrr_b = _to_float(row.get("mrr_strict_b"))
        distractor_a = _to_float(row.get("distractor_rate_a"))
        distractor_b = _to_float(row.get("distractor_rate_b"))
        critical_a = _to_float(row.get("critical_fn_rate_a"))
        critical_b = _to_float(row.get("critical_fn_rate_b"))
        rows.append(
            {
                "query_bank_a_id": bundle_a["query_bank"].query_bank_id,
                "query_bank_b_id": bundle_b["query_bank"].query_bank_id,
                "budget_key": str(row.get("budget_key", "")),
                "budget_seconds": float(_to_float(row.get("budget_seconds")) or 0.0),
                "selected_uids_count": int(len(bundle_a["selected_uids"])),
                "budgets_matched": alignment_ok and bundle_a["budget_keys"] == bundle_b["budget_keys"],
                "mrr_strict_a": mrr_a,
                "mrr_strict_b": mrr_b,
                "delta_mrr_strict": None if mrr_a is None or mrr_b is None else round(mrr_b - mrr_a, 6),
                "distractor_rate_a": distractor_a,
                "distractor_rate_b": distractor_b,
                "critical_fn_rate_a": critical_a,
                "critical_fn_rate_b": critical_b,
                "query_strength_coverage_a": float(bundle_a["query_strength_snapshot"].get("coverage_rate", 0.0)),
                "query_strength_coverage_b": float(bundle_b["query_strength_snapshot"].get("coverage_rate", 0.0)),
                "weak_query_groups_count_a": int(bundle_a["query_strength_snapshot"].get("weak_query_groups_count", 0)),
                "weak_query_groups_count_b": int(bundle_b["query_strength_snapshot"].get("weak_query_groups_count", 0)),
                "admission_status_a": str(bundle_a["admission_snapshot"].get("admission_status", "unknown")),
                "admission_status_b": str(bundle_b["admission_snapshot"].get("admission_status", "unknown")),
                "provider_noise_summary_hash_a": bundle_a["provider_noise_summary_hash"],
                "provider_noise_summary_hash_b": bundle_b["provider_noise_summary_hash"],
            }
        )
    out_df = lib.DataFrame(rows)
    mean_delta_mrr = float(out_df["delta_mrr_strict"].dropna().mean()) if not out_df.empty and out_df["delta_mrr_strict"].notna().any() else 0.0
    compare_summary = {
        "run_a": str(run_a_root),
        "run_b": str(run_b_root),
        "query_bank_a_id": bundle_a["query_bank"].query_bank_id,
        "query_bank_a_hash": bundle_a["query_bank"].query_bank_hash,
        "query_bank_b_id": bundle_b["query_bank"].query_bank_id,
        "query_bank_b_hash": bundle_b["query_bank"].query_bank_hash,
        "compare_pair_id": bundle_a["compare_pair_id"],
        "alignment_ok": alignment_ok,
        "mismatch_reasons": mismatch_reasons,
        "selected_uids_count": int(len(bundle_a["selected_uids"])),
        "budget_keys": bundle_a["budget_keys"],
        "query_strength_coverage_a": float(bundle_a["query_strength_snapshot"].get("coverage_rate", 0.0)),
        "query_strength_coverage_b": float(bundle_b["query_strength_snapshot"].get("coverage_rate", 0.0)),
        "weak_query_groups_count_a": int(bundle_a["query_strength_snapshot"].get("weak_query_groups_count", 0)),
        "weak_query_groups_count_b": int(bundle_b["query_strength_snapshot"].get("weak_query_groups_count", 0)),
        "admission_status_a": str(bundle_a["admission_snapshot"].get("admission_status", "unknown")),
        "admission_status_b": str(bundle_b["admission_snapshot"].get("admission_status", "unknown")),
        "provider_signature_match": bundle_a["provider_signature"] == bundle_b["provider_signature"],
        "perception_signature_match": bundle_a["perception_signature"] == bundle_b["perception_signature"],
        "mean_delta_mrr_strict": mean_delta_mrr,
        "candidate_next_action_recommendation": str(bundle_b["signal_uplift_snapshot"].get("next_action_recommendation", "")),
        "candidate_should_try_sam3_next": bool(bundle_b["signal_uplift_snapshot"].get("should_try_sam3_next", False)),
    }
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "alignment": {
            "alignment_ok": alignment_ok,
            "mismatch_reasons": mismatch_reasons,
            "compare_pair_id": bundle_a["compare_pair_id"],
            "selected_uids": bundle_a["selected_uids"],
            "budget_keys": bundle_a["budget_keys"],
            "provider_signature_a": bundle_a["provider_signature"],
            "provider_signature_b": bundle_b["provider_signature"],
            "perception_signature_a": bundle_a["perception_signature"],
            "perception_signature_b": bundle_b["perception_signature"],
            "run_signature_hash_a": bundle_a["run_signature_hash"],
            "run_signature_hash_b": bundle_b["run_signature_hash"],
        },
        "compare_summary": compare_summary,
        "sources": {
            "run_a_compare_sources": bundle_a["compare_meta"].get("source_paths", {}),
            "run_b_compare_sources": bundle_b["compare_meta"].get("source_paths", {}),
        },
    }
    return {
        "table_df": out_df,
        "compare_summary": compare_summary,
        "snapshot": snapshot,
        "alignment_ok": alignment_ok,
        "mismatch_reasons": mismatch_reasons,
    }


def write_query_bank_compare_outputs(*, run_a: str | Path, run_b: str | Path, out_dir: str | Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    out_root = Path(out_dir).resolve()
    compare_dir = out_root / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts = build_query_bank_compare_artifacts(run_a=run_a, run_b=run_b)
    out_df = artifacts["table_df"]
    table_csv = tables_dir / "table_query_bank_compare.csv"
    table_md = tables_dir / "table_query_bank_compare.md"
    compare_summary_json = compare_dir / "compare_summary.json"
    snapshot_json = compare_dir / "snapshot.json"
    commands_sh = compare_dir / "commands.sh"
    readme_md = compare_dir / "README.md"
    out_df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Query Bank Compare\n\n" + df_to_markdown_table(out_df))
    _write_json(compare_summary_json, artifacts["compare_summary"])
    _write_json(snapshot_json, artifacts["snapshot"])
    _write_text(
        commands_sh,
        "\n".join(
            [
                f"python scripts/compare_query_banks.py --run_a {Path(run_a).resolve()} --run_b {Path(run_b).resolve()} --out_dir {out_root}",
            ]
        ),
    )

    figure_paths: list[str] = []
    fig_delta_base = figures_dir / "fig_query_bank_delta"
    plt.figure(figsize=(8.4, 4.8))
    if not out_df.empty:
        xs = out_df["budget_key"].astype(str).tolist()
        ys = [float(_to_float(value) or 0.0) for value in out_df.get("delta_mrr_strict", [])]
        plt.bar(xs, ys)
        plt.axhline(y=0.0, linewidth=1.0)
    else:
        plt.text(0.5, 0.5, "alignment mismatch", ha="center", va="center")
    plt.title("Query Bank Delta (v2 - v1)")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_delta_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    fig_tradeoff_base = figures_dir / "fig_query_bank_tradeoff"
    plt.figure(figsize=(8.4, 4.8))
    if not out_df.empty:
        x_a = [float(_to_float(value) or 0.0) for value in out_df["query_strength_coverage_a"].tolist()]
        y_a = [float(_to_float(value) or 0.0) for value in out_df["mrr_strict_a"].tolist()]
        x_b = [float(_to_float(value) or 0.0) for value in out_df["query_strength_coverage_b"].tolist()]
        y_b = [float(_to_float(value) or 0.0) for value in out_df["mrr_strict_b"].tolist()]
        plt.scatter(x_a, y_a, label=str(artifacts["compare_summary"].get("query_bank_a_id", "v1")))
        plt.scatter(x_b, y_b, label=str(artifacts["compare_summary"].get("query_bank_b_id", "v2")))
        for idx, budget_key in enumerate(out_df["budget_key"].astype(str).tolist()):
            plt.annotate(budget_key, (x_b[idx], y_b[idx]))
    else:
        plt.text(0.5, 0.5, "alignment mismatch", ha="center", va="center")
    plt.xlabel("Query Strength Coverage")
    plt.ylabel("MRR Strict")
    plt.title("Query Bank Tradeoff")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_tradeoff_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    readme_lines = [
        "# Query Bank Compare",
        "",
        f"- query_bank_a_id: `{artifacts['compare_summary'].get('query_bank_a_id', '')}`",
        f"- query_bank_b_id: `{artifacts['compare_summary'].get('query_bank_b_id', '')}`",
        f"- alignment_ok: `{artifacts['compare_summary'].get('alignment_ok', False)}`",
        f"- mismatch_reasons: `{artifacts['compare_summary'].get('mismatch_reasons', [])}`",
        f"- selected_uids_count: `{artifacts['compare_summary'].get('selected_uids_count', 0)}`",
        f"- mean_delta_mrr_strict: `{artifacts['compare_summary'].get('mean_delta_mrr_strict', 0.0)}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- compare_summary_json: `{compare_summary_json}`",
        f"- snapshot_json: `{snapshot_json}`",
        f"- commands_sh: `{commands_sh}`",
        f"- figures: `{figure_paths}`",
    ]
    _write_text(readme_md, "\n".join(readme_lines))
    if not artifacts["alignment_ok"]:
        raise ValueError("query-bank compare alignment failed: " + ", ".join(artifacts["mismatch_reasons"]))
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "compare_summary_json": compare_summary_json,
        "snapshot_json": snapshot_json,
        "commands_sh": commands_sh,
        "readme_md": readme_md,
        "figure_paths": figure_paths,
        "compare_summary": artifacts["compare_summary"],
    }


def build_query_bank_promotion_decision(*, compare_dir: str | Path) -> dict[str, Any]:
    compare_root = Path(compare_dir).resolve()
    summary = _read_json(compare_root / "compare" / "compare_summary.json")
    if not summary:
        summary = _read_json(compare_root / "compare_summary.json")
    table_path = compare_root / "compare" / "tables" / "table_query_bank_compare.csv"
    if not table_path.exists():
        table_path = compare_root / "tables" / "table_query_bank_compare.csv"
    df = _read_csv(table_path)
    alignment_ok = bool(summary.get("alignment_ok", False))
    if not alignment_ok:
        raise ValueError("Cannot build promotion decision from a misaligned compare.")

    mean_delta_mrr = float(df["delta_mrr_strict"].dropna().mean()) if not df.empty and df["delta_mrr_strict"].notna().any() else 0.0
    query_strength_delta = float(summary.get("query_strength_coverage_b", 0.0)) - float(summary.get("query_strength_coverage_a", 0.0))
    weak_group_delta = int(summary.get("weak_query_groups_count_b", 0)) - int(summary.get("weak_query_groups_count_a", 0))
    distractor_delta = 0.0
    critical_fn_delta = 0.0
    if not df.empty:
        distractor_delta = float(
            (df["distractor_rate_b"].astype(float) - df["distractor_rate_a"].astype(float)).mean()
        )
        critical_fn_delta = float(
            (df["critical_fn_rate_b"].astype(float) - df["critical_fn_rate_a"].astype(float)).mean()
        )

    admission_a = str(summary.get("admission_status_a", "unknown"))
    admission_b = str(summary.get("admission_status_b", "unknown"))
    provider_ok = bool(summary.get("provider_signature_match", False))
    candidate_should_try_sam3 = bool(summary.get("candidate_should_try_sam3_next", False))

    promotion_decision = "keep_v1"
    decision_confidence = "low"
    primary_basis = "no_clear_improvement"
    secondary_basis = "historical_v1_more_stable"
    recommended_next_step = "keep_v1_and_appendix_v2"
    if (
        mean_delta_mrr > 0.01
        and query_strength_delta > 0.10
        and weak_group_delta < 0
        and distractor_delta <= 0.0
        and critical_fn_delta <= 0.0
        and admission_b in {"ok", "partial"}
        and provider_ok
    ):
        promotion_decision = "promote_v2"
        decision_confidence = "high" if mean_delta_mrr > 0.03 else "medium"
        primary_basis = "strict_and_query_strength_improved"
        secondary_basis = "weak_query_groups_reduced"
        recommended_next_step = "freeze_v2_for_next_main_real"
    elif query_strength_delta > 0.10 and weak_group_delta < 0 and mean_delta_mrr >= -0.005 and provider_ok:
        promotion_decision = "expand_sample_first"
        decision_confidence = "medium"
        primary_basis = "query_strength_better_but_strict_gain_small"
        secondary_basis = "alignment_clean_expand_medium_scale"
        recommended_next_step = "expand_sample_with_v2_candidate"
    elif candidate_should_try_sam3:
        promotion_decision = "consider_sam3_next"
        decision_confidence = "medium"
        primary_basis = "signal_improved_but_query_still_weak"
        secondary_basis = "candidate_bank_points_to_segmentation_gap"
        recommended_next_step = "prototype_sam3_after_yolo26n"

    row = {
        "promotion_decision": promotion_decision,
        "decision_confidence": decision_confidence,
        "primary_basis": primary_basis,
        "secondary_basis": secondary_basis,
        "promotion_ready": bool(promotion_decision == "promote_v2"),
        "recommended_next_step": recommended_next_step,
        "mean_delta_mrr_strict": round(mean_delta_mrr, 6),
        "query_strength_delta": round(query_strength_delta, 6),
        "weak_query_groups_delta": int(weak_group_delta),
        "distractor_rate_delta": round(distractor_delta, 6),
        "critical_fn_rate_delta": round(critical_fn_delta, 6),
        "admission_status_a": admission_a,
        "admission_status_b": admission_b,
    }
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "compare_dir": str(compare_root),
        "compare_summary": summary,
        "promotion_decision_summary": row,
    }
    return {"row": row, "snapshot": snapshot}


def write_query_bank_promotion_decision_outputs(*, compare_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    lib = _require_pandas()
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_query_bank_promotion_decision(compare_dir=compare_dir)
    table_csv = tables_dir / "table_query_bank_promotion_decision.csv"
    table_md = tables_dir / "table_query_bank_promotion_decision.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    df = lib.DataFrame([artifacts["row"]])
    df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Query Bank Promotion Decision\n\n" + df_to_markdown_table(df))
    _write_json(snapshot_json, artifacts["snapshot"])
    report_lines = [
        "# Query Bank Promotion Decision",
        "",
        f"- promotion_decision: `{artifacts['row']['promotion_decision']}`",
        f"- decision_confidence: `{artifacts['row']['decision_confidence']}`",
        f"- primary_basis: `{artifacts['row']['primary_basis']}`",
        f"- secondary_basis: `{artifacts['row']['secondary_basis']}`",
        f"- recommended_next_step: `{artifacts['row']['recommended_next_step']}`",
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
