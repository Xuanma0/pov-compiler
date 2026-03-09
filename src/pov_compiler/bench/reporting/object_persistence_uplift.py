from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency expected in runtime env.
    pd = None

from pov_compiler.bench.query_bank import QueryBank
from pov_compiler.bench.reporting.latex import df_to_markdown_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency expected in runtime env.
        raise ImportError("pandas is required for object-persistence reporting.")
    return pd


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


def _norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split()) if text else ""


def _extract_query_literal(query: str, field_name: str) -> str:
    match = re.search(rf"{re.escape(field_name)}=([^\s]+)", str(query or ""))
    if not match:
        return ""
    return _norm_label(match.group(1).replace("_", " "))


def _collect_object_labels(output_payload: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    perception = output_payload.get("perception", {})
    if isinstance(perception, dict):
        frames = perception.get("frames", [])
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                for obj in frame.get("objects", []) or []:
                    if not isinstance(obj, dict):
                        continue
                    label = _norm_label(obj.get("label", ""))
                    if label:
                        labels.add(label)
    for item in output_payload.get("object_memory_v0", []) or []:
        if not isinstance(item, dict):
            continue
        label = _norm_label(item.get("object_name", ""))
        if label:
            labels.add(label)
    for event in output_payload.get("events_v1", []) or []:
        if not isinstance(event, dict):
            continue
        label = _norm_label(event.get("interaction_primary_object", ""))
        if label:
            labels.add(label)
    return labels


def _collect_persistent_labels(output_payload: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    perception = output_payload.get("perception", {})
    if not isinstance(perception, dict):
        return labels
    for frame in perception.get("frames", []) or []:
        if not isinstance(frame, dict):
            continue
        for obj in frame.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            label = _norm_label(obj.get("label", ""))
            if not label:
                continue
            if bool(obj.get("persistent", False)) or int(_to_float(obj.get("persistence_count")) or 0) >= 2:
                labels.add(label)
    return labels


def _count_persistent_tracks(output_payload: dict[str, Any]) -> int:
    track_ids: set[str] = set()
    perception = output_payload.get("perception", {})
    if not isinstance(perception, dict):
        return 0
    for frame in perception.get("frames", []) or []:
        if not isinstance(frame, dict):
            continue
        for obj in frame.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            track_id = str(obj.get("track_id", "")).strip()
            if track_id and bool(obj.get("persistent", False)):
                track_ids.add(track_id)
    summary = perception.get("summary", {}) if isinstance(perception.get("summary"), dict) else {}
    count = int(_to_float(summary.get("persistent_tracks_total")) or len(track_ids))
    return max(count, len(track_ids))


def _query_target_label(query: Any) -> str:
    query_text = str(getattr(query, "query", "") or "")
    for field_name in ("persistent_object", "lost_object", "chain_object", "interaction_object", "object"):
        value = _extract_query_literal(query_text, field_name)
        if value:
            return value
    return ""


def _matches_label(target_label: str, labels: set[str]) -> bool:
    if not target_label:
        return bool(labels)
    for label in labels:
        if target_label in label or label in target_label:
            return True
    return False


def _query_support_for_object_persistence(
    *,
    query: Any,
    object_labels: set[str],
    persistent_labels: set[str],
    object_memory_items_total: int,
    persistent_tracks_total: int,
    has_scene_change: bool,
) -> bool:
    signal_tags = {_norm_label(tag) for tag in getattr(query, "signal_tags", [])}
    sensitive_to = {_norm_label(tag) for tag in getattr(query, "sensitive_to", [])}
    all_tags = signal_tags | sensitive_to | {_norm_label(getattr(query, "group", ""))}
    target_label = _query_target_label(query)

    if "object_persistence" in all_tags or getattr(query, "group", "") == "object_persistence":
        if persistent_tracks_total <= 0 or not persistent_labels:
            return False
        return _matches_label(target_label, persistent_labels)

    if "lost_object" in all_tags or "object_memory" in all_tags or getattr(query, "group", "") == "lost_object":
        if object_memory_items_total <= 0:
            return False
        return _matches_label(target_label, persistent_labels or object_labels)

    if "chain" in all_tags or getattr(query, "group", "") == "chain_support":
        if not has_scene_change or persistent_tracks_total <= 0:
            return False
        return _matches_label(target_label, persistent_labels or object_labels)

    return _matches_label(target_label, object_labels)


def _query_strength_summary(
    *,
    query_bank: QueryBank,
    output_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    object_labels = _collect_object_labels(output_payload)
    persistent_labels = _collect_persistent_labels(output_payload)
    object_memory_items_total = int(len(output_payload.get("object_memory_v0", []) or []))
    persistent_tracks_total = int(_count_persistent_tracks(output_payload))
    tokens = output_payload.get("token_codec", {}).get("tokens", [])
    has_scene_change = any(
        isinstance(token, dict) and str(token.get("type", "")).strip().upper() == "SCENE_CHANGE"
        for token in tokens or []
    )

    enabled_queries = [query for query in query_bank.queries if bool(query.enabled)]
    supported_queries = 0
    weak_query_groups_count = 0
    group_support: dict[str, float] = {}
    group_rows: list[dict[str, Any]] = []
    lost_total = 0
    lost_supported = 0
    persistence_total = 0
    persistence_supported = 0
    chain_total = 0
    chain_supported = 0
    for group in query_bank.groups:
        group_queries = [query for query in enabled_queries if query.group == group.group_id]
        if not group_queries:
            continue
        supported = 0
        for query in group_queries:
            is_supported = _query_support_for_object_persistence(
                query=query,
                object_labels=object_labels,
                persistent_labels=persistent_labels,
                object_memory_items_total=object_memory_items_total,
                persistent_tracks_total=persistent_tracks_total,
                has_scene_change=has_scene_change,
            )
            if is_supported:
                supported += 1
                supported_queries += 1
            query_tags = {_norm_label(tag) for tag in query.signal_tags}
            if "lost_object" in query_tags or "object_memory" in query_tags or query.group == "lost_object":
                lost_total += 1
                if is_supported:
                    lost_supported += 1
            if "object_persistence" in query_tags or query.group == "object_persistence":
                persistence_total += 1
                if is_supported:
                    persistence_supported += 1
            if "chain" in query_tags or query.group == "chain_support":
                chain_total += 1
                if is_supported:
                    chain_supported += 1
        support_rate = float(supported / len(group_queries))
        group_support[group.group_id] = support_rate
        if support_rate < 0.5:
            weak_query_groups_count += 1
        group_rows.append(
            {
                "query_group": str(group.group_id),
                "queries_total": int(len(group_queries)),
                "supported_queries": int(supported),
                "support_rate": float(support_rate),
            }
        )

    coverage_rate = float(supported_queries / len(enabled_queries)) if enabled_queries else 0.0
    return (
        {
            "coverage_rate": float(coverage_rate),
            "weak_query_groups_count": int(weak_query_groups_count),
            "lost_object_query_support_rate": float(lost_supported / lost_total) if lost_total else 0.0,
            "object_persistence_support_rate": float(persistence_supported / persistence_total) if persistence_total else 0.0,
            "chain_object_grounding_support_rate": float(chain_supported / chain_total) if chain_total else 0.0,
            "group_support": group_rows,
        },
        group_support,
    )


def _variant_recommendation(
    *,
    object_memory_items_total: int,
    object_persistence_support_rate: float,
    lost_object_query_support_rate: float,
    chain_object_grounding_support_rate: float,
    query_strength_coverage_rate: float,
) -> str:
    if object_persistence_support_rate >= 0.70 and lost_object_query_support_rate >= 0.60 and chain_object_grounding_support_rate >= 0.60:
        if query_strength_coverage_rate >= 0.60:
            return "promote_sam3_to_next_stage"
        return "need_retrieval_side_fix"
    if object_persistence_support_rate >= 0.50 and object_memory_items_total <= 0:
        return "need_more_object_memory_logic"
    if object_persistence_support_rate <= 0.25 and lost_object_query_support_rate <= 0.25:
        return "keep_yolo26n_only"
    return "signal_gain_insufficient"


def compute_variant_object_persistence_metrics(
    *,
    variant_label: str,
    output_payload: dict[str, Any] | None = None,
    query_bank: QueryBank,
    source_mode: str,
    metrics_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = output_payload if isinstance(output_payload, dict) else {}
    override = dict(metrics_override or {})
    perception = payload.get("perception", {})
    perception_meta = perception.get("meta", {}) if isinstance(perception, dict) else {}
    perception_summary = perception.get("summary", {}) if isinstance(perception, dict) else {}
    object_labels = _collect_object_labels(payload)
    persistent_labels = _collect_persistent_labels(payload)
    persistent_tracks_total = int(
        _to_float(override.get("persistent_tracks_total"))
        or _to_float(perception_summary.get("persistent_tracks_total"))
        or _count_persistent_tracks(payload)
    )
    query_strength_summary, group_support = _query_strength_summary(query_bank=query_bank, output_payload=payload)

    object_detections_total = int(
        _to_float(override.get("object_detections_total"))
        or _to_float(perception_summary.get("objects_total"))
        or sum(len(frame.get("objects", [])) for frame in perception.get("frames", []) if isinstance(frame, dict))
    )
    object_vocab_size = int(_to_float(override.get("object_vocab_size")) or len(object_labels))
    object_memory_items_total = int(
        _to_float(override.get("object_memory_items_total")) or len(payload.get("object_memory_v0", []) or [])
    )
    query_strength_override = override.get("query_strength", {})
    if not isinstance(query_strength_override, dict):
        query_strength_override = {}
    object_persistence_support_rate = float(
        _to_float(override.get("object_persistence_support_rate"))
        or _to_float(query_strength_override.get("object_persistence_support_rate"))
        or query_strength_summary.get("object_persistence_support_rate", 0.0)
    )
    lost_object_query_support_rate = float(
        _to_float(override.get("lost_object_query_support_rate"))
        or _to_float(query_strength_override.get("lost_object_query_support_rate"))
        or query_strength_summary.get("lost_object_query_support_rate", 0.0)
    )
    chain_object_grounding_support_rate = float(
        _to_float(override.get("chain_object_grounding_support_rate"))
        or _to_float(query_strength_override.get("chain_object_grounding_support_rate"))
        or query_strength_summary.get("chain_object_grounding_support_rate", 0.0)
    )
    query_strength_coverage_rate = float(
        _to_float(override.get("query_strength_coverage_rate"))
        or _to_float(query_strength_override.get("coverage_rate"))
        or query_strength_summary.get("coverage_rate", 0.0)
    )
    weak_query_groups_count = int(
        _to_float(override.get("weak_query_groups_count"))
        or _to_float(query_strength_override.get("weak_query_groups_count"))
        or query_strength_summary.get("weak_query_groups_count", 0)
    )
    recommendation = str(
        override.get("delta_audit_main_recommendation", "") or override.get("main_recommendation", "")
    ).strip()
    if not recommendation:
        recommendation = _variant_recommendation(
            object_memory_items_total=object_memory_items_total,
            object_persistence_support_rate=object_persistence_support_rate,
            lost_object_query_support_rate=lost_object_query_support_rate,
            chain_object_grounding_support_rate=chain_object_grounding_support_rate,
            query_strength_coverage_rate=query_strength_coverage_rate,
        )

    return {
        "variant_label": str(variant_label),
        "source_mode": str(source_mode),
        "perception_backend": str(
            override.get("perception_backend")
            or perception_summary.get("perception_backend_used")
            or perception_meta.get("perception_backend_used")
            or perception_summary.get("backend")
            or perception_meta.get("backend")
            or ""
        ),
        "perception_model_name": str(
            override.get("perception_model_name")
            or perception_summary.get("perception_model_name")
            or perception_meta.get("perception_model_name")
            or ""
        ),
        "perception_model_path": str(
            override.get("perception_model_path")
            or perception_summary.get("perception_model_path")
            or perception_meta.get("perception_model_path")
            or ""
        ),
        "segmentation_backend_used": str(
            override.get("segmentation_backend_used")
            or perception_summary.get("segmentation_backend_used")
            or perception_meta.get("segmentation_backend_used")
            or ""
        ),
        "segmentation_model_name": str(
            override.get("segmentation_model_name")
            or perception_summary.get("segmentation_model_name")
            or perception_meta.get("segmentation_model_name")
            or ""
        ),
        "segmentation_model_path": str(
            override.get("segmentation_model_path")
            or perception_summary.get("segmentation_model_path")
            or perception_meta.get("segmentation_model_path")
            or ""
        ),
        "cache_used": bool(override.get("cache_used") if "cache_used" in override else perception_summary.get("cache_hit", False)),
        "object_detections_total": int(object_detections_total),
        "object_vocab_size": int(object_vocab_size),
        "object_memory_items_total": int(object_memory_items_total),
        "persistent_tracks_total": int(persistent_tracks_total),
        "persistent_object_labels_total": int(len(persistent_labels)),
        "object_persistence_support_rate": float(object_persistence_support_rate),
        "lost_object_query_support_rate": float(lost_object_query_support_rate),
        "chain_object_grounding_support_rate": float(chain_object_grounding_support_rate),
        "query_strength_coverage_rate": float(query_strength_coverage_rate),
        "weak_query_groups_count": int(weak_query_groups_count),
        "delta_audit_main_recommendation": recommendation,
        "query_group_support": [
            {"query_group": key, "support_rate": float(value)}
            for key, value in sorted(group_support.items())
        ],
    }


def _object_persistence_status(
    *,
    baseline_metrics: dict[str, Any],
    uplift_metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> str:
    gate = dict(thresholds or {})
    persistence_delta = float(uplift_metrics.get("object_persistence_support_rate", 0.0)) - float(
        baseline_metrics.get("object_persistence_support_rate", 0.0)
    )
    lost_delta = float(uplift_metrics.get("lost_object_query_support_rate", 0.0)) - float(
        baseline_metrics.get("lost_object_query_support_rate", 0.0)
    )
    chain_delta = float(uplift_metrics.get("chain_object_grounding_support_rate", 0.0)) - float(
        baseline_metrics.get("chain_object_grounding_support_rate", 0.0)
    )
    query_strength_delta = float(uplift_metrics.get("query_strength_coverage_rate", 0.0)) - float(
        baseline_metrics.get("query_strength_coverage_rate", 0.0)
    )
    min_persistence_delta = float(_to_float(gate.get("object_persistence_support_delta_min")) or 0.15)
    min_lost_delta = float(_to_float(gate.get("lost_object_support_delta_min")) or 0.10)
    min_chain_delta = float(_to_float(gate.get("chain_support_delta_min")) or 0.10)
    min_query_strength_delta = float(_to_float(gate.get("query_strength_delta_min")) or 0.10)
    if (
        persistence_delta >= min_persistence_delta
        or lost_delta >= min_lost_delta
        or chain_delta >= min_chain_delta
        or query_strength_delta >= min_query_strength_delta
    ):
        return "improved"
    if persistence_delta < -0.05 or lost_delta < -0.05 or chain_delta < -0.05 or query_strength_delta < -0.05:
        return "regressed"
    return "no_change"


def _next_action_recommendation(
    *,
    baseline_metrics: dict[str, Any],
    uplift_metrics: dict[str, Any],
    object_persistence_status: str,
) -> str:
    object_memory_delta = int(uplift_metrics.get("object_memory_items_total", 0)) - int(
        baseline_metrics.get("object_memory_items_total", 0)
    )
    persistence_delta = float(uplift_metrics.get("object_persistence_support_rate", 0.0)) - float(
        baseline_metrics.get("object_persistence_support_rate", 0.0)
    )
    lost_delta = float(uplift_metrics.get("lost_object_query_support_rate", 0.0)) - float(
        baseline_metrics.get("lost_object_query_support_rate", 0.0)
    )
    chain_delta = float(uplift_metrics.get("chain_object_grounding_support_rate", 0.0)) - float(
        baseline_metrics.get("chain_object_grounding_support_rate", 0.0)
    )
    query_strength_delta = float(uplift_metrics.get("query_strength_coverage_rate", 0.0)) - float(
        baseline_metrics.get("query_strength_coverage_rate", 0.0)
    )
    if (
        object_persistence_status == "improved"
        and persistence_delta >= 0.20
        and lost_delta >= 0.15
        and chain_delta >= 0.15
        and query_strength_delta >= 0.10
    ):
        return "promote_sam3_to_next_stage"
    if object_persistence_status == "improved" and object_memory_delta <= 0 and persistence_delta >= 0.15:
        return "need_more_object_memory_logic"
    if object_persistence_status == "improved" and (persistence_delta > 0.0 or chain_delta > 0.0) and query_strength_delta <= 0.0:
        return "need_retrieval_side_fix"
    if object_persistence_status in {"regressed", "no_change"}:
        return "keep_yolo26n_only"
    return "signal_gain_insufficient"


def write_object_persistence_compare_outputs(
    *,
    out_dir: str | Path,
    suite_id: str,
    query_bank: QueryBank,
    baseline_metrics: dict[str, Any],
    uplift_metrics: dict[str, Any],
    commands: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    lib = _require_pandas()
    out_root = Path(out_dir).resolve()
    compare_dir = out_root / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    status = _object_persistence_status(
        baseline_metrics=baseline_metrics,
        uplift_metrics=uplift_metrics,
        thresholds=thresholds,
    )
    next_action = _next_action_recommendation(
        baseline_metrics=baseline_metrics,
        uplift_metrics=uplift_metrics,
        object_persistence_status=status,
    )
    delta_row = {
        "variant_label": "delta",
        "source_mode": "derived",
        "perception_backend": "",
        "perception_model_name": "",
        "perception_model_path": "",
        "segmentation_backend_used": "",
        "segmentation_model_name": "",
        "segmentation_model_path": "",
        "cache_used": "",
        "object_detections_total": int(uplift_metrics["object_detections_total"]) - int(baseline_metrics["object_detections_total"]),
        "object_vocab_size": int(uplift_metrics["object_vocab_size"]) - int(baseline_metrics["object_vocab_size"]),
        "object_memory_items_total": int(uplift_metrics["object_memory_items_total"]) - int(
            baseline_metrics["object_memory_items_total"]
        ),
        "persistent_tracks_total": int(uplift_metrics["persistent_tracks_total"]) - int(
            baseline_metrics["persistent_tracks_total"]
        ),
        "persistent_object_labels_total": int(uplift_metrics["persistent_object_labels_total"]) - int(
            baseline_metrics["persistent_object_labels_total"]
        ),
        "object_persistence_support_rate": round(
            float(uplift_metrics["object_persistence_support_rate"]) - float(baseline_metrics["object_persistence_support_rate"]),
            6,
        ),
        "lost_object_query_support_rate": round(
            float(uplift_metrics["lost_object_query_support_rate"]) - float(baseline_metrics["lost_object_query_support_rate"]),
            6,
        ),
        "chain_object_grounding_support_rate": round(
            float(uplift_metrics["chain_object_grounding_support_rate"]) - float(baseline_metrics["chain_object_grounding_support_rate"]),
            6,
        ),
        "query_strength_coverage_rate": round(
            float(uplift_metrics["query_strength_coverage_rate"]) - float(baseline_metrics["query_strength_coverage_rate"]),
            6,
        ),
        "weak_query_groups_count": int(uplift_metrics["weak_query_groups_count"]) - int(
            baseline_metrics["weak_query_groups_count"]
        ),
        "delta_audit_main_recommendation": (
            f"{baseline_metrics['delta_audit_main_recommendation']} -> {uplift_metrics['delta_audit_main_recommendation']}"
        ),
        "object_persistence_status": status,
    }
    rows = [
        {**baseline_metrics, "object_persistence_status": status},
        {**uplift_metrics, "object_persistence_status": status},
        delta_row,
    ]
    df = lib.DataFrame(rows)
    table_csv = tables_dir / "table_object_persistence_uplift.csv"
    table_md = tables_dir / "table_object_persistence_uplift.md"
    df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Object Persistence Uplift Compare\n\n" + df_to_markdown_table(df))

    figure_paths: list[str] = []
    variants = [baseline_metrics["variant_label"], uplift_metrics["variant_label"]]
    x = list(range(len(variants)))

    fig_delta_base = figures_dir / "fig_object_persistence_uplift_delta"
    plt.figure(figsize=(8.8, 4.8))
    plt.bar(
        ["object_memory_delta", "persistence_delta", "lost_object_delta", "chain_support_delta", "query_strength_delta"],
        [
            delta_row["object_memory_items_total"],
            delta_row["object_persistence_support_rate"],
            delta_row["lost_object_query_support_rate"],
            delta_row["chain_object_grounding_support_rate"],
            delta_row["query_strength_coverage_rate"],
        ],
    )
    plt.axhline(y=0.0, linewidth=1.0)
    plt.title("Object Persistence Uplift Delta")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_delta_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    fig_object_memory_base = figures_dir / "fig_object_persistence_object_memory"
    plt.figure(figsize=(8.8, 4.8))
    plt.bar([item - 0.2 for item in x], [baseline_metrics["object_memory_items_total"], uplift_metrics["object_memory_items_total"]], width=0.2, label="object_memory_items_total")
    plt.bar([item for item in x], [baseline_metrics["persistent_tracks_total"], uplift_metrics["persistent_tracks_total"]], width=0.2, label="persistent_tracks_total")
    plt.bar([item + 0.2 for item in x], [baseline_metrics["object_persistence_support_rate"], uplift_metrics["object_persistence_support_rate"]], width=0.2, label="object_persistence_support_rate")
    plt.xticks(x, variants)
    plt.title("Object Memory / Persistence")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_object_memory_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    fig_chain_base = figures_dir / "fig_object_persistence_chain_support"
    plt.figure(figsize=(8.8, 4.8))
    plt.bar([item - 0.15 for item in x], [baseline_metrics["chain_object_grounding_support_rate"], uplift_metrics["chain_object_grounding_support_rate"]], width=0.3, label="chain_object_grounding_support_rate")
    plt.bar([item + 0.15 for item in x], [baseline_metrics["query_strength_coverage_rate"], uplift_metrics["query_strength_coverage_rate"]], width=0.3, label="query_strength_coverage_rate")
    plt.xticks(x, variants)
    plt.title("Chain Support / Query Strength")
    plt.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        target = fig_chain_base.with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    compare_summary = {
        "suite_id": str(suite_id),
        "query_bank_id": query_bank.query_bank_id,
        "query_bank_version": query_bank.query_bank_version,
        "query_bank_hash": query_bank.query_bank_hash,
        "baseline": baseline_metrics,
        "uplift": uplift_metrics,
        "object_persistence_status": status,
        "next_action_recommendation": next_action,
    }
    if extra_summary:
        compare_summary.update(extra_summary)
    compare_summary_path = compare_dir / "compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    commands_path = compare_dir / "commands.sh"
    _write_text(commands_path, "\n".join(commands or []))
    readme_path = compare_dir / "README.md"
    _write_text(
        readme_path,
        "\n".join(
            [
                "# Object Persistence Pilot",
                "",
                f"- suite_id: `{suite_id}`",
                f"- query_bank_id: `{query_bank.query_bank_id}`",
                f"- object_persistence_status: `{status}`",
                f"- next_action_recommendation: `{next_action}`",
                f"- baseline_backend: `{baseline_metrics.get('perception_backend', '')}`",
                f"- baseline_model: `{baseline_metrics.get('perception_model_name', '')}`",
                f"- baseline_segmentation: `{baseline_metrics.get('segmentation_backend_used', '')}`",
                f"- uplift_backend: `{uplift_metrics.get('perception_backend', '')}`",
                f"- uplift_model: `{uplift_metrics.get('perception_model_name', '')}`",
                f"- uplift_segmentation: `{uplift_metrics.get('segmentation_backend_used', '')}`",
                f"- cache_used: baseline=`{baseline_metrics.get('cache_used', False)}`, uplift=`{uplift_metrics.get('cache_used', False)}`",
                "",
                "## Files",
                "",
                f"- table_csv: `{table_csv}`",
                f"- table_md: `{table_md}`",
                f"- figures: `{figure_paths}`",
                f"- commands: `{commands_path}`",
            ]
        ),
    )
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_id": str(suite_id),
        "query_bank_id": query_bank.query_bank_id,
        "query_bank_hash": query_bank.query_bank_hash,
        "baseline": baseline_metrics,
        "uplift": uplift_metrics,
        "object_persistence_status": status,
        "next_action_recommendation": next_action,
        "thresholds": dict(thresholds or {}),
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
            "compare_summary_json": str(compare_summary_path),
            "commands_sh": str(commands_path),
            "readme_md": str(readme_path),
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "figure_paths": figure_paths,
        "compare_summary_json": compare_summary_path,
        "snapshot_json": snapshot_path,
        "object_persistence_status": status,
        "next_action_recommendation": next_action,
    }


def build_object_persistence_report(
    *,
    suite_dir: str | Path,
    baseline_dir: str | Path,
) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    baseline_root = Path(baseline_dir).resolve()
    suite_summary = _read_json(suite_root / "compare" / "compare_summary.json")
    baseline_summary = _read_json(baseline_root / "compare" / "compare_summary.json")
    suite_uplift = suite_summary.get("uplift", {}) if isinstance(suite_summary.get("uplift"), dict) else {}
    suite_baseline = suite_summary.get("baseline", {}) if isinstance(suite_summary.get("baseline"), dict) else {}
    baseline_uplift = baseline_summary.get("uplift", {}) if isinstance(baseline_summary.get("uplift"), dict) else {}

    object_memory_improved = bool(int(suite_uplift.get("object_memory_items_total", 0)) > int(suite_baseline.get("object_memory_items_total", 0)))
    object_persistence_improved = bool(float(suite_uplift.get("object_persistence_support_rate", 0.0)) > float(suite_baseline.get("object_persistence_support_rate", 0.0)))
    lost_object_support_improved = bool(float(suite_uplift.get("lost_object_query_support_rate", 0.0)) > float(suite_baseline.get("lost_object_query_support_rate", 0.0)))
    chain_object_grounding_improved = bool(float(suite_uplift.get("chain_object_grounding_support_rate", 0.0)) > float(suite_baseline.get("chain_object_grounding_support_rate", 0.0)))
    query_strength_improved = bool(float(suite_uplift.get("query_strength_coverage_rate", 0.0)) > float(suite_baseline.get("query_strength_coverage_rate", 0.0)) or int(suite_uplift.get("weak_query_groups_count", 0)) < int(suite_baseline.get("weak_query_groups_count", 0)))
    persistence_gain_vs_fake = round(
        (float(suite_uplift.get("object_persistence_support_rate", 0.0)) - float(suite_baseline.get("object_persistence_support_rate", 0.0)))
        - (float(baseline_uplift.get("object_persistence_support_rate", 0.0)) - float(baseline_summary.get("baseline", {}).get("object_persistence_support_rate", 0.0))),
        6,
    )
    next_action = str(suite_summary.get("next_action_recommendation", "")).strip() or "signal_gain_insufficient"
    if object_persistence_improved and lost_object_support_improved and chain_object_grounding_improved and query_strength_improved:
        next_action = "promote_sam3_to_next_stage"
    elif object_persistence_improved and not object_memory_improved:
        next_action = "need_more_object_memory_logic"
    elif object_persistence_improved and not query_strength_improved:
        next_action = "need_retrieval_side_fix"
    elif not object_persistence_improved and not lost_object_support_improved and not chain_object_grounding_improved:
        next_action = "keep_yolo26n_only"
    elif persistence_gain_vs_fake <= 0.0:
        next_action = "signal_gain_insufficient"

    rows = [
        {
            "suite_id": str(suite_summary.get("suite_id", suite_root.name)),
            "baseline_suite_id": str(baseline_summary.get("suite_id", baseline_root.name)),
            "baseline_query_bank_id": str(baseline_summary.get("query_bank_id", "")),
            "candidate_query_bank_id": str(suite_summary.get("query_bank_id", "")),
            "object_persistence_status": str(suite_summary.get("object_persistence_status", "no_change")),
            "object_memory_improved": object_memory_improved,
            "object_persistence_improved": object_persistence_improved,
            "lost_object_support_improved": lost_object_support_improved,
            "chain_object_grounding_improved": chain_object_grounding_improved,
            "query_strength_improved": query_strength_improved,
            "persistence_gain_vs_fake": persistence_gain_vs_fake,
            "next_action_recommendation": next_action,
            "should_formalize_sam3_next": next_action == "promote_sam3_to_next_stage",
            "should_abandon_sam3_route": next_action in {"keep_yolo26n_only", "signal_gain_insufficient"},
        }
    ]
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "baseline_dir": str(baseline_root),
        "suite_id": str(suite_summary.get("suite_id", suite_root.name)),
        "baseline_suite_id": str(baseline_summary.get("suite_id", baseline_root.name)),
        "baseline_query_bank_id": str(baseline_summary.get("query_bank_id", "")),
        "candidate_query_bank_id": str(suite_summary.get("query_bank_id", "")),
        "object_persistence_status": str(suite_summary.get("object_persistence_status", "no_change")),
        "object_memory_improved": object_memory_improved,
        "object_persistence_improved": object_persistence_improved,
        "lost_object_support_improved": lost_object_support_improved,
        "chain_object_grounding_improved": chain_object_grounding_improved,
        "query_strength_improved": query_strength_improved,
        "persistence_gain_vs_fake": persistence_gain_vs_fake,
        "next_action_recommendation": next_action,
        "should_formalize_sam3_next": next_action == "promote_sam3_to_next_stage",
        "should_abandon_sam3_route": next_action in {"keep_yolo26n_only", "signal_gain_insufficient"},
        "suite_compare_summary": suite_summary,
        "baseline_compare_summary": baseline_summary,
    }
    return lib.DataFrame(rows), snapshot


def write_object_persistence_report_outputs(
    *,
    suite_dir: str | Path,
    baseline_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df, snapshot = build_object_persistence_report(suite_dir=suite_dir, baseline_dir=baseline_dir)
    table_csv = tables_dir / "table_object_persistence_uplift_summary.csv"
    table_md = tables_dir / "table_object_persistence_uplift_summary.md"
    alias_table_csv = tables_dir / "table_signal_uplift_summary.csv"
    alias_table_md = tables_dir / "table_signal_uplift_summary.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    df.to_csv(table_csv, index=False)
    df.to_csv(alias_table_csv, index=False)
    table_markdown = "# Object Persistence Uplift Summary\n\n" + df_to_markdown_table(df)
    _write_text(table_md, table_markdown)
    _write_text(alias_table_md, table_markdown)

    fig_base = figures_dir / "fig_object_persistence_uplift_summary"
    alias_fig_base = figures_dir / "fig_signal_uplift_summary"
    plt.figure(figsize=(8.2, 4.6))
    plt.bar(
        ["object_memory", "object_persistence", "lost_object_support", "chain_grounding", "query_strength"],
        [
            1.0 if bool(df.loc[0, "object_memory_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "object_persistence_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "lost_object_support_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "chain_object_grounding_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "query_strength_improved"]) else 0.0,
        ],
    )
    plt.ylim(0.0, 1.0)
    plt.title("Object Persistence Uplift Summary")
    plt.tight_layout()
    figure_paths: list[str] = []
    for ext in ("png", "pdf"):
        target = fig_base.with_suffix(f".{ext}")
        alias_target = alias_fig_base.with_suffix(f".{ext}")
        plt.savefig(target)
        plt.savefig(alias_target)
        figure_paths.append(str(target))
        figure_paths.append(str(alias_target))
    plt.close()

    report_lines = [
        "# Object Persistence Uplift Report",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- baseline_dir: `{Path(baseline_dir).resolve()}`",
        f"- object_persistence_status: `{snapshot.get('object_persistence_status', 'no_change')}`",
        f"- object_memory_improved: `{snapshot.get('object_memory_improved', False)}`",
        f"- object_persistence_improved: `{snapshot.get('object_persistence_improved', False)}`",
        f"- lost_object_support_improved: `{snapshot.get('lost_object_support_improved', False)}`",
        f"- chain_object_grounding_improved: `{snapshot.get('chain_object_grounding_improved', False)}`",
        f"- query_strength_improved: `{snapshot.get('query_strength_improved', False)}`",
        f"- persistence_gain_vs_fake: `{snapshot.get('persistence_gain_vs_fake', 0.0)}`",
        f"- next_action_recommendation: `{snapshot.get('next_action_recommendation', '')}`",
        f"- should_formalize_sam3_next: `{snapshot.get('should_formalize_sam3_next', False)}`",
        f"- should_abandon_sam3_route: `{snapshot.get('should_abandon_sam3_route', False)}`",
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
        "figure_paths": figure_paths,
        "next_action_recommendation": snapshot.get("next_action_recommendation", ""),
        "should_formalize_sam3_next": bool(snapshot.get("should_formalize_sam3_next", False)),
        "object_persistence_status": snapshot.get("object_persistence_status", "no_change"),
    }
