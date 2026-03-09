from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from pov_compiler.bench.query_bank import QueryBank
from pov_compiler.bench.reporting.latex import df_to_markdown_table


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for persistent object-memory uplift reporting.")
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
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split()) if text else ""


def _extract_query_literal(query: str, field_name: str) -> str:
    match = re.search(rf"{re.escape(field_name)}=([^\s]+)", str(query or ""))
    if not match:
        return ""
    return _norm_label(match.group(1))


def _read_memory_items(output_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in output_payload.get("object_memory_v0", []) or []:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
        rows.append(
            {
                "object_name": _norm_label(item.get("object_name", "")),
                "last_seen_t_ms": int(_to_float(item.get("last_seen_t_ms")) or 0),
                "last_contact_t_ms": int(_to_float(item.get("last_contact_t_ms")) or 0)
                if _to_float(item.get("last_contact_t_ms")) is not None
                else None,
                "last_place_id": str(item.get("last_place_id", "") or ""),
                "evidence_event_ids": [str(x) for x in (item.get("evidence_event_ids", []) or []) if str(x).strip()],
                "score": float(_to_float(item.get("score")) or 0.0),
                "logic_variant": str(meta.get("logic_variant", "") or ""),
                "persistence_backed": bool(meta.get("persistence_backed", False)),
                "persistence_frame_count": int(_to_float(meta.get("persistence_frame_count")) or 0),
                "persistence_score_max": float(_to_float(meta.get("persistence_score_max")) or 0.0),
                "memory_tier": str(meta.get("memory_tier", "") or ""),
                "short_term_score": float(_to_float(meta.get("short_term_score")) or 0.0),
                "long_term_score": float(_to_float(meta.get("long_term_score")) or 0.0),
                "reappearance_score": float(_to_float(meta.get("reappearance_score")) or 0.0),
                "reappearance_count": int(_to_float(meta.get("reappearance_count")) or 0),
                "persistence_confidence": float(_to_float(meta.get("persistence_confidence")) or 0.0),
                "last_tracked_t_ms": int(_to_float(meta.get("last_tracked_t_ms")) or 0)
                if _to_float(meta.get("last_tracked_t_ms")) is not None
                else None,
                "last_interacted_t_ms": int(_to_float(meta.get("last_interacted_t_ms")) or 0)
                if _to_float(meta.get("last_interacted_t_ms")) is not None
                else None,
                "seen_count": int(_to_float(meta.get("seen_count")) or 0),
                "contact_count": int(_to_float(meta.get("contact_count")) or 0),
                "occlusion_gap_max_ms": int(_to_float(meta.get("occlusion_gap_max_ms")) or 0),
            }
        )
    return rows


def _has_scene_change(output_payload: dict[str, Any]) -> bool:
    tokens = output_payload.get("token_codec", {}).get("tokens", [])
    return any(
        isinstance(token, dict) and str(token.get("type", "")).strip().upper() == "SCENE_CHANGE"
        for token in tokens or []
    )


def _query_target_label(query: Any) -> str:
    query_text = str(getattr(query, "query", "") or "")
    for field_name in ("lost_object", "chain_object", "persistent_object", "object", "interaction_object"):
        value = _extract_query_literal(query_text, field_name)
        if value:
            return value
    return ""


def _matches_label(target_label: str, object_name: str) -> bool:
    if not target_label:
        return bool(object_name)
    if not object_name:
        return False
    return target_label in object_name or object_name in target_label


def _supported_memory_items(*, query: Any, memory_items: list[dict[str, Any]], has_scene_change: bool) -> bool:
    target_label = _query_target_label(query)
    matching = [item for item in memory_items if _matches_label(target_label, str(item.get("object_name", "")))]
    if not matching:
        return False

    signal_tags = {_norm_label(tag) for tag in getattr(query, "signal_tags", [])}
    sensitive_to = {_norm_label(tag) for tag in getattr(query, "sensitive_to", [])}
    group_id = _norm_label(getattr(query, "group", ""))
    all_tags = signal_tags | sensitive_to | {group_id}

    if group_id in {"persistent_recall", "object_memory_recall"} or "object_memory" in all_tags:
        return any(
            bool(item.get("persistence_backed"))
            and (
                float(item.get("short_term_score", 0.0)) >= 0.40
                or float(item.get("long_term_score", 0.0)) >= 0.45
                or float(item.get("persistence_confidence", 0.0)) >= 0.50
            )
            for item in matching
        )
    if group_id == "lost_object" or "lost_object" in all_tags:
        return any(
            str(item.get("memory_tier", "")) == "long_term"
            or float(item.get("reappearance_score", 0.0)) >= 0.25
            or item.get("last_interacted_t_ms") is not None
            for item in matching
        )
    if group_id == "chain_support" or "chain" in all_tags:
        return has_scene_change and any(
            bool(item.get("evidence_event_ids"))
            and (
                str(item.get("memory_tier", "")) == "long_term"
                or float(item.get("long_term_score", 0.0)) >= 0.55
            )
            for item in matching
        )
    if group_id == "reappearance" or "reappearance" in all_tags:
        return any(
            int(item.get("reappearance_count", 0)) >= 1
            or float(item.get("reappearance_score", 0.0)) >= 0.25
            or float(item.get("long_term_score", 0.0)) >= 0.60
            for item in matching
        )
    if "last_tracked" in all_tags:
        return any(item.get("last_tracked_t_ms") is not None for item in matching)
    if "last_interacted" in all_tags:
        return any(item.get("last_interacted_t_ms") is not None for item in matching)
    return bool(matching)


def _query_strength_summary(*, query_bank: QueryBank, output_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    memory_items = _read_memory_items(output_payload)
    has_scene_change = _has_scene_change(output_payload)
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
    reappearance_total = 0
    reappearance_supported = 0

    for group in query_bank.groups:
        group_queries = [query for query in enabled_queries if query.group == group.group_id]
        if not group_queries:
            continue
        supported = 0
        for query in group_queries:
            is_supported = _supported_memory_items(query=query, memory_items=memory_items, has_scene_change=has_scene_change)
            if is_supported:
                supported += 1
                supported_queries += 1
            query_tags = {_norm_label(tag) for tag in query.signal_tags}
            if "lost_object" in query_tags or query.group == "lost_object":
                lost_total += 1
                if is_supported:
                    lost_supported += 1
            if "object_memory" in query_tags or query.group in {"object_memory_recall", "persistent_recall"}:
                persistence_total += 1
                if is_supported:
                    persistence_supported += 1
            if "chain" in query_tags or query.group == "chain_support":
                chain_total += 1
                if is_supported:
                    chain_supported += 1
            if "reappearance" in query_tags or query.group == "reappearance":
                reappearance_total += 1
                if is_supported:
                    reappearance_supported += 1
        support_rate = float(supported / len(group_queries))
        group_support[group.group_id] = support_rate
        if support_rate < 0.5:
            weak_query_groups_count += 1
        group_rows.append({"query_group": str(group.group_id), "queries_total": int(len(group_queries)), "supported_queries": int(supported), "support_rate": float(support_rate)})

    coverage_rate = float(supported_queries / len(enabled_queries)) if enabled_queries else 0.0
    return (
        {
            "coverage_rate": float(coverage_rate),
            "weak_query_groups_count": int(weak_query_groups_count),
            "lost_object_query_support_rate": float(lost_supported / lost_total) if lost_total else 0.0,
            "object_persistence_support_rate": float(persistence_supported / persistence_total) if persistence_total else 0.0,
            "chain_object_grounding_support_rate": float(chain_supported / chain_total) if chain_total else 0.0,
            "reappearance_support_rate": float(reappearance_supported / reappearance_total) if reappearance_total else 0.0,
            "group_support": group_rows,
        },
        group_support,
    )


def _variant_recommendation(
    *,
    object_memory_items_total: int,
    object_memory_long_term_items_total: int,
    reappearance_support_rate: float,
    lost_object_query_support_rate: float,
    chain_object_grounding_support_rate: float,
    query_strength_coverage_rate: float,
) -> str:
    if (
        object_memory_long_term_items_total >= 2
        and reappearance_support_rate >= 0.60
        and lost_object_query_support_rate >= 0.60
        and chain_object_grounding_support_rate >= 0.60
        and query_strength_coverage_rate >= 0.70
    ):
        return "promote_persistent_object_memory"
    if object_memory_long_term_items_total > 0 and (lost_object_query_support_rate < 0.60 or chain_object_grounding_support_rate < 0.60):
        return "keep_current_memory_logic"
    if object_memory_items_total > 0 and query_strength_coverage_rate >= 0.50:
        return "need_retrieval_side_fix"
    if object_memory_items_total > 0:
        return "need_decision_side_fix"
    return "persistent_memory_gain_insufficient"


def compute_variant_persistent_object_memory_metrics(
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
    memory_items = _read_memory_items(payload)
    query_strength_summary, group_support = _query_strength_summary(query_bank=query_bank, output_payload=payload)

    query_strength_override = override.get("query_strength", {})
    if not isinstance(query_strength_override, dict):
        query_strength_override = {}

    object_detections_total = int(
        _to_float(override.get("object_detections_total"))
        or _to_float(perception_summary.get("objects_total"))
        or sum(len(frame.get("objects", [])) for frame in perception.get("frames", []) if isinstance(frame, dict))
    )
    object_vocab_size = int(
        _to_float(override.get("object_vocab_size"))
        or len({_norm_label(item.get("object_name", "")) for item in memory_items if _norm_label(item.get("object_name", ""))})
    )
    object_memory_items_total = int(_to_float(override.get("object_memory_items_total")) or len(memory_items))
    object_memory_persistence_items_total = int(
        _to_float(override.get("object_memory_persistence_items_total"))
        or sum(1 for item in memory_items if bool(item.get("persistence_backed")))
    )
    object_memory_short_term_items_total = int(
        _to_float(override.get("object_memory_short_term_items_total"))
        or sum(1 for item in memory_items if str(item.get("memory_tier", "")) == "short_term")
    )
    object_memory_long_term_items_total = int(
        _to_float(override.get("object_memory_long_term_items_total"))
        or sum(1 for item in memory_items if str(item.get("memory_tier", "")) == "long_term")
    )
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
    reappearance_support_rate = float(
        _to_float(override.get("reappearance_support_rate"))
        or _to_float(query_strength_override.get("reappearance_support_rate"))
        or query_strength_summary.get("reappearance_support_rate", 0.0)
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

    recommendation = str(override.get("delta_audit_main_recommendation", "") or override.get("main_recommendation", "")).strip()
    if not recommendation:
        recommendation = _variant_recommendation(
            object_memory_items_total=object_memory_items_total,
            object_memory_long_term_items_total=object_memory_long_term_items_total,
            reappearance_support_rate=reappearance_support_rate,
            lost_object_query_support_rate=lost_object_query_support_rate,
            chain_object_grounding_support_rate=chain_object_grounding_support_rate,
            query_strength_coverage_rate=query_strength_coverage_rate,
        )

    logic_variant = str(
        override.get("object_memory_logic_variant")
        or payload.get("meta", {}).get("object_memory_logic_variant", "")
        or (memory_items[0].get("logic_variant", "") if memory_items else "")
    )

    return {
        "variant_label": str(variant_label),
        "source_mode": str(source_mode),
        "perception_backend": str(
            override.get("perception_backend")
            or perception_summary.get("perception_backend_used")
            or perception_meta.get("perception_backend_used")
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
        "object_memory_logic_variant": logic_variant,
        "object_detections_total": int(object_detections_total),
        "object_vocab_size": int(object_vocab_size),
        "object_memory_items_total": int(object_memory_items_total),
        "object_memory_persistence_items_total": int(object_memory_persistence_items_total),
        "object_memory_short_term_items_total": int(object_memory_short_term_items_total),
        "object_memory_long_term_items_total": int(object_memory_long_term_items_total),
        "object_persistence_support_rate": float(object_persistence_support_rate),
        "reappearance_support_rate": float(reappearance_support_rate),
        "lost_object_query_support_rate": float(lost_object_query_support_rate),
        "chain_object_grounding_support_rate": float(chain_object_grounding_support_rate),
        "query_strength_coverage_rate": float(query_strength_coverage_rate),
        "weak_query_groups_count": int(weak_query_groups_count),
        "delta_audit_main_recommendation": recommendation,
        "query_group_support": [{"query_group": key, "support_rate": float(value)} for key, value in sorted(group_support.items())],
    }


def _persistent_status(*, baseline_metrics: dict[str, Any], uplift_metrics: dict[str, Any]) -> str:
    if any(
        float(uplift_metrics.get(field, 0.0)) < float(baseline_metrics.get(field, 0.0)) - 0.10
        for field in (
            "reappearance_support_rate",
            "lost_object_query_support_rate",
            "chain_object_grounding_support_rate",
            "query_strength_coverage_rate",
        )
    ):
        return "regressed"
    if (
        int(uplift_metrics.get("object_memory_long_term_items_total", 0)) > int(baseline_metrics.get("object_memory_long_term_items_total", 0))
        or float(uplift_metrics.get("reappearance_support_rate", 0.0)) > float(baseline_metrics.get("reappearance_support_rate", 0.0))
        or float(uplift_metrics.get("lost_object_query_support_rate", 0.0)) > float(baseline_metrics.get("lost_object_query_support_rate", 0.0))
        or float(uplift_metrics.get("chain_object_grounding_support_rate", 0.0)) > float(baseline_metrics.get("chain_object_grounding_support_rate", 0.0))
        or float(uplift_metrics.get("query_strength_coverage_rate", 0.0)) > float(baseline_metrics.get("query_strength_coverage_rate", 0.0))
        or int(uplift_metrics.get("weak_query_groups_count", 0)) < int(baseline_metrics.get("weak_query_groups_count", 0))
    ):
        return "improved"
    return "no_change"


def _next_action_recommendation(*, status: str, baseline_metrics: dict[str, Any], uplift_metrics: dict[str, Any]) -> str:
    long_term_delta = int(uplift_metrics.get("object_memory_long_term_items_total", 0)) - int(
        baseline_metrics.get("object_memory_long_term_items_total", 0)
    )
    reappearance_delta = float(uplift_metrics.get("reappearance_support_rate", 0.0)) - float(
        baseline_metrics.get("reappearance_support_rate", 0.0)
    )
    lost_delta = float(uplift_metrics.get("lost_object_query_support_rate", 0.0)) - float(
        baseline_metrics.get("lost_object_query_support_rate", 0.0)
    )
    chain_delta = float(uplift_metrics.get("chain_object_grounding_support_rate", 0.0)) - float(
        baseline_metrics.get("chain_object_grounding_support_rate", 0.0)
    )
    query_delta = float(uplift_metrics.get("query_strength_coverage_rate", 0.0)) - float(
        baseline_metrics.get("query_strength_coverage_rate", 0.0)
    )

    if status == "improved" and long_term_delta > 0 and reappearance_delta > 0 and lost_delta > 0 and chain_delta > 0 and query_delta > 0:
        return "promote_persistent_object_memory"
    if status == "improved" and long_term_delta > 0 and reappearance_delta > 0 and (lost_delta <= 0 or chain_delta <= 0):
        return "keep_current_memory_logic"
    if status == "improved" and query_delta > 0 and chain_delta <= 0:
        return "need_retrieval_side_fix"
    if status == "improved" and chain_delta > 0 and lost_delta <= 0:
        return "need_decision_side_fix"
    if status == "no_change":
        return "keep_current_memory_logic"
    return "persistent_memory_gain_insufficient"


def write_persistent_object_memory_compare_outputs(
    *,
    out_dir: str | Path,
    suite_id: str,
    query_bank: QueryBank,
    baseline_metrics: dict[str, Any],
    uplift_metrics: dict[str, Any],
    commands: list[str],
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

    status = _persistent_status(baseline_metrics=baseline_metrics, uplift_metrics=uplift_metrics)
    next_action = _next_action_recommendation(status=status, baseline_metrics=baseline_metrics, uplift_metrics=uplift_metrics)

    rows = [
        baseline_metrics,
        uplift_metrics,
        {
            "variant_label": "delta",
            "source_mode": "derived",
            "object_memory_items_total": int(uplift_metrics.get("object_memory_items_total", 0))
            - int(baseline_metrics.get("object_memory_items_total", 0)),
            "object_memory_persistence_items_total": int(uplift_metrics.get("object_memory_persistence_items_total", 0))
            - int(baseline_metrics.get("object_memory_persistence_items_total", 0)),
            "object_memory_short_term_items_total": int(uplift_metrics.get("object_memory_short_term_items_total", 0))
            - int(baseline_metrics.get("object_memory_short_term_items_total", 0)),
            "object_memory_long_term_items_total": int(uplift_metrics.get("object_memory_long_term_items_total", 0))
            - int(baseline_metrics.get("object_memory_long_term_items_total", 0)),
            "reappearance_support_rate": float(uplift_metrics.get("reappearance_support_rate", 0.0))
            - float(baseline_metrics.get("reappearance_support_rate", 0.0)),
            "lost_object_query_support_rate": float(uplift_metrics.get("lost_object_query_support_rate", 0.0))
            - float(baseline_metrics.get("lost_object_query_support_rate", 0.0)),
            "chain_object_grounding_support_rate": float(uplift_metrics.get("chain_object_grounding_support_rate", 0.0))
            - float(baseline_metrics.get("chain_object_grounding_support_rate", 0.0)),
            "object_persistence_support_rate": float(uplift_metrics.get("object_persistence_support_rate", 0.0))
            - float(baseline_metrics.get("object_persistence_support_rate", 0.0)),
            "query_strength_coverage_rate": float(uplift_metrics.get("query_strength_coverage_rate", 0.0))
            - float(baseline_metrics.get("query_strength_coverage_rate", 0.0)),
            "weak_query_groups_count": int(uplift_metrics.get("weak_query_groups_count", 0))
            - int(baseline_metrics.get("weak_query_groups_count", 0)),
            "persistent_object_memory_status": status,
            "delta_audit_main_recommendation": next_action,
        },
    ]
    df = lib.DataFrame(rows)
    table_csv = tables_dir / "table_persistent_object_memory_uplift.csv"
    table_md = tables_dir / "table_persistent_object_memory_uplift.md"
    df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Persistent Object Memory Uplift\n\n" + df_to_markdown_table(df))

    variants = [str(baseline_metrics.get("variant_label", "baseline")), str(uplift_metrics.get("variant_label", "uplift"))]
    x = [0, 1]
    figure_paths: list[str] = []

    def _save_current_figure(base: Path) -> None:
        for ext in ("png", "pdf"):
            target = base.with_suffix(f".{ext}")
            plt.savefig(target)
            figure_paths.append(str(target))
        plt.close()

    plt.figure(figsize=(8.8, 4.8))
    plt.bar([item - 0.18 for item in x], [baseline_metrics["object_memory_items_total"], uplift_metrics["object_memory_items_total"]], width=0.18, label="object_memory_items_total")
    plt.bar([item for item in x], [baseline_metrics["object_memory_long_term_items_total"], uplift_metrics["object_memory_long_term_items_total"]], width=0.18, label="long_term_items")
    plt.bar([item + 0.18 for item in x], [baseline_metrics["query_strength_coverage_rate"], uplift_metrics["query_strength_coverage_rate"]], width=0.18, label="query_strength")
    plt.xticks(x, variants)
    plt.title("Persistent Object Memory Delta")
    plt.legend()
    plt.tight_layout()
    _save_current_figure(figures_dir / "fig_persistent_object_memory_delta")

    plt.figure(figsize=(8.8, 4.8))
    plt.bar([item - 0.18 for item in x], [baseline_metrics["lost_object_query_support_rate"], uplift_metrics["lost_object_query_support_rate"]], width=0.18, label="lost_object_support")
    plt.bar([item for item in x], [baseline_metrics["object_memory_persistence_items_total"], uplift_metrics["object_memory_persistence_items_total"]], width=0.18, label="persistence_items")
    plt.bar([item + 0.18 for item in x], [baseline_metrics["reappearance_support_rate"], uplift_metrics["reappearance_support_rate"]], width=0.18, label="reappearance_support")
    plt.xticks(x, variants)
    plt.title("Persistent Memory Lost-Object Support")
    plt.legend()
    plt.tight_layout()
    _save_current_figure(figures_dir / "fig_persistent_object_memory_lost_object")

    plt.figure(figsize=(8.8, 4.8))
    plt.bar([item - 0.18 for item in x], [baseline_metrics["chain_object_grounding_support_rate"], uplift_metrics["chain_object_grounding_support_rate"]], width=0.18, label="chain_grounding")
    plt.bar([item for item in x], [baseline_metrics["object_persistence_support_rate"], uplift_metrics["object_persistence_support_rate"]], width=0.18, label="object_persistence")
    plt.bar([item + 0.18 for item in x], [baseline_metrics["weak_query_groups_count"], uplift_metrics["weak_query_groups_count"]], width=0.18, label="weak_groups")
    plt.xticks(x, variants)
    plt.title("Persistent Memory Chain Support")
    plt.legend()
    plt.tight_layout()
    _save_current_figure(figures_dir / "fig_persistent_object_memory_chain_support")

    plt.figure(figsize=(8.8, 4.8))
    plt.bar([item - 0.18 for item in x], [baseline_metrics["reappearance_support_rate"], uplift_metrics["reappearance_support_rate"]], width=0.18, label="reappearance_support")
    plt.bar([item for item in x], [baseline_metrics["object_memory_short_term_items_total"], uplift_metrics["object_memory_short_term_items_total"]], width=0.18, label="short_term_items")
    plt.bar([item + 0.18 for item in x], [baseline_metrics["object_memory_long_term_items_total"], uplift_metrics["object_memory_long_term_items_total"]], width=0.18, label="long_term_items")
    plt.xticks(x, variants)
    plt.title("Persistent Memory Reappearance")
    plt.legend()
    plt.tight_layout()
    _save_current_figure(figures_dir / "fig_persistent_object_memory_reappearance")

    compare_summary = {
        "suite_id": str(suite_id),
        "query_bank_id": query_bank.query_bank_id,
        "query_bank_version": query_bank.query_bank_version,
        "query_bank_hash": query_bank.query_bank_hash,
        "baseline": baseline_metrics,
        "uplift": uplift_metrics,
        "persistent_object_memory_status": status,
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
                "# Persistent Object Memory Pilot",
                "",
                f"- suite_id: `{suite_id}`",
                f"- query_bank_id: `{query_bank.query_bank_id}`",
                f"- persistent_object_memory_status: `{status}`",
                f"- next_action_recommendation: `{next_action}`",
                f"- baseline_logic_variant: `{baseline_metrics.get('object_memory_logic_variant', '')}`",
                f"- uplift_logic_variant: `{uplift_metrics.get('object_memory_logic_variant', '')}`",
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
        "persistent_object_memory_status": status,
        "next_action_recommendation": next_action,
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
        "persistent_object_memory_status": status,
        "next_action_recommendation": next_action,
    }


def build_persistent_object_memory_report(*, suite_dir: str | Path, baseline_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    lib = _require_pandas()
    suite_root = Path(suite_dir).resolve()
    baseline_root = Path(baseline_dir).resolve()
    suite_summary = _read_json(suite_root / "compare" / "compare_summary.json")
    baseline_summary = _read_json(baseline_root / "compare" / "compare_summary.json")
    suite_uplift = suite_summary.get("uplift", {}) if isinstance(suite_summary.get("uplift"), dict) else {}
    suite_baseline = suite_summary.get("baseline", {}) if isinstance(suite_summary.get("baseline"), dict) else {}
    baseline_uplift = baseline_summary.get("uplift", {}) if isinstance(baseline_summary.get("uplift"), dict) else {}
    baseline_baseline = baseline_summary.get("baseline", {}) if isinstance(baseline_summary.get("baseline"), dict) else {}

    object_memory_improved = bool(int(suite_uplift.get("object_memory_items_total", 0)) > int(suite_baseline.get("object_memory_items_total", 0)))
    short_term_memory_improved = bool(int(suite_uplift.get("object_memory_short_term_items_total", 0)) > int(suite_baseline.get("object_memory_short_term_items_total", 0)))
    long_term_memory_improved = bool(int(suite_uplift.get("object_memory_long_term_items_total", 0)) > int(suite_baseline.get("object_memory_long_term_items_total", 0)))
    reappearance_support_improved = bool(float(suite_uplift.get("reappearance_support_rate", 0.0)) > float(suite_baseline.get("reappearance_support_rate", 0.0)))
    lost_object_support_improved = bool(float(suite_uplift.get("lost_object_query_support_rate", 0.0)) > float(suite_baseline.get("lost_object_query_support_rate", 0.0)))
    chain_object_grounding_improved = bool(float(suite_uplift.get("chain_object_grounding_support_rate", 0.0)) > float(suite_baseline.get("chain_object_grounding_support_rate", 0.0)))
    query_strength_improved = bool(
        float(suite_uplift.get("query_strength_coverage_rate", 0.0)) > float(suite_baseline.get("query_strength_coverage_rate", 0.0))
        or int(suite_uplift.get("weak_query_groups_count", 0)) < int(suite_baseline.get("weak_query_groups_count", 0))
    )
    persistent_memory_gain_vs_fake = round(
        (float(suite_uplift.get("object_memory_long_term_items_total", 0.0)) - float(suite_baseline.get("object_memory_long_term_items_total", 0.0)))
        - (float(baseline_uplift.get("object_memory_long_term_items_total", 0.0)) - float(baseline_baseline.get("object_memory_long_term_items_total", 0.0))),
        6,
    )
    next_action = str(suite_summary.get("next_action_recommendation", "")).strip() or "keep_current_memory_logic"
    if long_term_memory_improved and reappearance_support_improved and lost_object_support_improved and chain_object_grounding_improved and query_strength_improved:
        next_action = "promote_persistent_object_memory"
    elif long_term_memory_improved and reappearance_support_improved and (not lost_object_support_improved or not chain_object_grounding_improved):
        next_action = "keep_current_memory_logic"
    elif not long_term_memory_improved and not lost_object_support_improved and not chain_object_grounding_improved:
        next_action = "persistent_memory_gain_insufficient"
    elif persistent_memory_gain_vs_fake <= 0.0:
        next_action = "need_retrieval_side_fix"

    rows = [
        {
            "suite_id": str(suite_summary.get("suite_id", suite_root.name)),
            "baseline_suite_id": str(baseline_summary.get("suite_id", baseline_root.name)),
            "query_bank_id": str(suite_summary.get("query_bank_id", "")),
            "persistent_object_memory_status": str(suite_summary.get("persistent_object_memory_status", "no_change")),
            "object_memory_improved": object_memory_improved,
            "short_term_memory_improved": short_term_memory_improved,
            "long_term_memory_improved": long_term_memory_improved,
            "reappearance_support_improved": reappearance_support_improved,
            "lost_object_support_improved": lost_object_support_improved,
            "chain_object_grounding_improved": chain_object_grounding_improved,
            "query_strength_improved": query_strength_improved,
            "persistent_memory_gain_vs_fake": persistent_memory_gain_vs_fake,
            "next_action_recommendation": next_action,
            "should_formalize_persistent_object_memory": next_action == "promote_persistent_object_memory",
            "should_stop_perception_route": next_action in {"need_retrieval_side_fix", "need_decision_side_fix"},
        }
    ]
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "baseline_dir": str(baseline_root),
        "suite_id": str(suite_summary.get("suite_id", suite_root.name)),
        "baseline_suite_id": str(baseline_summary.get("suite_id", baseline_root.name)),
        "query_bank_id": str(suite_summary.get("query_bank_id", "")),
        "persistent_object_memory_status": str(suite_summary.get("persistent_object_memory_status", "no_change")),
        "object_memory_improved": object_memory_improved,
        "short_term_memory_improved": short_term_memory_improved,
        "long_term_memory_improved": long_term_memory_improved,
        "reappearance_support_improved": reappearance_support_improved,
        "lost_object_support_improved": lost_object_support_improved,
        "chain_object_grounding_improved": chain_object_grounding_improved,
        "query_strength_improved": query_strength_improved,
        "persistent_memory_gain_vs_fake": persistent_memory_gain_vs_fake,
        "next_action_recommendation": next_action,
        "should_formalize_persistent_object_memory": next_action == "promote_persistent_object_memory",
        "should_stop_perception_route": next_action in {"need_retrieval_side_fix", "need_decision_side_fix"},
        "suite_compare_summary": suite_summary,
        "baseline_compare_summary": baseline_summary,
    }
    return lib.DataFrame(rows), snapshot


def write_persistent_object_memory_report_outputs(*, suite_dir: str | Path, baseline_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df, snapshot = build_persistent_object_memory_report(suite_dir=suite_dir, baseline_dir=baseline_dir)
    table_csv = tables_dir / "table_persistent_object_memory_uplift_summary.csv"
    table_md = tables_dir / "table_persistent_object_memory_uplift_summary.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"
    df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Persistent Object Memory Uplift Summary\n\n" + df_to_markdown_table(df))

    plt.figure(figsize=(8.2, 4.6))
    plt.bar(
        ["long_term", "reappearance", "lost_object", "chain_grounding", "query_strength"],
        [
            1.0 if bool(df.loc[0, "long_term_memory_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "reappearance_support_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "lost_object_support_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "chain_object_grounding_improved"]) else 0.0,
            1.0 if bool(df.loc[0, "query_strength_improved"]) else 0.0,
        ],
    )
    plt.ylim(0.0, 1.0)
    plt.title("Persistent Object Memory Uplift Summary")
    plt.tight_layout()
    figure_paths: list[str] = []
    for ext in ("png", "pdf"):
        target = (figures_dir / "fig_persistent_object_memory_uplift_summary").with_suffix(f".{ext}")
        plt.savefig(target)
        figure_paths.append(str(target))
    plt.close()

    report_lines = [
        "# Persistent Object Memory Uplift Report",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- baseline_dir: `{Path(baseline_dir).resolve()}`",
        f"- persistent_object_memory_status: `{snapshot.get('persistent_object_memory_status', 'no_change')}`",
        f"- object_memory_improved: `{snapshot.get('object_memory_improved', False)}`",
        f"- short_term_memory_improved: `{snapshot.get('short_term_memory_improved', False)}`",
        f"- long_term_memory_improved: `{snapshot.get('long_term_memory_improved', False)}`",
        f"- reappearance_support_improved: `{snapshot.get('reappearance_support_improved', False)}`",
        f"- lost_object_support_improved: `{snapshot.get('lost_object_support_improved', False)}`",
        f"- chain_object_grounding_improved: `{snapshot.get('chain_object_grounding_improved', False)}`",
        f"- query_strength_improved: `{snapshot.get('query_strength_improved', False)}`",
        f"- persistent_memory_gain_vs_fake: `{snapshot.get('persistent_memory_gain_vs_fake', 0.0)}`",
        f"- next_action_recommendation: `{snapshot.get('next_action_recommendation', '')}`",
        f"- should_formalize_persistent_object_memory: `{snapshot.get('should_formalize_persistent_object_memory', False)}`",
        f"- should_stop_perception_route: `{snapshot.get('should_stop_perception_route', False)}`",
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
        "should_formalize_persistent_object_memory": bool(snapshot.get("should_formalize_persistent_object_memory", False)),
        "persistent_object_memory_status": snapshot.get("persistent_object_memory_status", "no_change"),
    }
