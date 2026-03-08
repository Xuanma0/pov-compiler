from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - dependency expected in runtime env.
    pd = None

from pov_compiler.bench.query_bank import (
    QueryBank,
    QueryBankEntry,
    QueryBankGroup,
    stable_query_bank_hash,
)
from pov_compiler.bench.reporting.latex import df_to_markdown_table

GENERIC_OBJECT_LABELS = {"person", "people", "face", "hand", "hands"}


def _require_pandas() -> Any:
    if pd is None:  # pragma: no cover - dependency expected in runtime env.
        raise ImportError("pandas is required for query-bank selection reporting.")
    return pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


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


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency expected in runtime env.
        raise RuntimeError("PyYAML is required for query-bank rewrite outputs.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _resolve_path(raw_value: str | None, base_dir: Path) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("Expected non-empty path.")
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return (_repo_root() / path).resolve()


def _strip_repo_summary_terms(query: str) -> str:
    parts = [part for part in str(query or "").split() if part]
    filtered = [part for part in parts if not part.startswith("repo_mode=") and not part.startswith("summary_topk=")]
    return " ".join(filtered)


def _query_literal(text: str, key: str) -> str:
    prefix = f"{key}="
    for part in str(text or "").split():
        if part.startswith(prefix):
            return part.split("=", 1)[1].strip()
    return ""


def _sanitize_label(label: str) -> str:
    return str(label or "").strip().lower().replace(" ", "_")


def _human_label(label: str) -> str:
    return str(label or "").strip().lower().replace("_", " ")


def _load_manifest(suite_root: Path) -> dict[str, Any]:
    return _load_yaml(suite_root / "manifest" / "experiment_manifest.yaml")


def _load_bank_from_manifest_field(manifest_payload: dict[str, Any], field_path: list[str], base_dir: Path) -> QueryBank | None:
    current: Any = manifest_payload
    for key in field_path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if not str(current or "").strip():
        return None
    return QueryBank.from_path(_resolve_path(str(current), base_dir))


def _load_primary_bank(suite_root: Path, manifest_payload: dict[str, Any]) -> QueryBank:
    bank = _load_bank_from_manifest_field(manifest_payload, ["query_bank"], suite_root / "manifest")
    if bank is not None:
        return bank
    copied_root = suite_root / "manifest" / "query_banks"
    copied = sorted(copied_root.glob("*.yaml"))
    if copied:
        return QueryBank.from_path(copied[0])
    raise FileNotFoundError(f"No query bank could be loaded for suite: {suite_root}")


def _load_source_bank(suite_root: Path, manifest_payload: dict[str, Any], current_bank: QueryBank) -> QueryBank:
    bank = _load_bank_from_manifest_field(manifest_payload, ["rewrite", "source_query_bank"], suite_root / "manifest")
    return bank if bank is not None else current_bank


def _load_signal_seed_bank(suite_root: Path, manifest_payload: dict[str, Any]) -> QueryBank | None:
    return _load_bank_from_manifest_field(manifest_payload, ["rewrite", "signal_seed_query_bank"], suite_root / "manifest")


def _evidence_suite_dirs(suite_root: Path, manifest_payload: dict[str, Any]) -> dict[str, Path]:
    evidence = manifest_payload.get("rewrite", {}).get("evidence_dirs", {})
    if not isinstance(evidence, dict):
        evidence = {}
    out: dict[str, Path] = {"suite": suite_root}
    for key, raw in evidence.items():
        text = str(raw or "").strip()
        if not text:
            continue
        out[str(key)] = _resolve_path(text, suite_root / "manifest")
    return out


def _artifact_json(paths: dict[str, Path], relpath: str) -> dict[str, Any]:
    for base in (paths.get("suite"), paths.get("query_strength_suite"), paths.get("signal_uplift_suite")):
        if base is None:
            continue
        candidate = base / relpath
        if candidate.exists():
            return _read_json(candidate)
    return {}


def _rank_non_generic_labels(paths: dict[str, Path]) -> list[str]:
    search_bases = [paths.get("suite"), paths.get("signal_uplift_suite")]
    counts: Counter[str] = Counter()
    for base in search_bases:
        if base is None:
            continue
        payload = _read_json(base / "runs" / "uplift" / "output.json")
        if not payload:
            continue
        for frame in payload.get("perception", {}).get("frames", []) or []:
            if not isinstance(frame, dict):
                continue
            for obj in frame.get("objects", []) or []:
                if not isinstance(obj, dict):
                    continue
                label = str(obj.get("label", "")).strip().lower()
                if not label or label in GENERIC_OBJECT_LABELS:
                    continue
                counts[label] += 1
        for item in payload.get("object_memory_v0", []) or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("object_name", "")).strip().lower()
            if not label or label in GENERIC_OBJECT_LABELS:
                continue
            counts[label] += 1
    if not counts:
        return ["door", "cup"]
    ranked = [label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    if len(ranked) == 1:
        return ranked + ["cup"]
    return ranked[:2]


def _candidate_groups(
    query_strength_snapshot: dict[str, Any],
    uplift_summary: dict[str, Any],
    source_bank: QueryBank,
) -> tuple[list[str], list[str], list[str]]:
    source_group_ids = [group.group_id for group in source_bank.groups]
    candidate_groups = [str(item) for item in query_strength_snapshot.get("uplift_candidate_groups", []) if str(item)]
    if not candidate_groups:
        candidate_groups = [
            str(item.get("query_group"))
            for item in uplift_summary.get("candidate_groups", [])
            if isinstance(item, dict) and str(item.get("query_group", "")).strip()
        ]
    analysis_groups = [str(item) for item in query_strength_snapshot.get("analysis_only_query_groups", []) if str(item)]
    if not analysis_groups:
        analysis_groups = [
            str(item.get("query_group"))
            for item in uplift_summary.get("analysis_only_groups", [])
            if isinstance(item, dict) and str(item.get("query_group", "")).strip()
        ]
    candidate_groups = [group for group in source_group_ids if group in candidate_groups]
    analysis_groups = [group for group in source_group_ids if group in analysis_groups and group not in candidate_groups]
    drop_groups = [group for group in source_group_ids if group not in candidate_groups and group not in analysis_groups]
    return candidate_groups, analysis_groups, drop_groups


def _selection_confidence(signal_snapshot: dict[str, Any], query_strength_snapshot: dict[str, Any], chosen_labels: list[str]) -> str:
    if bool(signal_snapshot.get("object_coverage_improved")) and bool(signal_snapshot.get("object_memory_improved")):
        if int(query_strength_snapshot.get("uplift_candidate_count", 0) or 0) >= 2 and len(chosen_labels) >= 2:
            return "medium"
    return "low"


def _decision_queries(seed_bank: QueryBank | None, source_bank: QueryBank) -> list[QueryBankEntry]:
    seed_entries = [entry for entry in (seed_bank.queries if seed_bank else []) if entry.group == "decision" and bool(entry.enabled)]
    if not seed_entries:
        seed_entries = [entry for entry in source_bank.queries if entry.group == "decision" and bool(entry.enabled)]
    out: list[QueryBankEntry] = []
    for entry in seed_entries:
        query_text = _strip_repo_summary_terms(entry.query)
        query_id_suffix = _sanitize_label(_query_literal(query_text, "decision") or entry.query_id)
        out.append(
            QueryBankEntry(
                query_id=f"v2_decision_{query_id_suffix}_last",
                group="decision",
                task="nlq",
                mode="hard_pseudo_nlq",
                query=query_text,
                signal_tags=["decision"],
                sensitive_to=["decision_backend"],
                enabled=True,
                notes=f"rewritten_from={entry.query_id}",
            )
        )
    deduped: list[QueryBankEntry] = []
    seen_queries: set[str] = set()
    for entry in out:
        if entry.query in seen_queries:
            continue
        deduped.append(entry)
        seen_queries.add(entry.query)
    return deduped[:2]


def _chain_queries(labels: list[str]) -> list[QueryBankEntry]:
    out: list[QueryBankEntry] = []
    for idx, label in enumerate(labels[:2]):
        slug = _sanitize_label(label)
        out.append(
            QueryBankEntry(
                query_id=f"v2_chain_scene_to_{slug}",
                group="chain",
                task="streaming" if idx == 1 else "nlq",
                mode="hard_pseudo_chain",
                query=f"token=SCENE_CHANGE which=last top_k=6 then interaction_object={slug} top_k=6 chain_derive=time+object",
                signal_tags=["chain", "object"],
                sensitive_to=["chain_backoff"] + (["streaming"] if idx == 1 else []),
                enabled=True,
                notes=f"generated_from_yolo26n_label={_human_label(label)}",
            )
        )
    return out


def _lost_object_queries(labels: list[str]) -> list[QueryBankEntry]:
    out: list[QueryBankEntry] = []
    for idx, label in enumerate(labels[:2]):
        slug = _sanitize_label(label)
        query = f"lost_object={slug} which=last top_k=6"
        mode = "hard_pseudo_nlq"
        signal_tags = ["lost_object", "object_memory"]
        sensitive_to = ["object_memory"]
        if idx == 1:
            query = f"{query} then token=SCENE_CHANGE which=last top_k=6 chain_derive=time+object"
            mode = "hard_pseudo_chain"
            signal_tags.append("chain")
            sensitive_to.append("chain_backoff")
        out.append(
            QueryBankEntry(
                query_id=f"v2_lost_object_{slug}",
                group="lost_object",
                task="nlq",
                mode=mode,
                query=query,
                signal_tags=signal_tags,
                sensitive_to=sensitive_to,
                enabled=True,
                notes=f"generated_from_yolo26n_label={_human_label(label)}",
            )
        )
    return out


def _candidate_bank_payload(
    *,
    source_bank: QueryBank,
    signal_seed_bank: QueryBank | None,
    signal_snapshot: dict[str, Any],
    query_strength_snapshot: dict[str, Any],
    uplift_summary: dict[str, Any],
    chosen_labels: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    candidate_groups = [
        QueryBankGroup(group_id="decision", description="Decision queries preserved without repo-summary dependency", tags=["decision"]),
        QueryBankGroup(group_id="chain", description="Chain queries rewritten toward non-generic YOLO26n object evidence", tags=["chain", "object"]),
        QueryBankGroup(group_id="lost_object", description="Lost-object queries rewritten toward non-generic YOLO26n object-memory evidence", tags=["lost_object", "object_memory"]),
    ]
    candidate_queries = [
        *_decision_queries(signal_seed_bank, source_bank),
        *_chain_queries(chosen_labels),
        *_lost_object_queries(chosen_labels),
    ]
    analysis_queries = [entry.model_dump() for entry in source_bank.queries if entry.group == "repo_summary" and bool(entry.enabled)]
    confidence = _selection_confidence(signal_snapshot, query_strength_snapshot, chosen_labels)
    candidate_payload: dict[str, Any] = {
        "query_bank_id": "core_real_v2_candidate",
        "query_bank_version": "2-candidate",
        "description": "Signal-aware candidate main-result query bank rewritten to consume stronger local YOLO26n object evidence.",
        "source_query_bank_id": source_bank.query_bank_id,
        "source_query_bank_hash": source_bank.query_bank_hash,
        "rewrite_basis": [
            "query_strength_audit uplift-candidate groups",
            "query_uplift_candidates candidate groups",
            "signal_uplift object-coverage/object-memory gains",
            f"chosen_non_generic_labels={[_human_label(label) for label in chosen_labels[:2]]}",
        ],
        "selection_confidence": confidence,
        "provenance": {
            "strategy": "signal_aware_query_bank_rewrite",
            "source": "v1.53",
            "seed_signal_bank": signal_seed_bank.query_bank_id if signal_seed_bank else None,
            "target_signal_model": "yolo26n",
            "uplift_basis": uplift_summary.get("uplift_basis", {}),
        },
        "groups": [group.model_dump() for group in candidate_groups],
        "queries": [entry.model_dump() for entry in candidate_queries],
    }
    candidate_payload["query_bank_hash"] = stable_query_bank_hash(candidate_payload)

    analysis_payload: dict[str, Any] = {
        "query_bank_id": "core_real_v2_analysis_only",
        "query_bank_version": "2-candidate",
        "description": "Analysis-only companion bank for queries that remain too weak for main-result promotion under current signal evidence.",
        "source_query_bank_id": source_bank.query_bank_id,
        "source_query_bank_hash": source_bank.query_bank_hash,
        "rewrite_basis": [
            "query_strength_audit analysis-only groups",
            "query_uplift_candidates analysis_only groups",
            "keep weak repo-summary queries out of the main candidate bank",
        ],
        "selection_confidence": confidence,
        "provenance": {
            "strategy": "signal_aware_query_bank_rewrite",
            "source": "v1.53",
            "target_role": "analysis_only",
        },
        "groups": [
            {
                "group_id": "repo_summary",
                "description": "Weak but still informative repo-summary queries kept for analysis only",
                "tags": ["repo_summary", "summary_first"],
            }
        ],
        "queries": analysis_queries,
    }
    analysis_payload["query_bank_hash"] = stable_query_bank_hash(analysis_payload)

    selection_rows: list[dict[str, Any]] = []
    for entry in candidate_queries:
        if entry.group == "decision":
            signal_support = "decision_signal_stable"
            selection_reason = "kept_decision_query_without_repo_summary_dependency"
        else:
            label = _human_label(_query_literal(entry.query, "interaction_object") or _query_literal(entry.query, "lost_object"))
            signal_support = f"non_generic_yolo_label:{label}"
            selection_reason = "rewritten_to_consume_yolo26n_object_signal"
        selection_rows.append(
            {
                "query_id": entry.query_id,
                "source_query_id": str(entry.notes).replace("rewritten_from=", "").strip() if "rewritten_from=" in str(entry.notes) else "",
                "source_group": entry.group,
                "decision": "candidate",
                "selection_reason": selection_reason,
                "signal_support": signal_support,
                "query_strength_flag": "uplift_candidate_group",
                "delta_support": str(query_strength_snapshot.get("main_recommendation", "") or uplift_summary.get("uplift_basis", {}).get("source_main_recommendation", "")),
                "uplift_support": "signal_uplift_improved" if bool(signal_snapshot.get("signal_uplift_status") == "improved") else "signal_uplift_unclear",
            }
        )
    for query in analysis_queries:
        selection_rows.append(
            {
                "query_id": str(query.get("query_id", "")),
                "source_query_id": str(query.get("query_id", "")),
                "source_group": str(query.get("group", "repo_summary")),
                "decision": "analysis_only",
                "selection_reason": "kept_for_analysis_only_due_to_no_matched_tasks",
                "signal_support": "repo_summary_weak",
                "query_strength_flag": "analysis_only_group",
                "delta_support": str(query_strength_snapshot.get("main_recommendation", "") or "no_matched_tasks"),
                "uplift_support": "signal_uplift_not_consumed_by_current_query_bank",
            }
        )
    return candidate_payload, analysis_payload, selection_rows


def build_query_bank_rewrite_artifacts(*, suite_dir: str | Path) -> dict[str, Any]:
    suite_root = Path(suite_dir).resolve()
    manifest_payload = _load_manifest(suite_root)
    current_bank = _load_primary_bank(suite_root, manifest_payload)
    source_bank = _load_source_bank(suite_root, manifest_payload, current_bank)
    signal_seed_bank = _load_signal_seed_bank(suite_root, manifest_payload)
    evidence_dirs = _evidence_suite_dirs(suite_root, manifest_payload)

    query_strength_snapshot = _artifact_json(evidence_dirs, "query_strength_audit/snapshot.json")
    uplift_summary = _artifact_json(evidence_dirs, "query_uplift_candidates/query_uplift_candidates/uplift_summary.json")
    signal_snapshot = _artifact_json(evidence_dirs, "signal_uplift/snapshot.json")
    delta_snapshot = _artifact_json(evidence_dirs, "delta_audit/snapshot.json")
    admission_snapshot = _artifact_json(evidence_dirs, "admission_calibration/snapshot.json")
    provider_summary = _artifact_json(evidence_dirs, "provider_telemetry/summary.json")

    chosen_labels = _rank_non_generic_labels(evidence_dirs)
    candidate_groups, analysis_groups, drop_groups = _candidate_groups(query_strength_snapshot, uplift_summary, source_bank)
    candidate_payload, analysis_payload, selection_rows = _candidate_bank_payload(
        source_bank=source_bank,
        signal_seed_bank=signal_seed_bank,
        signal_snapshot=signal_snapshot,
        query_strength_snapshot=query_strength_snapshot,
        uplift_summary=uplift_summary,
        chosen_labels=chosen_labels,
    )
    selection_rows = [
        row
        for row in selection_rows
        if (
            (row["decision"] == "candidate" and row["source_group"] in candidate_groups)
            or (row["decision"] == "analysis_only" and row["source_group"] in analysis_groups)
        )
    ]

    dropped_rows: list[dict[str, Any]] = []
    for query in source_bank.queries:
        if query.group not in drop_groups:
            continue
        dropped_rows.append(
            {
                "query_id": query.query_id,
                "source_query_id": query.query_id,
                "source_group": query.group,
                "decision": "drop",
                "selection_reason": "group_not_selected_for_v2_candidate",
                "signal_support": "insufficient",
                "query_strength_flag": "dropped_group",
                "delta_support": str(delta_snapshot.get("main_recommendation", "")),
                "uplift_support": "not_supported",
            }
        )
    selection_rows.extend(dropped_rows)

    selection_counter = Counter(str(row.get("decision", "")) for row in selection_rows)
    rewrite_summary = {
        "candidate_count": int(len(candidate_payload.get("queries", []))),
        "analysis_only_count": int(len(analysis_payload.get("queries", []))),
        "dropped_count": int(len(dropped_rows)),
        "source_query_bank_id": source_bank.query_bank_id,
        "source_query_bank_hash": source_bank.query_bank_hash,
        "rewrite_rules_applied": [
            "candidate_groups_from_query_uplift_candidates_or_query_strength_audit",
            "analysis_only_groups_from_query_strength_audit",
            "non_generic_object_labels_ranked_from_uplift_output",
            "decision_queries_stripped_of_repo_summary_dependency",
            "chain_and_lost_object_queries_retargeted_to_yolo26n_labels",
        ],
        "rewrite_confidence": candidate_payload.get("selection_confidence", "low"),
        "candidate_groups": candidate_groups,
        "analysis_only_groups": analysis_groups,
        "drop_groups": drop_groups,
        "chosen_signal_labels": [_human_label(label) for label in chosen_labels[:2]],
        "signal_uplift_status": signal_snapshot.get("signal_uplift_status", "unknown"),
        "signal_next_action": signal_snapshot.get("next_action_recommendation", ""),
        "query_strength_main_recommendation": query_strength_snapshot.get("main_recommendation", ""),
        "delta_audit_main_recommendation": delta_snapshot.get("main_recommendation", ""),
        "admission_calibration_status": admission_snapshot.get("calibration_status", ""),
        "provider_availability": provider_summary.get("availability", ""),
    }
    selection_summary = {
        "candidate_count": int(selection_counter.get("candidate", 0)),
        "analysis_only_count": int(selection_counter.get("analysis_only", 0)),
        "drop_count": int(selection_counter.get("drop", 0)),
        "candidate_groups": candidate_groups,
        "analysis_only_groups": analysis_groups,
        "dropped_groups": drop_groups,
    }
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_root),
        "source_query_bank_id": source_bank.query_bank_id,
        "source_query_bank_hash": source_bank.query_bank_hash,
        "current_query_bank_id": current_bank.query_bank_id,
        "current_query_bank_hash": current_bank.query_bank_hash,
        "selected_signal_labels": [_human_label(label) for label in chosen_labels[:2]],
        "rewrite_summary": rewrite_summary,
        "selection_summary": selection_summary,
        "evidence": {
            "query_strength_snapshot": query_strength_snapshot,
            "query_uplift_summary": uplift_summary,
            "signal_uplift_snapshot": signal_snapshot,
            "delta_audit_snapshot": delta_snapshot,
            "admission_calibration_snapshot": admission_snapshot,
            "provider_telemetry_summary": provider_summary,
        },
    }
    return {
        "candidate_payload": candidate_payload,
        "analysis_payload": analysis_payload,
        "selection_rows": selection_rows,
        "rewrite_summary": rewrite_summary,
        "selection_summary": selection_summary,
        "snapshot": snapshot,
    }


def write_query_bank_rewrite_outputs(*, suite_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    artifacts = build_query_bank_rewrite_artifacts(suite_dir=suite_dir)
    out_root = Path(out_dir).resolve()
    rewrite_root = out_root / "query_bank_rewrite"
    candidate_path = rewrite_root / "core_real_v2_candidate.yaml"
    analysis_path = rewrite_root / "core_real_v2_analysis_only.yaml"
    summary_path = rewrite_root / "rewrite_summary.json"
    report_path = rewrite_root / "report.md"
    snapshot_path = rewrite_root / "snapshot.json"

    _dump_yaml(candidate_path, artifacts["candidate_payload"])
    _dump_yaml(analysis_path, artifacts["analysis_payload"])
    _write_json(summary_path, artifacts["rewrite_summary"])
    _write_json(snapshot_path, artifacts["snapshot"])

    report_lines = [
        "# Query Bank Rewrite",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- source_query_bank_id: `{artifacts['rewrite_summary'].get('source_query_bank_id', '')}`",
        f"- source_query_bank_hash: `{artifacts['rewrite_summary'].get('source_query_bank_hash', '')}`",
        f"- rewrite_confidence: `{artifacts['rewrite_summary'].get('rewrite_confidence', 'low')}`",
        f"- candidate_groups: `{artifacts['rewrite_summary'].get('candidate_groups', [])}`",
        f"- analysis_only_groups: `{artifacts['rewrite_summary'].get('analysis_only_groups', [])}`",
        f"- chosen_signal_labels: `{artifacts['rewrite_summary'].get('chosen_signal_labels', [])}`",
        f"- signal_uplift_status: `{artifacts['rewrite_summary'].get('signal_uplift_status', 'unknown')}`",
        f"- query_strength_main_recommendation: `{artifacts['rewrite_summary'].get('query_strength_main_recommendation', '')}`",
        f"- delta_audit_main_recommendation: `{artifacts['rewrite_summary'].get('delta_audit_main_recommendation', '')}`",
        "",
        "## Files",
        "",
        f"- rewrite_summary_json: `{summary_path}`",
        f"- candidate_yaml: `{candidate_path}`",
        f"- analysis_only_yaml: `{analysis_path}`",
        f"- report_md: `{report_path}`",
        f"- snapshot_json: `{snapshot_path}`",
    ]
    _write_text(report_path, "\n".join(report_lines))
    return {
        "candidate_yaml": candidate_path,
        "analysis_yaml": analysis_path,
        "rewrite_summary_json": summary_path,
        "report_md": report_path,
        "snapshot_json": snapshot_path,
        "rewrite_summary": artifacts["rewrite_summary"],
    }


def write_query_bank_selection_outputs(*, suite_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    lib = _require_pandas()
    artifacts = build_query_bank_rewrite_artifacts(suite_dir=suite_dir)
    out_root = Path(out_dir).resolve()
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_csv = tables_dir / "table_query_bank_selection.csv"
    table_md = tables_dir / "table_query_bank_selection.md"
    report_md = out_root / "report.md"
    snapshot_json = out_root / "snapshot.json"

    df = lib.DataFrame(artifacts["selection_rows"])
    if df.empty:
        df = lib.DataFrame(
            [
                {
                    "query_id": "none",
                    "source_query_id": "",
                    "source_group": "",
                    "decision": "drop",
                    "selection_reason": "no_selection_rows",
                    "signal_support": "none",
                    "query_strength_flag": "missing",
                    "delta_support": "",
                    "uplift_support": "",
                }
            ]
        )
    df.to_csv(table_csv, index=False)
    _write_text(table_md, "# Query Bank Selection\n\n" + df_to_markdown_table(df))

    snapshot = {
        **artifacts["snapshot"],
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "report_md": str(report_md),
            "snapshot_json": str(snapshot_json),
        },
    }
    report_lines = [
        "# Query Bank Selection",
        "",
        f"- suite_dir: `{Path(suite_dir).resolve()}`",
        f"- selection_summary: `{json.dumps(artifacts['selection_summary'], ensure_ascii=False, sort_keys=True)}`",
        f"- rewrite_summary: `{json.dumps(artifacts['rewrite_summary'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Files",
        "",
        f"- table_csv: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- report_md: `{report_md}`",
        f"- snapshot_json: `{snapshot_json}`",
    ]
    _write_text(report_md, "\n".join(report_lines))
    _write_json(snapshot_json, snapshot)
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "report_md": report_md,
        "snapshot_json": snapshot_json,
        "selection_summary": artifacts["selection_summary"],
    }
