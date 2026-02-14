from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        s = int(round(float(self.max_total_s)))
        return f"{s}/{int(self.max_tokens)}/{int(self.max_decisions)}"


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLQ evaluation across explicit budget points")
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", default="hard_pseudo_nlq")
    parser.add_argument("--budgets", required=True, help='e.g. "60/200/12,40/100/8,20/50/4"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    _parse_bool_with_neg(parser, "allow-gt-fallback", default=False)
    _parse_bool_with_neg(parser, "hard-constraints", default=True)
    _parse_bool_with_neg(parser, "safety-report", default=True)
    _parse_bool_with_neg(parser, "strict-uids", default=True)
    parser.add_argument("--allow-fallback-all-uids", action="store_true")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--eval-script", default=str(ROOT / "scripts" / "eval_nlq.py"))
    return parser.parse_args()


def _normalize_uid(text: str) -> str:
    token = str(text).replace("\ufeff", "").strip()
    if not token:
        return ""
    if token.lower().endswith(".mp4"):
        token = token[:-4]
    return token.strip().lower()


def uid_from_pov_json_path(p: Path) -> str:
    stem = p.stem
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    cleaned = re.sub(r"(?i)_v\d+_token$", "", cleaned)
    cleaned = re.sub(r"(?i)_token$", "", cleaned)
    uuid_re = re.compile(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
    match = uuid_re.search(cleaned) or uuid_re.search(stem)
    if match:
        return str(match.group(1)).lower()
    return cleaned


def _read_uids_file(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "")
        if "#" in text:
            text = text.split("#", 1)[0]
        text = text.strip()
        if not text:
            continue
        for token in re.split(r"[,\s]+", text):
            norm = _normalize_uid(token)
            if norm:
                out.append(norm)
    return out


def _discover_json_by_uid(json_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    files = sorted(json_dir.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(json_dir.glob("*.json"), key=lambda p: p.name.lower())
    for p in files:
        uid = _normalize_uid(uid_from_pov_json_path(p))
        if uid and uid not in out:
            out[uid] = p
    return out


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        chunks = [x.strip() for x in text.split("/") if x.strip()]
        if len(chunks) != 3:
            raise ValueError(f"invalid budget {text}")
        out.append(BudgetPoint(max_total_s=float(chunks[0]), max_tokens=int(chunks[1]), max_decisions=int(chunks[2])))
    if not out:
        raise ValueError("no budgets parsed")
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:
        return None
    return x


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NLQ Budget Sweep",
        "",
        "## Selection",
        "",
        f"- selection_mode: {selection.get('selection_mode', '')}",
        f"- uids_file_path: {selection.get('uids_file_path', '')}",
        f"- uids_requested: {selection.get('uids_requested', 0)}",
        f"- uids_found: {selection.get('uids_found', 0)}",
        f"- uids_missing_count: {selection.get('uids_missing_count', 0)}",
        f"- uids_missing_sample: {selection.get('uids_missing_sample', [])}",
        f"- dir_uids_sample: {selection.get('dir_uids_sample', [])}",
        "",
    ]
    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _to_int(v: Any) -> int | None:
    f = _to_float(v)
    if f is None:
        return None
    return int(round(float(f)))


def _normalize_reason(name: str) -> str:
    key = str(name or "").strip().lower()
    mapping = {
        "budget_insufficient": "budget_insufficient",
        "evidence_missing": "evidence_missing",
        "constraints_over_filtered": "constraints_over_filtered",
        "constraints_overfilter": "constraints_over_filtered",
        "retrieval_distractor": "retrieval_distractor",
        "retrieval_distractors": "retrieval_distractor",
        "distractor": "retrieval_distractor",
    }
    return mapping.get(key, "other")


def _parse_reason_counts(payload: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    rc = payload.get("reason_counts")
    if isinstance(rc, dict):
        for key, value in rc.items():
            reason = _normalize_reason(str(key))
            val = _to_int(value) or 0
            out[reason] = out.get(reason, 0) + int(val)
    elif isinstance(rc, list):
        for item in rc:
            if not isinstance(item, dict):
                continue
            reason = _normalize_reason(str(item.get("reason", item.get("name", ""))))
            val = _to_int(item.get("count", item.get("value", 0))) or 0
            out[reason] = out.get(reason, 0) + int(val)

    # Fallback: derive from critical_failures list.
    if not out:
        failures = payload.get("critical_failures", [])
        if isinstance(failures, list):
            for item in failures:
                if not isinstance(item, dict):
                    continue
                reason = _normalize_reason(str(item.get("reason", "")))
                out[reason] = out.get(reason, 0) + 1
    return out


def _extract_safety_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    denom = _to_int(payload.get("critical_fn_denominator"))
    if denom is None:
        denom = _to_int(payload.get("evaluated_critical_queries"))
    denom = int(denom or 0)
    count = int(_to_int(payload.get("critical_fn_count")) or 0)
    rate = _to_float(payload.get("critical_fn_rate"))
    if rate is None:
        rate = float(count / max(1, denom))
    gran = str(payload.get("count_granularity", "row=(variant,budget,query)"))

    reason_counts = _parse_reason_counts(payload)
    b_cnt = int(reason_counts.get("budget_insufficient", 0))
    e_cnt = int(reason_counts.get("evidence_missing", 0))
    c_cnt = int(reason_counts.get("constraints_over_filtered", 0))
    r_cnt = int(reason_counts.get("retrieval_distractor", 0))
    known_sum = b_cnt + e_cnt + c_cnt + r_cnt
    other_cnt = max(0, int(sum(reason_counts.values()) - known_sum))
    if known_sum == 0 and sum(reason_counts.values()) == 0 and count > 0:
        other_cnt = int(count)

    return {
        "safety_count_granularity": gran,
        "safety_critical_fn_denominator": int(denom),
        "safety_critical_fn_count": int(count),
        "safety_critical_fn_rate": float(rate),
        "safety_reason_budget_insufficient_rate": float(b_cnt / max(1, denom)),
        "safety_reason_evidence_missing_rate": float(e_cnt / max(1, denom)),
        "safety_reason_constraints_over_filtered_rate": float(c_cnt / max(1, denom)),
        "safety_reason_retrieval_distractor_rate": float(r_cnt / max(1, denom)),
        "safety_reason_other_rate": float(other_cnt / max(1, denom)),
        "safety_budget_insufficient_share": float(b_cnt / max(1, count)),
    }


def _zero_safety_metrics() -> dict[str, Any]:
    return {
        "safety_count_granularity": "row=(variant,budget,query)",
        "safety_critical_fn_denominator": 0,
        "safety_critical_fn_count": 0,
        "safety_critical_fn_rate": 0.0,
        "safety_reason_budget_insufficient_rate": 0.0,
        "safety_reason_evidence_missing_rate": 0.0,
        "safety_reason_constraints_over_filtered_rate": 0.0,
        "safety_reason_retrieval_distractor_rate": 0.0,
        "safety_reason_other_rate": 0.0,
        "safety_budget_insufficient_share": 0.0,
    }


def _extract_uid_metrics(uid_dir: Path, include_safety: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    results_rows = _read_csv(uid_dir / "nlq_results.csv")
    full_rows = [r for r in results_rows if str(r.get("variant", "")) == "full"]
    out["nlq_full_hit_at_k_strict"] = _mean([_to_float(r.get("hit_at_k_strict")) or 0.0 for r in full_rows])
    out["nlq_full_hit_at_1_strict"] = _mean([_to_float(r.get("hit_at_1_strict")) or 0.0 for r in full_rows])
    out["nlq_full_top1_in_distractor_rate"] = _mean(
        [_to_float(r.get("top1_in_distractor")) or 0.0 for r in full_rows]
    )
    out["nlq_full_fp_rate"] = out["nlq_full_top1_in_distractor_rate"]
    out["nlq_full_mrr"] = _mean([_to_float(r.get("mrr")) or 0.0 for r in full_rows])
    summary_rows = _read_csv(uid_dir / "nlq_summary.csv")
    lost_rows = [
        r
        for r in summary_rows
        if str(r.get("variant", "")) == "full" and str(r.get("query_type", "")) == "hard_pseudo_lost_object"
    ]
    if lost_rows:
        out["lost_object_hit_at_k_strict"] = _mean([_to_float(r.get("hit_at_k_strict")) or 0.0 for r in lost_rows])
        out["lost_object_mrr"] = _mean([_to_float(r.get("mrr")) or 0.0 for r in lost_rows])
        out["lost_object_top1_in_distractor_rate"] = _mean(
            [_to_float(r.get("top1_in_distractor_rate")) or 0.0 for r in lost_rows]
        )
        out["lost_object_critical_fn_rate"] = _mean(
            [_to_float(r.get("critical_fn_rate")) or 0.0 for r in lost_rows]
        )
    chain_rows = [
        r
        for r in summary_rows
        if str(r.get("variant", "")) == "full" and str(r.get("query_type", "")) == "hard_pseudo_chain"
    ]
    if chain_rows:
        out["chain_hit_at_k_strict"] = _mean([_to_float(r.get("hit_at_k_strict")) or 0.0 for r in chain_rows])
        out["chain_mrr"] = _mean([_to_float(r.get("mrr")) or 0.0 for r in chain_rows])
        out["chain_top1_in_distractor_rate"] = _mean(
            [_to_float(r.get("top1_in_distractor_rate")) or 0.0 for r in chain_rows]
        )
        out["chain_success_rate"] = _mean([_to_float(r.get("chain_success_rate")) or 0.0 for r in chain_rows])
        out["chain_step1_has_hit_rate"] = _mean(
            [_to_float(r.get("chain_step1_has_hit_rate")) or 0.0 for r in chain_rows]
        )
        out["chain_step2_has_hit_rate"] = _mean(
            [_to_float(r.get("chain_step2_has_hit_rate")) or 0.0 for r in chain_rows]
        )
        out["chain_filtered_ratio_step2"] = _mean(
            [_to_float(r.get("chain_filtered_ratio_step2")) or 0.0 for r in chain_rows]
        )
    if include_safety:
        out.update(_zero_safety_metrics())
        safety_path = uid_dir / "safety_report.json"
        if safety_path.exists():
            try:
                payload = json.loads(safety_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    # prefer full variant stats when available, fall back to global fields.
                    var_stats = payload.get("variant_stats", {})
                    if isinstance(var_stats, dict) and isinstance(var_stats.get("full"), dict):
                        full_payload = dict(payload)
                        full_payload["critical_fn_rate"] = float(var_stats["full"].get("critical_fn_rate", 0.0))
                        full_payload["critical_fn_count"] = int(var_stats["full"].get("critical_fn_count", 0))
                        full_payload["critical_fn_denominator"] = int(
                            var_stats["full"].get("critical_fn_denominator", payload.get("critical_fn_denominator", 0))
                        )
                        out.update(_extract_safety_metrics(full_payload))
                    else:
                        out.update(_extract_safety_metrics(payload))
            except Exception:
                out.update(_zero_safety_metrics())
    return out


def _make_figures(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: float(r.get("budget_seconds", 0.0)))
    xs = [float(r.get("budget_seconds", 0.0)) for r in sorted_rows]
    y_quality = [float(r.get("nlq_full_mrr", 0.0)) for r in sorted_rows]
    y_hitk = [float(r.get("nlq_full_hit_at_k_strict", 0.0)) for r in sorted_rows]
    y_fp = [float(r.get("nlq_full_fp_rate", 0.0)) for r in sorted_rows]
    y_critical = [float(_to_float(r.get("safety_critical_fn_rate")) or 0.0) for r in sorted_rows]
    y_budget_ins = [
        float(_to_float(r.get("safety_reason_budget_insufficient_rate")) or 0.0) for r in sorted_rows
    ]
    y_evidence = [float(_to_float(r.get("safety_reason_evidence_missing_rate")) or 0.0) for r in sorted_rows]
    y_constraints = [
        float(_to_float(r.get("safety_reason_constraints_over_filtered_rate")) or 0.0) for r in sorted_rows
    ]
    y_distractor = [
        float(_to_float(r.get("safety_reason_retrieval_distractor_rate")) or 0.0) for r in sorted_rows
    ]
    y_other = [float(_to_float(r.get("safety_reason_other_rate")) or 0.0) for r in sorted_rows]

    out_paths: list[str] = []

    p1 = out_dir / "fig_nlq_quality_vs_budget_seconds"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, y_quality, marker="o")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("nlq_full_mrr")
    plt.title("NLQ Quality vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        path = p1.with_suffix(f".{ext}")
        plt.savefig(path)
        out_paths.append(str(path))
    plt.close()

    p2 = out_dir / "fig_nlq_strict_vs_budget_seconds"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, y_hitk, marker="o", label="nlq_full_hit_at_k_strict")
    plt.plot(xs, y_fp, marker="o", label="nlq_full_fp_rate")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("Metric")
    plt.title("NLQ Strict Metrics vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        path = p2.with_suffix(f".{ext}")
        plt.savefig(path)
        out_paths.append(str(path))
    plt.close()

    p3 = out_dir / "fig_nlq_critical_fn_rate_vs_budget_seconds"
    plt.figure(figsize=(7.0, 4.2))
    plt.plot(xs, y_critical, marker="o")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("safety_critical_fn_rate")
    plt.title("NLQ Critical FN Rate vs Budget Seconds")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    for ext in formats:
        path = p3.with_suffix(f".{ext}")
        plt.savefig(path)
        out_paths.append(str(path))
    plt.close()

    p4 = out_dir / "fig_nlq_failure_attribution_vs_budget_seconds"
    plt.figure(figsize=(7.2, 4.4))
    plt.stackplot(
        xs,
        y_budget_ins,
        y_evidence,
        y_constraints,
        y_distractor,
        y_other,
        labels=[
            "budget_insufficient",
            "evidence_missing",
            "constraints_over_filtered",
            "retrieval_distractor",
            "other",
        ],
    )
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel("Failure Attribution Rate (reason_count / denominator)")
    plt.title("NLQ Failure Attribution vs Budget Seconds")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    for ext in formats:
        path = p4.with_suffix(f".{ext}")
        plt.savefig(path)
        out_paths.append(str(path))
    plt.close()

    return out_paths


def _write_lost_object_tables(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: float(r.get("budget_seconds", 0.0))):
        table_rows.append(
            {
                "budget_key": str(row.get("budget_key", "")),
                "budget_seconds": float(row.get("budget_seconds", 0.0)),
                "lost_object_hit_at_k_strict": float(_to_float(row.get("lost_object_hit_at_k_strict")) or 0.0),
                "lost_object_mrr": float(_to_float(row.get("lost_object_mrr")) or 0.0),
                "lost_object_top1_in_distractor_rate": float(
                    _to_float(row.get("lost_object_top1_in_distractor_rate")) or 0.0
                ),
                "lost_object_critical_fn_rate": float(_to_float(row.get("lost_object_critical_fn_rate")) or 0.0),
            }
        )
    csv_path = out_dir / "table_lost_object_budget.csv"
    md_path = out_dir / "table_lost_object_budget.md"
    _write_csv(csv_path, table_rows)
    _write_md(
        md_path,
        table_rows,
        {
            "selection_mode": "budget_aggregate",
            "uids_file_path": "",
            "uids_requested": "",
            "uids_found": "",
            "uids_missing_count": "",
            "uids_missing_sample": [],
            "dir_uids_sample": [],
        },
    )
    return csv_path, md_path


def _write_chain_tables(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: float(r.get("budget_seconds", 0.0))):
        table_rows.append(
            {
                "budget_key": str(row.get("budget_key", "")),
                "budget_seconds": float(row.get("budget_seconds", 0.0)),
                "chain_hit_at_k_strict": float(_to_float(row.get("chain_hit_at_k_strict")) or 0.0),
                "chain_mrr": float(_to_float(row.get("chain_mrr")) or 0.0),
                "chain_success_rate": float(_to_float(row.get("chain_success_rate")) or 0.0),
                "chain_top1_in_distractor_rate": float(_to_float(row.get("chain_top1_in_distractor_rate")) or 0.0),
                "chain_step1_has_hit_rate": float(_to_float(row.get("chain_step1_has_hit_rate")) or 0.0),
                "chain_step2_has_hit_rate": float(_to_float(row.get("chain_step2_has_hit_rate")) or 0.0),
                "chain_filtered_ratio_step2": float(_to_float(row.get("chain_filtered_ratio_step2")) or 0.0),
            }
        )
    csv_path = out_dir / "table_chain_summary.csv"
    md_path = out_dir / "table_chain_summary.md"
    _write_csv(csv_path, table_rows)
    _write_md(
        md_path,
        table_rows,
        {
            "selection_mode": "budget_aggregate",
            "uids_file_path": "",
            "uids_requested": "",
            "uids_found": "",
            "uids_missing_count": "",
            "uids_missing_sample": [],
            "dir_uids_sample": [],
        },
    )
    return csv_path, md_path


def main() -> int:
    args = parse_args()
    json_dir = Path(args.json_dir)
    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    per_budget_root = out_dir / "per_budget"
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_budget_root.mkdir(parents=True, exist_ok=True)

    budgets = _parse_budgets(args.budgets)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    discovered = _discover_json_by_uid(json_dir)
    if not discovered:
        print(f"error=no json found in {json_dir}")
        return 2
    dir_uids = sorted(discovered.keys())

    selection: dict[str, Any] = {
        "selection_mode": "all_json",
        "uids_file_path": None,
        "uids_requested": 0,
        "uids_found": 0,
        "uids_missing_count": 0,
        "uids_missing_sample": [],
        "dir_uids_sample": dir_uids[:5],
    }
    if args.uids_file:
        uids_file = Path(args.uids_file)
        requested = _read_uids_file(uids_file)
        selected_uids: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for uid in requested:
            norm = _normalize_uid(uid)
            if norm in discovered:
                if norm not in seen:
                    selected_uids.append(norm)
                    seen.add(norm)
            else:
                missing.append(uid)
        selection.update(
            {
                "selection_mode": "uids_file",
                "uids_file_path": str(uids_file),
                "uids_requested": len(requested),
                "uids_found": len(selected_uids),
                "uids_missing_count": len(missing),
                "uids_missing_sample": missing[:10],
            }
        )
        if bool(args.strict_uids):
            if not selected_uids or missing:
                print("error=uid selection failed under strict mode")
                print(f"uids_file_path={uids_file}")
                print(f"uids_requested={len(requested)}")
                print(f"uids_found={len(selected_uids)}")
                print(f"uids_missing_count={len(missing)}")
                print(f"uids_requested_sample={requested[:10]}")
                print(f"uids_missing_sample={missing[:10]}")
                print(f"dir_uid_sample={dir_uids[:5]}")
                return 2
        else:
            if not selected_uids:
                if bool(args.allow_fallback_all_uids):
                    selected_uids = list(dir_uids)
                    selection["selection_mode"] = "fallback_all_json"
                else:
                    print("error=no uid matched and fallback is disabled")
                    return 2
            elif missing:
                selection["selection_mode"] = "uids_file_partial"
    else:
        selected_uids = list(dir_uids)
        selection["uids_found"] = len(selected_uids)

    if bool(args.allow_fallback_all_uids) and bool(args.strict_uids):
        print("error=--allow-fallback-all-uids requires --no-strict-uids")
        return 2
    if not selected_uids:
        print("error=no selected uids")
        return 2

    print(f"selection_mode={selection.get('selection_mode')}")
    print(f"selected_uids={len(selected_uids)}")
    print(f"uids_file_path={selection.get('uids_file_path')}")
    print(f"uids_missing_count={selection.get('uids_missing_count')}")

    eval_script = Path(args.eval_script)
    if not eval_script.exists():
        print(f"error=eval script not found: {eval_script}")
        return 2

    run_rows: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    for budget in budgets:
        budget_dir = per_budget_root / budget.key.replace("/", "_")
        for uid in selected_uids:
            json_path = discovered[uid]
            uid_out = budget_dir / uid
            uid_out.mkdir(parents=True, exist_ok=True)
            index_prefix = index_dir / uid

            cmd = [
                sys.executable,
                str(eval_script),
                "--json",
                str(json_path),
                "--index",
                str(index_prefix),
                "--out_dir",
                str(uid_out),
                "--mode",
                str(args.mode),
                "--seed",
                str(int(args.seed)),
                "--top-k",
                str(int(args.top_k)),
                "--budget-max-total-s",
                str(float(budget.max_total_s)),
                "--budget-max-tokens",
                str(int(budget.max_tokens)),
                "--budget-max-decisions",
                str(int(budget.max_decisions)),
            ]
            if bool(args.allow_gt_fallback):
                cmd.append("--allow-gt-fallback")
            else:
                cmd.append("--no-allow-gt-fallback")
            cmd.extend(["--hard-constraints", "on" if bool(args.hard_constraints) else "off"])
            cmd.append("--no-safety-gate")

            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
            (uid_out / "runner.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (uid_out / "runner.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

            row: dict[str, Any] = {
                "video_uid": uid,
                "budget_key": budget.key,
                "budget_seconds": float(budget.max_total_s),
                "budget_max_total_s": float(budget.max_total_s),
                "budget_max_tokens": int(budget.max_tokens),
                "budget_max_decisions": int(budget.max_decisions),
                "returncode": int(proc.returncode),
            }
            if proc.returncode == 0:
                row.update(_extract_uid_metrics(uid_out, include_safety=bool(args.safety_report)))
            run_rows.append(row)
            run_meta.append(
                {
                    "uid": uid,
                    "budget_key": budget.key,
                    "returncode": int(proc.returncode),
                    "stdout_log": str(uid_out / "runner.stdout.log"),
                    "stderr_log": str(uid_out / "runner.stderr.log"),
                    "out_dir": str(uid_out),
                }
            )

    by_budget: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        by_budget.setdefault(str(row["budget_key"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for budget in budgets:
        key = budget.key
        rows = by_budget.get(key, [])
        granularity = "row=(variant,budget,query)"
        for row in rows:
            maybe = str(row.get("safety_count_granularity", "")).strip()
            if maybe:
                granularity = maybe
                break
        agg: dict[str, Any] = {
            "budget_key": key,
            "budget_seconds": float(budget.max_total_s),
            "budget_max_total_s": float(budget.max_total_s),
            "budget_max_tokens": int(budget.max_tokens),
            "budget_max_decisions": int(budget.max_decisions),
            "num_uids": len(selected_uids),
            "runs_total": len(rows),
            "runs_ok": sum(1 for r in rows if int(r.get("returncode", 1)) == 0),
            "safety_count_granularity": granularity,
        }
        numeric_keys = [
            "nlq_full_hit_at_k_strict",
            "nlq_full_hit_at_1_strict",
            "nlq_full_top1_in_distractor_rate",
            "nlq_full_fp_rate",
            "nlq_full_mrr",
            "lost_object_hit_at_k_strict",
            "lost_object_mrr",
            "lost_object_top1_in_distractor_rate",
            "lost_object_critical_fn_rate",
            "chain_hit_at_k_strict",
            "chain_mrr",
            "chain_top1_in_distractor_rate",
            "chain_success_rate",
            "chain_step1_has_hit_rate",
            "chain_step2_has_hit_rate",
            "chain_filtered_ratio_step2",
            "safety_critical_fn_denominator",
            "safety_critical_fn_count",
            "safety_critical_fn_rate",
            "safety_reason_budget_insufficient_rate",
            "safety_reason_evidence_missing_rate",
            "safety_reason_constraints_over_filtered_rate",
            "safety_reason_retrieval_distractor_rate",
            "safety_reason_other_rate",
            "safety_budget_insufficient_share",
        ]
        for nk in numeric_keys:
            vals = [float(r[nk]) for r in rows if isinstance(r.get(nk), (int, float))]
            if vals:
                agg[nk] = _mean(vals)
        aggregate_rows.append(agg)

    metrics_csv = agg_dir / "metrics_by_budget.csv"
    metrics_md = agg_dir / "metrics_by_budget.md"
    _write_csv(metrics_csv, aggregate_rows)
    _write_md(metrics_md, aggregate_rows, selection)
    lost_csv, lost_md = _write_lost_object_tables(aggregate_rows, agg_dir)
    chain_csv, chain_md = _write_chain_tables(aggregate_rows, agg_dir)
    figure_paths = _make_figures(aggregate_rows, fig_dir, formats)

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json_dir": str(json_dir),
            "index_dir": str(index_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "mode": str(args.mode),
            "budgets": [b.__dict__ for b in budgets],
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "allow_gt_fallback": bool(args.allow_gt_fallback),
            "hard_constraints": bool(args.hard_constraints),
            "safety_report": bool(args.safety_report),
            "strict_uids": bool(args.strict_uids),
            "allow_fallback_all_uids": bool(args.allow_fallback_all_uids),
            "eval_script": str(eval_script),
        },
        "selection": selection,
        "selected_uids": list(selected_uids),
        "runs": run_meta,
        "outputs": {
            "metrics_by_budget_csv": str(metrics_csv),
            "metrics_by_budget_md": str(metrics_md),
            "lost_object_table_csv": str(lost_csv),
            "lost_object_table_md": str(lost_md),
            "chain_table_csv": str(chain_csv),
            "chain_table_md": str(chain_md),
            "figures": figure_paths,
            "safety_figures": [
                x
                for x in figure_paths
                if "fig_nlq_critical_fn_rate_vs_budget_seconds" in x
                or "fig_nlq_failure_attribution_vs_budget_seconds" in x
            ],
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"budgets={len(budgets)}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    print(f"saved_lost_object_table={lost_csv}")
    print(f"saved_chain_table={chain_csv}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
