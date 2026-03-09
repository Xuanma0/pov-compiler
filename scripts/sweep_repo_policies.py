from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.repository import build_repo_chunks, deduplicate_chunks, select_chunks_for_query
from pov_compiler.repository.policy import load_policy_yaml, policy_cfg_hash
from pov_compiler.schemas import Output


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        return f"{int(round(self.max_total_s))}/{int(self.max_tokens)}/{int(self.max_decisions)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoV1 policy sweep (write x read x budgets)")
    parser.add_argument("--pov-json-dir", default="", help="Directory containing *_v03_decisions.json files")
    parser.add_argument("--json", default="", help="Single *_v03_decisions.json input")
    parser.add_argument("--uids-file", default=None, help="Optional uid list file (strict match)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--repo-cfg", default="", help="Optional repo config yaml/json")
    parser.add_argument("--budgets", default="20/50/4,60/200/12")
    parser.add_argument("--write-policies", default="fixed_interval,event_triggered,novelty")
    parser.add_argument("--read-policies", default="budgeted_topk,diverse")
    parser.add_argument("--query", default="anchor=turn_head top_k=6")
    parser.add_argument("--with-nlq", action="store_true", help="Optional; currently proxy-first sweep")
    parser.add_argument("--strict-uids", dest="strict_uids", action="store_true", default=True)
    parser.add_argument("--no-strict-uids", dest="strict_uids", action="store_false")
    parser.add_argument("--formats", default="png,pdf")
    return parser.parse_args()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for chunk in str(raw).split(","):
        token = chunk.strip()
        if not token:
            continue
        parts = [x.strip() for x in token.split("/") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"invalid budget token: {token}")
        out.append(BudgetPoint(float(parts[0]), int(parts[1]), int(parts[2])))
    if not out:
        raise ValueError("no budgets parsed")
    return out


def _parse_csv_tokens(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _normalize_uid(text: str) -> str:
    token = str(text).replace("\ufeff", "").strip()
    if token.lower().endswith(".mp4"):
        token = token[:-4]
    return token.lower()


def _uid_from_json(path: Path) -> str:
    stem = path.stem
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    uuid_re = re.compile(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
    m = uuid_re.search(cleaned) or uuid_re.search(stem)
    if m:
        return _normalize_uid(m.group(1))
    return _normalize_uid(cleaned)


def _discover_jsons(args: argparse.Namespace) -> tuple[list[Path], dict[str, Any]]:
    selection = {
        "selection_mode": "",
        "uids_file_path": "",
        "uids_requested": 0,
        "uids_found": 0,
        "uids_missing_count": 0,
        "uids_missing_sample": [],
        "dir_uids_sample": [],
    }
    if args.json:
        p = Path(args.json)
        if not p.exists():
            raise FileNotFoundError(str(p))
        selection["selection_mode"] = "single_json"
        return [p], selection

    pov_json_dir = Path(args.pov_json_dir)
    if not pov_json_dir.exists():
        raise FileNotFoundError(str(pov_json_dir))
    files = sorted(pov_json_dir.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(pov_json_dir.glob("*.json"), key=lambda p: p.name.lower())
    by_uid: dict[str, Path] = {}
    for p in files:
        uid = _uid_from_json(p)
        if uid and uid not in by_uid:
            by_uid[uid] = p
    dir_uids = sorted(by_uid.keys())
    selection["dir_uids_sample"] = dir_uids[:5]

    uids_file = args.uids_file
    if not uids_file:
        selection["selection_mode"] = "all_json"
        return [by_uid[k] for k in dir_uids], selection

    uid_path = Path(str(uids_file))
    requested: list[str] = []
    for line in uid_path.read_text(encoding="utf-8").splitlines():
        text = str(line).replace("\ufeff", "")
        if "#" in text:
            text = text.split("#", 1)[0]
        text = text.strip()
        if not text:
            continue
        for token in re.split(r"[,\s]+", text):
            u = _normalize_uid(token)
            if u:
                requested.append(u)
    found = [u for u in requested if u in by_uid]
    missing = [u for u in requested if u not in by_uid]
    selection.update(
        {
            "selection_mode": "uids_file",
            "uids_file_path": str(uid_path),
            "uids_requested": len(requested),
            "uids_found": len(found),
            "uids_missing_count": len(missing),
            "uids_missing_sample": missing[:10],
        }
    )
    if args.strict_uids and (not found or missing):
        raise RuntimeError(
            f"strict uid matching failed: found={len(found)} missing={len(missing)} "
            f"uids_file={uid_path} sample_missing={missing[:5]} dir_uid_sample={dir_uids[:5]}"
        )
    return [by_uid[u] for u in found], selection


def _model_validate_output(payload: dict[str, Any]) -> Output:
    if hasattr(Output, "model_validate"):
        return Output.model_validate(payload)  # type: ignore[attr-defined]
    return Output.parse_obj(payload)


def _interval_union_seconds(items: list[dict[str, Any]]) -> float:
    intervals: list[tuple[float, float]] = []
    for item in items:
        t0 = _to_float(item.get("t0", 0.0), 0.0)
        t1 = _to_float(item.get("t1", 0.0), 0.0)
        if t1 > t0:
            intervals.append((t0, t1))
    if not intervals:
        return 0.0
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float]] = []
    s0, s1 = intervals[0]
    for a0, a1 in intervals[1:]:
        if a0 <= s1:
            s1 = max(s1, a1)
            continue
        merged.append((s0, s1))
        s0, s1 = a0, a1
    merged.append((s0, s1))
    return float(sum(max(0.0, b - a) for a, b in merged))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in cols:
                cols.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# RepoV1 Policy Sweep\n\nNo rows.\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [
        "# RepoV1 Policy Sweep",
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
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    settings = sorted({str(r["setting"]) for r in rows})

    def _save(fig, out_prefix: Path) -> None:
        for ext in formats:
            p = out_prefix.with_suffix(f".{ext}")
            fig.savefig(p)
            paths.append(str(p))
        plt.close(fig)

    # Quality vs budget.
    fig1 = plt.figure(figsize=(7.6, 4.2))
    for setting in settings:
        s_rows = sorted([r for r in rows if str(r["setting"]) == setting], key=lambda x: float(x["budget_seconds"]))
        plt.plot(
            [float(r["budget_seconds"]) for r in s_rows],
            [float(r["quality_proxy"]) for r in s_rows],
            marker="o",
            label=setting,
        )
    plt.xlabel("Budget Seconds")
    plt.ylabel("quality_proxy")
    plt.title("RepoV1 Quality Proxy vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    _save(fig1, out_dir / "fig_repo_quality_vs_budget_seconds")

    # Size/latency proxy vs budget.
    fig2 = plt.figure(figsize=(7.6, 4.2))
    for setting in settings:
        s_rows = sorted([r for r in rows if str(r["setting"]) == setting], key=lambda x: float(x["budget_seconds"]))
        plt.plot(
            [float(r["budget_seconds"]) for r in s_rows],
            [float(r["selected_tokens_est"]) for r in s_rows],
            marker="o",
            label=setting,
        )
    plt.xlabel("Budget Seconds")
    plt.ylabel("selected_tokens_est")
    plt.title("RepoV1 Context Size Proxy vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    _save(fig2, out_dir / "fig_repo_size_vs_budget_seconds")
    return paths


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths, selection = _discover_jsons(args)
    if not json_paths:
        raise RuntimeError("no input json selected")
    budgets = _parse_budgets(args.budgets)
    write_policies = _parse_csv_tokens(args.write_policies)
    read_policies = _parse_csv_tokens(args.read_policies)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    base_cfg = {}
    if args.repo_cfg:
        payload = load_policy_yaml(args.repo_cfg)
        base_cfg = dict(payload.get("repo", payload))
    if args.with_nlq:
        print("[warn] --with-nlq is currently proxy-only in RepoV1 sweep; NLQ metrics are not executed in this script.")

    rows_raw: list[dict[str, Any]] = []
    for json_path in json_paths:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        output = _model_validate_output(payload)
        duration_s = _to_float(output.meta.get("duration_s", 0.0), 0.0)
        for wp in write_policies:
            for rp in read_policies:
                for b in budgets:
                    repo_cfg = {
                        **base_cfg,
                        "write_policy": {"name": wp, **dict(base_cfg.get("write_policy", {}))},
                        "read_policy": {"name": rp, **dict(base_cfg.get("read_policy", {}))},
                    }
                    # Ensure policy name is not overridden by base cfg.
                    repo_cfg["write_policy"]["name"] = wp
                    repo_cfg["read_policy"]["name"] = rp

                    chunks_before = build_repo_chunks(output, cfg=repo_cfg)
                    chunks_after = deduplicate_chunks(chunks_before, cfg=dict(repo_cfg.get("dedup", {})))
                    selected = select_chunks_for_query(
                        chunks_after,
                        query=str(args.query or ""),
                        budget={
                            "max_repo_chunks": max(4, min(64, int(b.max_tokens // 8))),
                            "max_tokens": int(b.max_tokens),
                            "max_seconds": float(b.max_total_s),
                            "repo_read_policy": rp,
                        },
                        cfg={"read_policy": {"name": rp}},
                    )

                    sel_rows = [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in selected]
                    coverage_s = _interval_union_seconds(sel_rows)
                    coverage_ratio = float(coverage_s / max(1e-6, duration_s)) if duration_s > 0 else 0.0
                    importance_mean = float(mean([float(c.importance) for c in selected])) if selected else 0.0
                    selected_tokens_est = int(sum(int(c.meta.get("token_est", max(1, len(c.text) // 4))) for c in selected))
                    quality_proxy = float(
                        0.45 * min(1.0, coverage_ratio)
                        + 0.35 * min(1.0, importance_mean)
                        + 0.2 * min(1.0, len(selected) / max(1, int(b.max_tokens // 8)))
                    )
                    setting = f"{wp}|{rp}"
                    rows_raw.append(
                        {
                            "video_id": str(output.video_id),
                            "setting": setting,
                            "write_policy": wp,
                            "read_policy": rp,
                            "budget_key": b.key,
                            "budget_seconds": float(b.max_total_s),
                            "budget_max_total_s": float(b.max_total_s),
                            "budget_max_tokens": int(b.max_tokens),
                            "budget_max_decisions": int(b.max_decisions),
                            "chunks_before_dedup": int(len(chunks_before)),
                            "chunks_after_dedup": int(len(chunks_after)),
                            "selected_chunks": int(len(selected)),
                            "selected_tokens_est": int(selected_tokens_est),
                            "coverage_s": float(coverage_s),
                            "coverage_ratio": float(coverage_ratio),
                            "importance_mean": float(importance_mean),
                            "quality_proxy": float(quality_proxy),
                            "cfg_hash": policy_cfg_hash(repo_cfg),
                        }
                    )

    # Aggregate over video_id (macro average).
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows_raw:
        key = (str(row["setting"]), str(row["budget_key"]))
        grouped.setdefault(key, []).append(row)
    agg_rows: list[dict[str, Any]] = []
    for (setting, budget_key), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], _to_float(kv[1][0]["budget_seconds"], 0.0))):
        sample = rows[0]
        agg_rows.append(
            {
                "setting": setting,
                "write_policy": sample["write_policy"],
                "read_policy": sample["read_policy"],
                "budget_key": budget_key,
                "budget_seconds": float(sample["budget_seconds"]),
                "budget_max_total_s": float(sample["budget_max_total_s"]),
                "budget_max_tokens": int(sample["budget_max_tokens"]),
                "budget_max_decisions": int(sample["budget_max_decisions"]),
                "num_uids": len(rows),
                "chunks_before_dedup": float(mean([_to_float(r["chunks_before_dedup"], 0.0) for r in rows])),
                "chunks_after_dedup": float(mean([_to_float(r["chunks_after_dedup"], 0.0) for r in rows])),
                "selected_chunks": float(mean([_to_float(r["selected_chunks"], 0.0) for r in rows])),
                "selected_tokens_est": float(mean([_to_float(r["selected_tokens_est"], 0.0) for r in rows])),
                "coverage_ratio": float(mean([_to_float(r["coverage_ratio"], 0.0) for r in rows])),
                "importance_mean": float(mean([_to_float(r["importance_mean"], 0.0) for r in rows])),
                "quality_proxy": float(mean([_to_float(r["quality_proxy"], 0.0) for r in rows])),
                "objective": float(
                    mean(
                        [
                            0.6 * _to_float(r["quality_proxy"], 0.0)
                            - 0.25 * min(1.0, _to_float(r["selected_tokens_est"], 0.0) / max(1.0, float(sample["budget_max_tokens"])))
                            - 0.15 * min(1.0, _to_float(r["selected_chunks"], 0.0) / max(1.0, _to_float(sample["selected_chunks"], 1.0)))
                            for r in rows
                        ]
                    )
                ),
            }
        )

    metrics_csv = agg_dir / "metrics_by_setting.csv"
    metrics_md = agg_dir / "metrics_by_setting.md"
    _write_csv(metrics_csv, agg_rows)
    _write_md(metrics_md, agg_rows, selection=selection)
    figure_paths = _plot(agg_rows, fig_dir, formats)

    default_setting = "fixed_interval|budgeted_topk"
    best = max(agg_rows, key=lambda r: float(r.get("objective", 0.0))) if agg_rows else None
    default_rows = [r for r in agg_rows if str(r.get("setting")) == default_setting]
    default_best = max(default_rows, key=lambda r: float(r.get("objective", 0.0))) if default_rows else None

    best_report = out_dir / "best_report.md"
    report_lines = ["# RepoV1 Sweep Best Report", ""]
    if best is None:
        report_lines.append("No rows.")
    else:
        report_lines.extend(
            [
                f"- best_setting: `{best.get('setting', '')}`",
                f"- best_budget: `{best.get('budget_key', '')}`",
                f"- best_objective: `{float(best.get('objective', 0.0)):.4f}`",
                f"- best_quality_proxy: `{float(best.get('quality_proxy', 0.0)):.4f}`",
            ]
        )
        if default_best:
            report_lines.extend(
                [
                    "",
                    "## Delta vs default (fixed_interval|budgeted_topk)",
                    "",
                    f"- default_budget: `{default_best.get('budget_key', '')}`",
                    f"- delta_objective: `{float(best.get('objective', 0.0)) - float(default_best.get('objective', 0.0)):.4f}`",
                    f"- delta_quality_proxy: `{float(best.get('quality_proxy', 0.0)) - float(default_best.get('quality_proxy', 0.0)):.4f}`",
                ]
            )
    best_report.write_text("\n".join(report_lines), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "json_count": len(json_paths),
            "json_paths": [str(p) for p in json_paths],
            "query": str(args.query or ""),
            "budgets": [b.__dict__ for b in budgets],
            "write_policies": write_policies,
            "read_policies": read_policies,
            "with_nlq": bool(args.with_nlq),
            "selection": selection,
        },
        "outputs": {
            "metrics_by_setting_csv": str(metrics_csv),
            "metrics_by_setting_md": str(metrics_md),
            "best_report_md": str(best_report),
            "figures": figure_paths,
            "best": best or {},
            "default_best": default_best or {},
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selection_mode={selection.get('selection_mode', '')}")
    print(f"selected_uids={len(json_paths)}")
    print(f"settings={len(write_policies) * len(read_policies)}")
    print(f"budgets={len(budgets)}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_metrics_md={metrics_md}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_best_report={best_report}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
