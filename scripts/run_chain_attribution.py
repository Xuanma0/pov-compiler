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
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass(frozen=True)
class BudgetPoint:
    max_total_s: float
    max_tokens: int
    max_decisions: int

    @property
    def key(self) -> str:
        return f"{int(round(self.max_total_s))}/{int(self.max_tokens)}/{int(self.max_decisions)}"

    @property
    def tag(self) -> str:
        return f"s{int(round(self.max_total_s))}_t{int(self.max_tokens)}_d{int(self.max_decisions)}"


@dataclass(frozen=True)
class VariantRun:
    code: str
    name: str
    use_repo: bool
    repo_policy: str
    reranker_off: bool


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chain attribution panel runner (A/B/C/D)")
    parser.add_argument("--pov-json-dir", required=True)
    parser.add_argument("--uids-file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="20/50/4,60/200/12")
    parser.add_argument("--queries-total", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mode", default="hard_pseudo_chain")
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--eval-script", default=str(ROOT / "scripts" / "eval_nlq.py"))
    parser.add_argument("--repo-policy", default="query_aware")
    parser.add_argument("--formats", default="png,pdf")
    _parse_bool_with_neg(parser, "strict-uids", default=True)
    parser.add_argument("--allow-fallback-all-uids", action="store_true")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return float(out)


def _normalize_uid(text: str) -> str:
    token = str(text).replace("\ufeff", "").strip()
    if not token:
        return ""
    if token.lower().endswith(".mp4"):
        token = token[:-4]
    return token.lower()


def _uid_from_json_path(path: Path) -> str:
    stem = path.stem
    cleaned = re.sub(r"(?i)_v\d+_decisions$", "", stem)
    cleaned = re.sub(r"(?i)_v03_decisions$", "", cleaned)
    cleaned = re.sub(r"(?i)_decisions$", "", cleaned)
    uuid_re = re.compile(r"(?i)([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
    match = uuid_re.search(cleaned) or uuid_re.search(stem)
    if match:
        return str(match.group(1)).lower()
    return cleaned.lower()


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
            uid = _normalize_uid(token)
            if uid:
                out.append(uid)
    return out


def _discover_jsons(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    files = sorted(root.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(root.glob("*.json"), key=lambda p: p.name.lower())
    for path in files:
        uid = _normalize_uid(_uid_from_json_path(path))
        if uid and uid not in out:
            out[uid] = path
    return out


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for token in str(raw).split(","):
        part = token.strip()
        if not part:
            continue
        pieces = [x.strip() for x in part.split("/") if x.strip()]
        if len(pieces) != 3:
            raise ValueError(f"invalid budget chunk: {part}")
        out.append(BudgetPoint(max_total_s=float(pieces[0]), max_tokens=int(pieces[1]), max_decisions=int(pieces[2])))
    if not out:
        raise ValueError("no budgets parsed")
    return out


def _render_cmd(cmd: list[str]) -> str:
    quoted = [f'"{x}"' if (" " in str(x) or "\t" in str(x)) else str(x) for x in cmd]
    return " ".join(quoted)


def _run(cmd: list[str], *, cwd: Path, log_prefix: Path, commands_file: Path) -> int:
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with commands_file.open("a", encoding="utf-8") as f:
        f.write(f"# {ts}\n{_render_cmd(cmd)}\n\n")
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    (log_prefix.with_suffix(".stdout.log")).write_text(proc.stdout or "", encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
    return int(proc.returncode)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str, summary_lines: list[str]) -> None:
    lines = [f"# {title}", "", *summary_lines, "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _index_prefix_for_uid(uid: str, index_dir: Path | None, json_dir: Path) -> str | None:
    base = index_dir if index_dir is not None else (json_dir.parent / "cache")
    if not base.exists():
        return None
    prefix = base / uid
    return str(prefix)


def _build_reranker_off_config(path: Path) -> Path:
    from pov_compiler.retrieval.reranker_config import WeightConfig

    cfg = WeightConfig().to_dict()
    cfg["name"] = "reranker_off"
    for key, value in list(cfg.items()):
        if isinstance(value, (int, float)):
            cfg[key] = 0.0
    cfg["w_semantic"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _extract_metrics(nlq_summary_csv: Path, safety_json: Path) -> dict[str, float]:
    rows = _read_csv(nlq_summary_csv)
    chain_rows = [r for r in rows if str(r.get("query_type", "")).strip() == "hard_pseudo_chain"]
    full_rows = [r for r in chain_rows if str(r.get("variant", "")).strip() == "full"]
    chosen = full_rows[0] if full_rows else (chain_rows[0] if chain_rows else {})
    safety_payload: dict[str, Any] = {}
    if safety_json.exists():
        try:
            payload = json.loads(safety_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                safety_payload = payload
        except Exception:
            safety_payload = {}

    def _num(key: str, default: float = 0.0) -> float:
        value = _to_float(chosen.get(key))
        return float(default if value is None else value)

    return {
        "chain_success_rate": _num("chain_success_rate"),
        "chain_waiting_rate": _num("chain_waiting_rate"),
        "chain_fail_constraints_over_filtered_rate": _num("chain_fail_constraints_over_filtered_rate"),
        "chain_fail_retrieval_distractor_rate": _num("chain_fail_retrieval_distractor_rate"),
        "chain_fail_evidence_missing_rate": _num("chain_fail_evidence_missing_rate"),
        "mrr_strict": _num("mrr"),
        "hit_at_k_strict": _num("hit_at_k_strict"),
        "top1_in_distractor_rate": _num("top1_in_distractor_rate"),
        "critical_fn_rate": float(_to_float(safety_payload.get("critical_fn_rate")) or 0.0),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _make_figures(rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: list[str] = []
    variants = ["A", "B", "C", "D"]
    by_budget: dict[float, dict[str, dict[str, float]]] = {}
    for row in rows:
        sec = float(_to_float(row.get("budget_seconds")) or 0.0)
        code = str(row.get("variant_code", ""))
        by_budget.setdefault(sec, {})[code] = row
    xs = sorted(by_budget.keys())
    if not xs:
        return fig_paths

    # 1) Success vs budget.
    p1 = out_dir / "fig_chain_attribution_success_vs_budget_seconds"
    plt.figure(figsize=(8.2, 4.4))
    for code in variants:
        ys: list[float] = []
        for x in xs:
            ys.append(float(_to_float(by_budget.get(x, {}).get(code, {}).get("chain_success_rate")) or 0.0))
        plt.plot(xs, ys, marker="o", label=f"variant_{code}")
    plt.xlabel("budget_seconds")
    plt.ylabel("chain_success_rate")
    plt.title("Chain Attribution: Success vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()

    # 2) Delta success vs budget (B/C/D against A).
    p2 = out_dir / "fig_chain_attribution_delta_success_vs_budget_seconds"
    plt.figure(figsize=(8.2, 4.4))
    for code in ("B", "C", "D"):
        ys: list[float] = []
        for x in xs:
            row = by_budget.get(x, {}).get(code, {})
            ys.append(float(_to_float(row.get("delta_success")) or 0.0))
        plt.plot(xs, ys, marker="o", label=f"{code}-A")
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("budget_seconds")
    plt.ylabel("delta_chain_success_rate")
    plt.title("Chain Attribution: Delta Success vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()

    # 3) Failure attribution (stacked; D variant for readability).
    p3 = out_dir / "fig_chain_attribution_failure_attribution_vs_budget_seconds"
    plt.figure(figsize=(8.2, 4.4))
    vals_constraints: list[float] = []
    vals_distractor: list[float] = []
    vals_evidence: list[float] = []
    for x in xs:
        row_d = by_budget.get(x, {}).get("D", {})
        vals_constraints.append(float(_to_float(row_d.get("chain_fail_constraints_over_filtered_rate")) or 0.0))
        vals_distractor.append(float(_to_float(row_d.get("chain_fail_retrieval_distractor_rate")) or 0.0))
        vals_evidence.append(float(_to_float(row_d.get("chain_fail_evidence_missing_rate")) or 0.0))
    plt.bar(xs, vals_constraints, width=2.5, label="constraints_over_filtered")
    plt.bar(xs, vals_distractor, width=2.5, bottom=vals_constraints, label="retrieval_distractor")
    stacked = [vals_constraints[i] + vals_distractor[i] for i in range(len(xs))]
    plt.bar(xs, vals_evidence, width=2.5, bottom=stacked, label="evidence_missing")
    plt.xlabel("budget_seconds")
    plt.ylabel("failure_rate (variant D)")
    plt.title("Chain Attribution Failure Breakdown (D)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p3.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()

    # 4) Tradeoff scatter (success vs waiting).
    p4 = out_dir / "fig_chain_attribution_tradeoff"
    plt.figure(figsize=(7.0, 4.8))
    for code in variants:
        xs_local: list[float] = []
        ys_local: list[float] = []
        for x in xs:
            row = by_budget.get(x, {}).get(code, {})
            xs_local.append(float(_to_float(row.get("chain_waiting_rate")) or 0.0))
            ys_local.append(float(_to_float(row.get("chain_success_rate")) or 0.0))
        plt.scatter(xs_local, ys_local, label=f"variant_{code}")
    plt.xlabel("chain_waiting_rate")
    plt.ylabel("chain_success_rate")
    plt.title("Chain Attribution Tradeoff")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p4.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()

    return fig_paths


def main() -> int:
    args = parse_args()
    pov_json_dir = Path(args.pov_json_dir)
    out_dir = Path(args.out_dir)
    compare_dir = out_dir / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    logs_dir = compare_dir / "logs"
    commands_file = compare_dir / "commands.sh"
    for p in (out_dir, compare_dir, tables_dir, figures_dir, logs_dir):
        p.mkdir(parents=True, exist_ok=True)
    if not commands_file.exists():
        commands_file.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    budgets = _parse_budgets(args.budgets)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]
    discovered = _discover_jsons(pov_json_dir)
    if not discovered:
        print(f"error=no_json_found dir={pov_json_dir}")
        return 2
    dir_uids = sorted(discovered.keys())
    selection: dict[str, Any] = {
        "selection_mode": "all_json",
        "uids_file_path": None,
        "uids_requested": 0,
        "uids_found": len(dir_uids),
        "uids_missing_count": 0,
        "uids_missing_sample": [],
        "dir_uids_sample": dir_uids[:5],
    }
    if args.uids_file:
        uids_path = Path(args.uids_file)
        requested = _read_uids_file(uids_path)
        selected: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for uid in requested:
            norm = _normalize_uid(uid)
            if norm in discovered:
                if norm not in seen:
                    selected.append(norm)
                    seen.add(norm)
            else:
                missing.append(uid)
        selection.update(
            {
                "selection_mode": "uids_file",
                "uids_file_path": str(uids_path),
                "uids_requested": len(requested),
                "uids_found": len(selected),
                "uids_missing_count": len(missing),
                "uids_missing_sample": missing[:10],
            }
        )
        if bool(args.strict_uids):
            if not selected or missing:
                print("error=uid_selection_failed")
                print(f"selection_mode=uids_file")
                print(f"uids_file_path={uids_path}")
                print(f"uids_requested={len(requested)}")
                print(f"uids_found={len(selected)}")
                print(f"uids_missing_count={len(missing)}")
                print(f"uids_missing_sample={missing[:10]}")
                print(f"dir_uids_sample={dir_uids[:5]}")
                return 2
        else:
            if not selected:
                if bool(args.allow_fallback_all_uids):
                    selected = list(dir_uids)
                    selection["selection_mode"] = "fallback_all_json"
                else:
                    print("error=uid_selection_empty_fallback_disabled")
                    return 2
        selected_uids = selected
    else:
        selected_uids = list(dir_uids)

    variants = [
        VariantRun(code="A", name="base_no_repo", use_repo=False, repo_policy=str(args.repo_policy), reranker_off=False),
        VariantRun(code="B", name="repo_query_aware", use_repo=True, repo_policy=str(args.repo_policy), reranker_off=False),
        VariantRun(code="C", name="base_no_repo_reranker_off", use_repo=False, repo_policy=str(args.repo_policy), reranker_off=True),
        VariantRun(code="D", name="repo_query_aware_reranker_off", use_repo=True, repo_policy=str(args.repo_policy), reranker_off=True),
    ]

    print(f"selection_mode={selection.get('selection_mode')}")
    print(f"selected_uids={len(selected_uids)}")
    print(f"budgets={len(budgets)}")
    print(f"variants={','.join(v.code for v in variants)}")

    rerank_off_cfg = _build_reranker_off_config(out_dir / "configs" / "reranker_off.json")
    python_bin = sys.executable
    eval_script = Path(args.eval_script)
    index_dir = Path(args.index_dir) if args.index_dir else None

    # run outputs keyed by (variant_code, uid, budget_key)
    run_status: dict[tuple[str, str, str], str] = {}
    run_metrics: dict[tuple[str, str, str], dict[str, float]] = {}

    for variant in variants:
        run_dir = out_dir / f"run_{variant.code}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for budget in budgets:
            for uid in selected_uids:
                json_path = discovered[uid]
                per_out = run_dir / "per_budget" / budget.tag / uid
                per_out.mkdir(parents=True, exist_ok=True)
                cmd = [
                    python_bin,
                    str(eval_script),
                    "--json",
                    str(json_path),
                    "--out_dir",
                    str(per_out),
                    "--mode",
                    str(args.mode),
                    "--n",
                    str(int(args.queries_total)),
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
                    "--no-allow-gt-fallback",
                    "--no-safety-gate",
                ]
                prefix = _index_prefix_for_uid(uid, index_dir=index_dir, json_dir=pov_json_dir)
                if prefix:
                    cmd.extend(["--index", str(prefix)])
                if variant.reranker_off:
                    cmd.extend(["--rerank-cfg", str(rerank_off_cfg)])
                rc = _run(
                    cmd,
                    cwd=ROOT,
                    log_prefix=logs_dir / f"run_{variant.code}_{budget.tag}_{uid}",
                    commands_file=commands_file,
                )
                key = (variant.code, uid, budget.key)
                run_status[key] = "ok" if rc == 0 else "failed"
                metrics = _extract_metrics(per_out / "nlq_summary.csv", per_out / "safety_report.json")
                metrics["use_repo"] = 1.0 if variant.use_repo else 0.0
                run_metrics[key] = metrics

    # aggregate rows (budget, variant)
    agg_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for budget in budgets:
        baseline_row: dict[str, Any] | None = None
        per_variant_rows: dict[str, dict[str, Any]] = {}
        for variant in variants:
            vals = [run_metrics.get((variant.code, uid, budget.key), {}) for uid in selected_uids]
            statuses = [run_status.get((variant.code, uid, budget.key), "missing") for uid in selected_uids]
            row = {
                "budget_key": budget.key,
                "budget_seconds": float(budget.max_total_s),
                "variant_code": variant.code,
                "variant_name": variant.name,
                "use_repo": bool(variant.use_repo),
                "repo_policy": str(variant.repo_policy),
                "reranker_mode": "off" if variant.reranker_off else "default",
                "uids_total": int(len(selected_uids)),
                "runs_ok": int(sum(1 for s in statuses if s == "ok")),
                "chain_success_rate": _mean([float(v.get("chain_success_rate", 0.0)) for v in vals]),
                "chain_waiting_rate": _mean([float(v.get("chain_waiting_rate", 0.0)) for v in vals]),
                "chain_fail_constraints_over_filtered_rate": _mean(
                    [float(v.get("chain_fail_constraints_over_filtered_rate", 0.0)) for v in vals]
                ),
                "chain_fail_retrieval_distractor_rate": _mean(
                    [float(v.get("chain_fail_retrieval_distractor_rate", 0.0)) for v in vals]
                ),
                "chain_fail_evidence_missing_rate": _mean(
                    [float(v.get("chain_fail_evidence_missing_rate", 0.0)) for v in vals]
                ),
                "mrr_strict": _mean([float(v.get("mrr_strict", 0.0)) for v in vals]),
                "hit_at_k_strict": _mean([float(v.get("hit_at_k_strict", 0.0)) for v in vals]),
                "top1_in_distractor_rate": _mean([float(v.get("top1_in_distractor_rate", 0.0)) for v in vals]),
                "critical_fn_rate": _mean([float(v.get("critical_fn_rate", 0.0)) for v in vals]),
            }
            per_variant_rows[variant.code] = row
            if variant.code == "A":
                baseline_row = row
        baseline = baseline_row or {}
        for variant in variants:
            row = per_variant_rows.get(variant.code, {})
            row["delta_success"] = float(row.get("chain_success_rate", 0.0)) - float(baseline.get("chain_success_rate", 0.0))
            row["delta_waiting"] = float(row.get("chain_waiting_rate", 0.0)) - float(baseline.get("chain_waiting_rate", 0.0))
            row["delta_fail_constraints_over_filtered"] = float(row.get("chain_fail_constraints_over_filtered_rate", 0.0)) - float(
                baseline.get("chain_fail_constraints_over_filtered_rate", 0.0)
            )
            row["delta_fail_retrieval_distractor"] = float(row.get("chain_fail_retrieval_distractor_rate", 0.0)) - float(
                baseline.get("chain_fail_retrieval_distractor_rate", 0.0)
            )
            row["delta_fail_evidence_missing"] = float(row.get("chain_fail_evidence_missing_rate", 0.0)) - float(
                baseline.get("chain_fail_evidence_missing_rate", 0.0)
            )
            agg_rows.append(row)
            failure_rows.append(
                {
                    "budget_key": budget.key,
                    "budget_seconds": float(budget.max_total_s),
                    "variant_code": variant.code,
                    "variant_name": variant.name,
                    "chain_fail_constraints_over_filtered_rate": row.get("chain_fail_constraints_over_filtered_rate", 0.0),
                    "chain_fail_retrieval_distractor_rate": row.get("chain_fail_retrieval_distractor_rate", 0.0),
                    "chain_fail_evidence_missing_rate": row.get("chain_fail_evidence_missing_rate", 0.0),
                }
            )

    table_cols = [
        "budget_key",
        "budget_seconds",
        "variant_code",
        "variant_name",
        "use_repo",
        "repo_policy",
        "reranker_mode",
        "uids_total",
        "runs_ok",
        "chain_success_rate",
        "chain_waiting_rate",
        "chain_fail_constraints_over_filtered_rate",
        "chain_fail_retrieval_distractor_rate",
        "chain_fail_evidence_missing_rate",
        "mrr_strict",
        "hit_at_k_strict",
        "top1_in_distractor_rate",
        "critical_fn_rate",
        "delta_success",
        "delta_waiting",
        "delta_fail_constraints_over_filtered",
        "delta_fail_retrieval_distractor",
        "delta_fail_evidence_missing",
    ]
    failure_cols = [
        "budget_key",
        "budget_seconds",
        "variant_code",
        "variant_name",
        "chain_fail_constraints_over_filtered_rate",
        "chain_fail_retrieval_distractor_rate",
        "chain_fail_evidence_missing_rate",
    ]
    table_csv = tables_dir / "table_chain_attribution.csv"
    table_md = tables_dir / "table_chain_attribution.md"
    fail_csv = tables_dir / "table_chain_failure_breakdown.csv"
    fail_md = tables_dir / "table_chain_failure_breakdown.md"
    _write_csv(table_csv, agg_rows, table_cols)
    _write_md(
        table_md,
        agg_rows,
        table_cols,
        title="Chain Attribution",
        summary_lines=[
            f"- selection_mode: {selection.get('selection_mode', '')}",
            f"- selected_uids: {len(selected_uids)}",
            f"- budgets: {len(budgets)}",
            "- deltas are against variant A (base_no_repo)",
            "- variant B/D mark repo-aware configuration labels for chain analysis",
        ],
    )
    _write_csv(fail_csv, failure_rows, failure_cols)
    _write_md(
        fail_md,
        failure_rows,
        failure_cols,
        title="Chain Failure Breakdown",
        summary_lines=["- grouped by budget and variant"],
    )
    figure_paths = _make_figures(agg_rows, figures_dir, formats=formats)

    matched_budgets = sorted(
        {
            str(row.get("budget_key", ""))
            for row in agg_rows
            if int(row.get("runs_ok", 0)) == int(len(selected_uids))
        }
    )
    compare_summary = {
        "selection": selection,
        "uids_total": len(selected_uids),
        "budgets_total": len(budgets),
        "budgets_matched": len(matched_budgets),
        "budget_keys_matched": matched_budgets,
        "variants": [
            {
                "code": v.code,
                "name": v.name,
                "use_repo": v.use_repo,
                "repo_policy": v.repo_policy,
                "reranker_mode": "off" if v.reranker_off else "default",
            }
            for v in variants
        ],
        "outputs": {
            "table_chain_attribution_csv": str(table_csv),
            "table_chain_attribution_md": str(table_md),
            "table_chain_failure_breakdown_csv": str(fail_csv),
            "table_chain_failure_breakdown_md": str(fail_md),
            "figures": figure_paths,
        },
    }
    compare_summary_path = compare_dir / "compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(pov_json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "budgets": [b.key for b in budgets],
            "mode": str(args.mode),
            "queries_total": int(args.queries_total),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "index_dir": str(index_dir) if index_dir else None,
            "eval_script": str(eval_script),
            "reranker_off_cfg": str(rerank_off_cfg),
        },
        "selection": selection,
        "variants": compare_summary["variants"],
        "outputs": compare_summary["outputs"],
        "commands_sh": str(commands_file),
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = compare_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Chain Attribution Panel",
                "",
                "## Variants",
                "- A: base_no_repo",
                "- B: repo_query_aware",
                "- C: base_no_repo_reranker_off",
                "- D: repo_query_aware_reranker_off",
                "",
                "## Outputs",
                f"- table: `{table_csv}`",
                f"- failure_table: `{fail_csv}`",
                f"- figures: `{figure_paths}`",
                f"- summary: `{compare_summary_path}`",
                f"- snapshot: `{snapshot_path}`",
                f"- commands: `{commands_file}`",
            ]
        ),
        encoding="utf-8",
    )

    for v in variants:
        print(f"saved_run_{v.code}={out_dir / f'run_{v.code}'}")
    print(f"saved_compare={compare_dir}")
    print(f"budgets_matched={len(matched_budgets)}")
    print(f"saved_table={[str(table_csv), str(table_md), str(fail_csv), str(fail_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
