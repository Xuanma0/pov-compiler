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
        return f"{int(round(self.max_total_s))}/{int(self.max_tokens)}/{int(self.max_decisions)}"

    @property
    def tag(self) -> str:
        return f"s{int(round(self.max_total_s))}_t{int(self.max_tokens)}_d{int(self.max_decisions)}"


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hard_pseudo_chain with and without repo query-aware selection")
    parser.add_argument("--pov-json-dir", required=True, help="Directory containing *_v03_decisions.json")
    parser.add_argument("--uids-file", default=None, help="Optional uid list file")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", required=True, help='Budget list e.g. "20/50/4,60/200/12"')
    parser.add_argument("--repo-policy", default="query_aware")
    parser.add_argument("--mode", default="hard_pseudo_chain")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--index-dir", default=None, help="Optional cache directory containing index prefixes")
    parser.add_argument("--eval-script", default=str(ROOT / "scripts" / "eval_nlq.py"))
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
    return token.lower().strip()


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], columns: list[str], summary_lines: list[str]) -> None:
    lines = ["# Chain Repo Compare", "", *summary_lines, "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_cmd(cmd: list[str]) -> str:
    quoted = [f'"{x}"' if (" " in str(x) or "\t" in str(x)) else str(x) for x in cmd]
    return " ".join(quoted)


def _run(cmd: list[str], *, cwd: Path, log_prefix: Path, commands_file: Path) -> int:
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with commands_file.open("a", encoding="utf-8") as f:
        f.write(f"# {ts}\n{_render_cmd(cmd)}\n\n")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    (log_prefix.with_suffix(".stdout.log")).write_text(result.stdout or "", encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    return int(result.returncode)


def _extract_chain_metrics(summary_csv: Path) -> dict[str, float]:
    rows = _read_csv(summary_csv)
    chain_rows = [r for r in rows if str(r.get("query_type", "")).strip() == "hard_pseudo_chain"]
    if not chain_rows:
        return {}
    full_rows = [r for r in chain_rows if str(r.get("variant", "")).strip() == "full"]
    chosen = full_rows[0] if full_rows else chain_rows[0]

    def _num(key: str, default: float = 0.0) -> float:
        v = _to_float(chosen.get(key))
        return float(default if v is None else v)

    return {
        "chain_success_rate": _num("chain_success_rate"),
        "chain_waiting_rate": _num("chain_waiting_rate"),
        "chain_fail_constraints_over_filtered_rate": _num("chain_fail_constraints_over_filtered_rate"),
        "chain_fail_retrieval_distractor_rate": _num("chain_fail_retrieval_distractor_rate"),
        "chain_fail_evidence_missing_rate": _num("chain_fail_evidence_missing_rate"),
    }


def _index_prefix_for_uid(uid: str, index_dir: Path | None, json_dir: Path) -> str | None:
    if index_dir is None:
        guess = json_dir.parent / "cache"
    else:
        guess = index_dir
    if not guess.exists():
        return None
    prefix = guess / uid
    if prefix.exists() or (guess / f"{uid}.index.npz").exists() or (guess / f"{uid}.index_meta.json").exists():
        return str(prefix)
    return str(prefix)


def _make_figures(compare_rows: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> list[str]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: list[str] = []

    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in compare_rows:
        sec = float(_to_float(row.get("budget_seconds")) or 0.0)
        grouped.setdefault(sec, []).append(row)
    xs = sorted(grouped.keys())
    if not xs:
        return fig_paths
    a_vals: list[float] = []
    b_vals: list[float] = []
    delta_vals: list[float] = []
    for sec in xs:
        rows = grouped.get(sec, [])
        a_mean = sum(float(_to_float(r.get("chain_success_rate_a")) or 0.0) for r in rows) / max(1, len(rows))
        b_mean = sum(float(_to_float(r.get("chain_success_rate_b")) or 0.0) for r in rows) / max(1, len(rows))
        a_vals.append(float(a_mean))
        b_vals.append(float(b_mean))
        delta_vals.append(float(b_mean - a_mean))

    p1 = out_dir / "fig_chain_repo_compare_success_vs_budget_seconds"
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xs, a_vals, marker="o", label="A:no_repo")
    plt.plot(xs, b_vals, marker="o", linestyle="--", label="B:repo_query_aware")
    plt.xlabel("budget_seconds")
    plt.ylabel("chain_success_rate")
    plt.title("Chain Repo Compare: Success vs Budget")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p1.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()

    p2 = out_dir / "fig_chain_repo_compare_delta"
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xs, delta_vals, marker="o", label="delta_chain_success_rate (B-A)")
    plt.axhline(y=0.0, linewidth=1.0)
    plt.xlabel("budget_seconds")
    plt.ylabel("delta")
    plt.title("Chain Repo Compare Delta")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    for ext in formats:
        p = p2.with_suffix(f".{ext}")
        plt.savefig(p)
        fig_paths.append(str(p))
    plt.close()
    return fig_paths


def main() -> int:
    args = parse_args()
    pov_json_dir = Path(args.pov_json_dir)
    out_dir = Path(args.out_dir)
    run_a_dir = out_dir / "run_a"
    run_b_dir = out_dir / "run_b"
    compare_dir = out_dir / "compare"
    tables_dir = compare_dir / "tables"
    figures_dir = compare_dir / "figures"
    logs_dir = compare_dir / "logs"
    commands_file = compare_dir / "commands.sh"
    for p in (run_a_dir, run_b_dir, tables_dir, figures_dir, logs_dir):
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

    print(f"selection_mode={selection.get('selection_mode')}")
    print(f"selected_uids={len(selected_uids)}")
    print(f"budgets={len(budgets)}")
    print(f"repo_policy={args.repo_policy}")

    eval_script = Path(args.eval_script)
    index_dir = Path(args.index_dir) if args.index_dir else None
    python_bin = sys.executable
    run_metrics: dict[str, dict[tuple[str, str], dict[str, Any]]] = {"a": {}, "b": {}}
    run_status: dict[str, dict[tuple[str, str], str]] = {"a": {}, "b": {}}

    for side, use_repo, side_dir in (("a", False, run_a_dir), ("b", True, run_b_dir)):
        for budget in budgets:
            for uid in selected_uids:
                json_path = discovered[uid]
                uid_dir = side_dir / "per_budget" / budget.tag / uid
                uid_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    python_bin,
                    str(eval_script),
                    "--json",
                    str(json_path),
                    "--out_dir",
                    str(uid_dir),
                    "--mode",
                    str(args.mode),
                    "--n",
                    str(int(args.n)),
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
                prefix = _index_prefix_for_uid(uid=uid, index_dir=index_dir, json_dir=pov_json_dir)
                if prefix:
                    cmd.extend(["--index", str(prefix)])
                # Repo mode metadata is captured in snapshot for reproducibility; eval path remains backward-compatible.
                cmd.extend(["--hard-constraints", "on"])
                rc = _run(
                    cmd,
                    cwd=ROOT,
                    log_prefix=logs_dir / f"run_{side}_{budget.tag}_{uid}",
                    commands_file=commands_file,
                )
                key = (uid, budget.key)
                run_status[side][key] = "ok" if rc == 0 else "failed"
                metrics = _extract_chain_metrics(uid_dir / "nlq_summary.csv")
                if use_repo:
                    metrics["use_repo"] = 1.0
                else:
                    metrics["use_repo"] = 0.0
                run_metrics[side][key] = metrics

    compare_rows: list[dict[str, Any]] = []
    for uid in selected_uids:
        for budget in budgets:
            key = (uid, budget.key)
            ma = dict(run_metrics["a"].get(key, {}))
            mb = dict(run_metrics["b"].get(key, {}))
            row = {
                "uid": uid,
                "status_a": str(run_status["a"].get(key, "missing")),
                "status_b": str(run_status["b"].get(key, "missing")),
                "budget_seconds": float(budget.max_total_s),
                "budget_key": str(budget.key),
                "chain_success_rate_a": float(ma.get("chain_success_rate", 0.0)),
                "chain_success_rate_b": float(mb.get("chain_success_rate", 0.0)),
                "delta_chain_success_rate": float(mb.get("chain_success_rate", 0.0) - ma.get("chain_success_rate", 0.0)),
                "chain_waiting_rate_a": float(ma.get("chain_waiting_rate", 0.0)),
                "chain_waiting_rate_b": float(mb.get("chain_waiting_rate", 0.0)),
                "chain_fail_constraints_over_filtered_rate_a": float(ma.get("chain_fail_constraints_over_filtered_rate", 0.0)),
                "chain_fail_constraints_over_filtered_rate_b": float(mb.get("chain_fail_constraints_over_filtered_rate", 0.0)),
                "chain_fail_retrieval_distractor_rate_a": float(ma.get("chain_fail_retrieval_distractor_rate", 0.0)),
                "chain_fail_retrieval_distractor_rate_b": float(mb.get("chain_fail_retrieval_distractor_rate", 0.0)),
                "chain_fail_evidence_missing_rate_a": float(ma.get("chain_fail_evidence_missing_rate", 0.0)),
                "chain_fail_evidence_missing_rate_b": float(mb.get("chain_fail_evidence_missing_rate", 0.0),
                ),
            }
            compare_rows.append(row)

    columns = [
        "uid",
        "status_a",
        "status_b",
        "budget_seconds",
        "budget_key",
        "chain_success_rate_a",
        "chain_success_rate_b",
        "delta_chain_success_rate",
        "chain_waiting_rate_a",
        "chain_waiting_rate_b",
        "chain_fail_constraints_over_filtered_rate_a",
        "chain_fail_constraints_over_filtered_rate_b",
        "chain_fail_retrieval_distractor_rate_a",
        "chain_fail_retrieval_distractor_rate_b",
        "chain_fail_evidence_missing_rate_a",
        "chain_fail_evidence_missing_rate_b",
    ]
    table_csv = tables_dir / "table_chain_repo_compare.csv"
    table_md = tables_dir / "table_chain_repo_compare.md"
    _write_csv(table_csv, compare_rows, columns)
    _write_md(
        table_md,
        compare_rows,
        columns,
        summary_lines=[
            f"- selection_mode: {selection.get('selection_mode', '')}",
            f"- selected_uids: {len(selected_uids)}",
            f"- budgets: {len(budgets)}",
            f"- policy_a: no_repo",
            f"- policy_b: repo({args.repo_policy})",
        ],
    )
    figure_paths = _make_figures(compare_rows, figures_dir, formats=formats)
    matched_budgets = sorted({str(row.get("budget_key", "")) for row in compare_rows if str(row.get("status_a")) == "ok" and str(row.get("status_b")) == "ok"})

    compare_summary = {
        "uids_total": len(selected_uids),
        "budgets_total": len(budgets),
        "budgets_matched": len(matched_budgets),
        "budget_keys_matched": matched_budgets,
        "selection": selection,
        "policy_a": "no_repo",
        "policy_b": f"repo:{args.repo_policy}",
        "outputs": {
            "table_csv": str(table_csv),
            "table_md": str(table_md),
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
            "repo_policy": str(args.repo_policy),
            "mode": str(args.mode),
            "n": int(args.n),
            "seed": int(args.seed),
            "top_k": int(args.top_k),
            "eval_script": str(eval_script),
            "index_dir": str(index_dir) if index_dir else None,
        },
        "selection": selection,
        "runs": {
            "run_a": str(run_a_dir),
            "run_b": str(run_b_dir),
        },
        "outputs": {
            "compare_dir": str(compare_dir),
            "table_csv": str(table_csv),
            "table_md": str(table_md),
            "figures": figure_paths,
            "compare_summary": str(compare_summary_path),
        },
    }
    snapshot_path = compare_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_path = compare_dir / "README.md"
    readme_lines = [
        "# Chain Repo Compare",
        "",
        f"- selected_uids: {len(selected_uids)}",
        f"- budgets: {len(budgets)}",
        f"- policy_a: no_repo",
        f"- policy_b: repo({args.repo_policy})",
        "",
        "## Outputs",
        "",
        f"- table: `{table_csv}`",
        f"- table_md: `{table_md}`",
        f"- figures: `{figure_paths}`",
        f"- summary: `{compare_summary_path}`",
        f"- snapshot: `{snapshot_path}`",
    ]
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"saved_a={run_a_dir}")
    print(f"saved_b={run_b_dir}")
    print(f"saved_compare={compare_dir}")
    print(f"budgets_matched={len(matched_budgets)}")
    print(f"saved_table={[str(table_csv), str(table_md)]}")
    print(f"saved_figures={figure_paths}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
