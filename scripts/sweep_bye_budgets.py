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
    def tag(self) -> str:
        total = int(round(float(self.max_total_s)))
        return f"s{total}_t{int(self.max_tokens)}_d{int(self.max_decisions)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_total_s": float(self.max_total_s),
            "max_tokens": int(self.max_tokens),
            "max_decisions": int(self.max_decisions),
            "tag": self.tag,
        }


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BYE report across multiple budget points")
    parser.add_argument("--pov-json-dir", required=True, help="Directory containing *_v03_decisions.json files")
    parser.add_argument("--uids-file", default=None, help="Optional UID list file (one uid per line)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--bye-root", default=None, help="Optional BYE root (fallback to BYE_ROOT env if omitted)")
    parser.add_argument("--budgets", required=True, help='Budget list, e.g. "20/50/4,40/100/8,60/200/12"')
    parser.add_argument("--primary-metric", default="qualityScore", help="Primary metric used in quality figure")
    _parse_bool_with_neg(parser, "strict-uids", default=True)
    parser.add_argument(
        "--allow-fallback-all-uids",
        action="store_true",
        help="Only valid with --no-strict-uids. Allow fallback to all json UIDs when no uid is matched.",
    )
    _parse_bool_with_neg(parser, "skip-lint", default=True)
    _parse_bool_with_neg(parser, "skip-report", default=False)
    _parse_bool_with_neg(parser, "skip-regression", default=True)
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated figure formats")
    return parser.parse_args()


def _normalize_uid(text: str) -> str:
    token = str(text).replace("\ufeff", "").strip()
    if not token:
        return ""
    if token.lower().endswith(".mp4"):
        token = token[:-4]
    return token.lower().strip()


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


def _discover_json_by_uid(pov_json_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    files = sorted(pov_json_dir.glob("*_v03_decisions.json"), key=lambda p: p.name.lower())
    if not files:
        files = sorted(pov_json_dir.glob("*.json"), key=lambda p: p.name.lower())
    for path in files:
        uid = _normalize_uid(uid_from_pov_json_path(path))
        if uid and uid not in out:
            out[uid] = path
    return out


def _parse_budgets(raw: str) -> list[BudgetPoint]:
    out: list[BudgetPoint] = []
    for chunk in str(raw).split(","):
        part = chunk.strip()
        if not part:
            continue
        pieces = [x.strip() for x in part.split("/") if x.strip()]
        if len(pieces) != 3:
            raise ValueError(f"invalid budget chunk: {part}")
        total_s = float(pieces[0])
        tokens = int(pieces[1])
        decisions = int(pieces[2])
        out.append(BudgetPoint(max_total_s=total_s, max_tokens=tokens, max_decisions=decisions))
    if not out:
        raise ValueError("no budgets parsed")
    return out


def _to_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if x != x:
        return None
    return x


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, rows: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        lines = [
            "# BYE Budget Sweep",
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
            "No rows.",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [
        "# BYE Budget Sweep",
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


def _select_primary_metric(aggregate_rows: list[dict[str, Any]], requested: str) -> str | None:
    if not aggregate_rows:
        return None
    keys = list(aggregate_rows[0].keys())
    if requested in keys:
        return requested
    reserved = {
        "budget_tag",
        "budget_max_total_s",
        "budget_max_tokens",
        "budget_max_decisions",
        "num_uids",
        "runs_ok",
        "runs_total",
    }
    numeric = [k for k in keys if k not in reserved and isinstance(aggregate_rows[0].get(k), (int, float))]
    return sorted(numeric)[0] if numeric else None


def _select_critical_metric(aggregate_rows: list[dict[str, Any]]) -> str | None:
    if not aggregate_rows:
        return None
    keys = list(aggregate_rows[0].keys())
    candidates = [k for k in keys if "critical" in k.lower() and ("fn" in k.lower() or "false_negative" in k.lower())]
    if not candidates:
        candidates = [k for k in keys if "critical_fn" in k.lower() or "criticalfn" in k.lower()]
    if not candidates:
        return None
    return sorted(candidates)[0]


def _make_figure(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    out_prefix: Path,
    formats: list[str],
) -> list[str]:
    import matplotlib.pyplot as plt

    sorted_rows = sorted(rows, key=lambda r: float(r.get("budget_max_total_s", 0.0)))
    xs = [float(r.get("budget_max_total_s", 0.0)) for r in sorted_rows]
    ys = [float(r.get(metric, 0.0)) for r in sorted_rows]
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Budget Max Total Seconds")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    out_files: list[str] = []
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        suffix = ext.strip().lower()
        if not suffix:
            continue
        out_path = out_prefix.with_suffix(f".{suffix}")
        plt.tight_layout()
        plt.savefig(out_path)
        out_files.append(str(out_path))
    plt.close()
    return out_files


def main() -> int:
    args = parse_args()
    pov_json_dir = Path(args.pov_json_dir)
    out_dir = Path(args.out_dir)
    run_root = out_dir / "runs"
    agg_dir = out_dir / "aggregate"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    budgets = _parse_budgets(args.budgets)
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]

    discovered = _discover_json_by_uid(pov_json_dir)
    if not discovered:
        raise FileNotFoundError(f"no json files found under {pov_json_dir}")

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
        uids_file_path = Path(args.uids_file)
        requested = _read_uids_file(uids_file_path)
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
                "uids_file_path": str(uids_file_path),
                "uids_requested": len(requested),
                "uids_found": len(selected_uids),
                "uids_missing_count": len(missing),
                "uids_missing_sample": missing[:10],
            }
        )
        if bool(args.strict_uids):
            if not selected_uids or missing:
                print("error=uid selection failed under strict mode")
                print(f"uids_file_path={uids_file_path}")
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
                    print(f"uids_file_path={uids_file_path}")
                    print(f"uids_requested_sample={requested[:10]}")
                    print(f"dir_uid_sample={dir_uids[:5]}")
                    return 2
            elif missing:
                selection["selection_mode"] = "uids_file_partial"
    else:
        selected_uids = list(dir_uids)
        selection["uids_found"] = len(selected_uids)
    if not selected_uids:
        print("error=no selected uids to run")
        return 2
    if bool(args.allow_fallback_all_uids) and bool(args.strict_uids):
        print("error=--allow-fallback-all-uids requires --no-strict-uids")
        return 2

    print(f"selection_mode={selection.get('selection_mode')}")
    print(f"uids_file_path={selection.get('uids_file_path')}")
    print(f"uids_requested={selection.get('uids_requested')}")
    print(f"uids_found={selection.get('uids_found')}")
    print(f"uids_missing_count={selection.get('uids_missing_count')}")
    print(f"uids_missing_sample={selection.get('uids_missing_sample')}")
    print(f"dir_uids_sample={selection.get('dir_uids_sample')}")

    python_bin = sys.executable
    smoke_script = ROOT / "scripts" / "bye_regression_smoke.py"
    all_run_rows: list[dict[str, Any]] = []
    run_meta: list[dict[str, Any]] = []
    resolved_entrypoints: dict[str, Any] | None = None

    for budget in budgets:
        budget_dir = run_root / budget.tag
        for uid in selected_uids:
            uid_norm = _normalize_uid(uid)
            json_path = discovered[uid_norm]
            uid_dir = budget_dir / uid_norm
            uid_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                python_bin,
                str(smoke_script),
                "--pov_json",
                str(json_path),
                "--video_id",
                uid_norm,
                "--out_dir",
                str(uid_dir),
                "--budget-mode",
                "filter",
                "--budget-max-total-s",
                str(float(budget.max_total_s)),
                "--budget-max-tokens",
                str(int(budget.max_tokens)),
                "--budget-max-decisions",
                str(int(budget.max_decisions)),
            ]
            if args.bye_root:
                cmd.extend(["--bye_root", str(args.bye_root)])
            if args.skip_lint:
                cmd.append("--skip_lint")
            if args.skip_report:
                cmd.append("--skip_report")
            if args.skip_regression:
                cmd.append("--skip_regression")

            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
            (uid_dir / "runner.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (uid_dir / "runner.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

            snapshot_path = uid_dir / "snapshot.json"
            snap: dict[str, Any] = {}
            if snapshot_path.exists():
                try:
                    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
                except Exception:
                    snap = {}
            if resolved_entrypoints is None:
                bye_info = snap.get("bye", {}) if isinstance(snap, dict) else {}
                if isinstance(bye_info, dict):
                    resolved_entrypoints = {
                        "root": bye_info.get("root"),
                        "entrypoints_resolved": bye_info.get("entrypoints_resolved"),
                        "entrypoints_method": bye_info.get("entrypoints_method"),
                    }

            metrics_csv = uid_dir / "bye_metrics.csv"
            metrics_rows = _read_csv(metrics_csv)
            metrics = metrics_rows[0] if metrics_rows else {}

            row: dict[str, Any] = {
                "video_uid": uid_norm,
                "budget_tag": budget.tag,
                "budget_max_total_s": float(budget.max_total_s),
                "budget_max_tokens": int(budget.max_tokens),
                "budget_max_decisions": int(budget.max_decisions),
                "returncode": int(proc.returncode),
                "bye_status": str(metrics.get("status", "")),
            }
            # Include numeric metrics from BYE parser.
            for key, value in metrics.items():
                if key in {"status", "report_path", "summary_keys"}:
                    continue
                num = _to_float(value)
                if num is not None:
                    row[str(key)] = float(num)

            budget_stats = snap.get("budget", {}) if isinstance(snap, dict) else {}
            if isinstance(budget_stats, dict):
                if "kept_duration_s" in budget_stats:
                    row["kept_duration_s"] = _to_float(budget_stats.get("kept_duration_s"))
                if "compression_ratio" in budget_stats:
                    row["compression_ratio"] = _to_float(budget_stats.get("compression_ratio"))
                row["before_total"] = budget_stats.get("before_total")
                row["after_total"] = budget_stats.get("after_total")

            bye_steps = (snap.get("bye", {}) or {}).get("steps", []) if isinstance(snap, dict) else []
            report_rc = None
            if isinstance(bye_steps, list):
                for step in bye_steps:
                    if isinstance(step, dict) and step.get("tool") == "report":
                        report_rc = step.get("returncode")
                        break
            row["bye_report_rc"] = report_rc
            all_run_rows.append(row)

            run_meta.append(
                {
                    "uid": uid_norm,
                    "budget": budget.as_dict(),
                    "returncode": int(proc.returncode),
                    "snapshot": str(snapshot_path),
                    "metrics_csv": str(metrics_csv),
                    "stdout_log": str(uid_dir / "runner.stdout.log"),
                    "stderr_log": str(uid_dir / "runner.stderr.log"),
                }
            )

    # Per-budget aggregate (macro over selected uids).
    by_budget: dict[str, list[dict[str, Any]]] = {}
    for row in all_run_rows:
        by_budget.setdefault(str(row.get("budget_tag", "")), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for budget in budgets:
        rows = by_budget.get(budget.tag, [])
        agg: dict[str, Any] = {
            "budget_tag": budget.tag,
            "budget_max_total_s": float(budget.max_total_s),
            "budget_max_tokens": int(budget.max_tokens),
            "budget_max_decisions": int(budget.max_decisions),
            "num_uids": len(selected_uids),
            "runs_total": len(rows),
            "runs_ok": sum(1 for r in rows if int(r.get("returncode", 1)) == 0),
            "selection_mode": str(selection.get("selection_mode", "")),
            "uids_requested": int(selection.get("uids_requested", 0)),
            "uids_found": int(selection.get("uids_found", 0)),
            "uids_missing_count": int(selection.get("uids_missing_count", 0)),
        }
        numeric_keys: set[str] = set()
        for r in rows:
            for k, v in r.items():
                if k in {"video_uid", "budget_tag", "bye_status"}:
                    continue
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)
        for key in sorted(numeric_keys):
            vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                agg[key] = float(sum(vals) / len(vals))
        aggregate_rows.append(agg)

    metrics_csv_path = agg_dir / "metrics_by_budget.csv"
    metrics_md_path = agg_dir / "metrics_by_budget.md"
    _write_csv(metrics_csv_path, aggregate_rows)
    _write_md(metrics_md_path, aggregate_rows, selection)

    primary_metric = _select_primary_metric(aggregate_rows, requested=str(args.primary_metric))
    critical_metric = _select_critical_metric(aggregate_rows)

    figure_paths: list[str] = []
    if primary_metric:
        figure_paths.extend(
            _make_figure(
                rows=aggregate_rows,
                metric=primary_metric,
                title=f"BYE {primary_metric} vs Budget Seconds",
                out_prefix=fig_dir / "fig_bye_quality_vs_budget_seconds",
                formats=formats,
            )
        )
    if critical_metric:
        figure_paths.extend(
            _make_figure(
                rows=aggregate_rows,
                metric=critical_metric,
                title=f"BYE {critical_metric} vs Budget Seconds",
                out_prefix=fig_dir / "fig_bye_critical_fn_vs_budget_seconds",
                formats=formats,
            )
        )

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pov_json_dir": str(pov_json_dir),
            "uids_file": str(args.uids_file) if args.uids_file else None,
            "bye_root": str(args.bye_root) if args.bye_root else None,
            "budgets": [x.as_dict() for x in budgets],
            "primary_metric_requested": str(args.primary_metric),
            "formats": formats,
            "skip_lint": bool(args.skip_lint),
            "skip_report": bool(args.skip_report),
            "skip_regression": bool(args.skip_regression),
            "strict_uids": bool(args.strict_uids),
            "allow_fallback_all_uids": bool(args.allow_fallback_all_uids),
        },
        "selected_uids": list(selected_uids),
        "selection": selection,
        "primary_metric": primary_metric,
        "critical_fn_metric": critical_metric,
        "bye": {
            "entrypoints": resolved_entrypoints,
        },
        "runs": run_meta,
        "outputs": {
            "metrics_by_budget_csv": str(metrics_csv_path),
            "metrics_by_budget_md": str(metrics_md_path),
            "figures": figure_paths,
        },
    }
    snapshot_path = out_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected_uids={len(selected_uids)}")
    print(f"budgets={len(budgets)}")
    print(f"saved_metrics_csv={metrics_csv_path}")
    print(f"saved_metrics_md={metrics_md_path}")
    for fig in figure_paths:
        print(f"saved_figure={fig}")
    print(f"saved_snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
