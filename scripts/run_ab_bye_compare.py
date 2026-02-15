from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_bool_with_neg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible AB runner (stub vs real) with optional BYE/NLQ/Figs compare")
    parser.add_argument("--root", required=True, help="Ego root")
    parser.add_argument("--uids-file", default=None, help="Optional UID list file for reproducible AB")
    parser.add_argument("--out_dir", required=True, help="Output root")
    parser.add_argument("--run-label", default="", help="Optional run label for trace/snapshot naming")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--n", type=int, default=None)

    _parse_bool_with_neg(parser, "with-eval", default=False)
    _parse_bool_with_neg(parser, "with-nlq", default=False)
    parser.add_argument(
        "--nlq-mode",
        choices=["mock", "pseudo_nlq", "hard_pseudo_nlq", "hard_pseudo_chain", "ego4d"],
        default="hard_pseudo_nlq",
    )
    nlq_budget_group = parser.add_mutually_exclusive_group()
    nlq_budget_group.add_argument("--with-nlq-budget-sweep", dest="with_nlq_budget_sweep", action="store_true")
    nlq_budget_group.add_argument("--no-with-nlq-budget-sweep", dest="with_nlq_budget_sweep", action="store_false")
    # If unspecified, auto-follow --with-nlq.
    parser.set_defaults(with_nlq_budget_sweep=None)
    parser.add_argument("--nlq-budgets", default="20/50/4,40/100/8,60/200/12")
    _parse_bool_with_neg(parser, "with-streaming-budget", default=True)
    parser.add_argument("--streaming-step-s", type=float, default=8.0)
    _parse_bool_with_neg(parser, "with-streaming-chain-backoff", default=False)
    parser.add_argument("--streaming-chain-backoff-budgets", default=None)
    parser.add_argument("--streaming-chain-backoff-step-s", type=float, default=None)
    parser.add_argument("--streaming-chain-backoff-policies", default="strict,ladder,adaptive")
    parser.add_argument("--streaming-chain-backoff-seed", type=int, default=None)
    _parse_bool_with_neg(parser, "with-repo", default=False)
    parser.add_argument("--repo-read-policy", default="query_aware")
    parser.add_argument("--repo-budget", default=None)
    _parse_bool_with_neg(parser, "with-figs", default=False)
    _parse_bool_with_neg(parser, "export-paper-ready", default=False)
    parser.add_argument("--paper-ready-format", choices=["md", "csv", "md+csv"], default="md+csv")
    _parse_bool_with_neg(parser, "with-reranker-sweep", default=False)
    parser.add_argument("--reranker-sweep-grid", default="", help="Optional grid for sweep_reranker.py")

    _parse_bool_with_neg(parser, "with-perception", default=False)
    parser.add_argument("--stub-perception-backend", choices=["stub", "real"], default="stub")
    parser.add_argument("--real-perception-backend", choices=["stub", "real"], default="real")
    parser.add_argument("--perception-fps", type=float, default=5.0)
    parser.add_argument("--perception-max-frames", type=int, default=300)
    _parse_bool_with_neg(parser, "real-perception-strict", default=True)
    _parse_bool_with_neg(parser, "stub-perception-strict", default=False)

    _parse_bool_with_neg(parser, "with-bye", default=False)
    _parse_bool_with_neg(parser, "with-bye-report", default=True)
    parser.add_argument("--bye-root", default=None)
    _parse_bool_with_neg(parser, "bye-gate", default=False)
    parser.add_argument("--max-bye-critical-fn", type=float, default=999.0)
    _parse_bool_with_neg(parser, "bye-skip-regression", default=True)
    bye_budget_group = parser.add_mutually_exclusive_group()
    bye_budget_group.add_argument("--with-bye-budget-sweep", dest="with_bye_budget_sweep", action="store_true")
    bye_budget_group.add_argument("--no-with-bye-budget-sweep", dest="with_bye_budget_sweep", action="store_false")
    # Backward-compatible alias.
    bye_budget_group.add_argument("--bye-budget-sweep", dest="with_bye_budget_sweep", action="store_true")
    parser.set_defaults(with_bye_budget_sweep=False)
    parser.add_argument("--bye-budgets", default="20/50/4,40/100/8,60/200/12")
    parser.add_argument("--bye-primary-metric", default="qualityScore")
    parser.add_argument("--bye-video-mode", choices=["none", "copy", "link"], default="none")
    parser.add_argument("--bye-lint", default=None)
    parser.add_argument("--bye-report", default=None)
    parser.add_argument("--bye-regression", default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if BYE missing when --with-bye")
    _parse_bool_with_neg(parser, "with-budget-recommend", default=False)
    parser.add_argument("--budget-weights-json", default=None)
    parser.add_argument("--budget-gates-json", default=None)
    parser.add_argument("--nlq-eval-script", default=None, help="Optional override for eval_nlq.py path (testing)")

    parser.add_argument("--prefer-short", action="store_true")
    parser.add_argument("--prefer-long", action="store_true")
    parser.add_argument("--min-duration-s", type=float, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--probe-candidates", type=int, default=50)
    parser.add_argument("--min-size-bytes", type=int, default=0)
    return parser.parse_args()


def _render_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _run(cmd: list[str], *, cwd: Path, log_prefix: Path, commands_file: Path) -> int:
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    commands_file.parent.mkdir(parents=True, exist_ok=True)
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


def _build_smoke_cmd(
    *,
    out_dir: Path,
    root: str,
    uids_file: str | None,
    jobs: int,
    n: int | None,
    with_eval: bool,
    with_nlq: bool,
    nlq_mode: str,
    with_perception: bool,
    perception_backend: str,
    perception_strict: bool,
    perception_fps: float,
    perception_max_frames: int,
    with_bye: bool,
    with_bye_report: bool,
    bye_root: str | None,
    bye_gate: bool,
    max_bye_critical_fn: float,
    bye_skip_regression: bool,
    bye_video_mode: str,
    bye_lint: str | None,
    bye_report: str | None,
    bye_regression: str | None,
    strict: bool,
    prefer_short: bool,
    prefer_long: bool,
    min_duration_s: float | None,
    max_duration_s: float | None,
    probe_candidates: int,
    min_size_bytes: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ego4d_smoke.py"),
        "--root",
        str(root),
        "--out_dir",
        str(out_dir),
        "--jobs",
        str(int(jobs)),
        "--min-size-bytes",
        str(int(min_size_bytes)),
        "--probe-candidates",
        str(int(probe_candidates)),
        "--no-proxy",
        "--resume",
    ]
    if uids_file:
        cmd.extend(["--uids-file", str(uids_file)])
    if n is not None:
        cmd.extend(["--n", str(int(n))])
    if prefer_short:
        cmd.append("--prefer-short")
    if prefer_long:
        cmd.append("--prefer-long")
    if min_duration_s is not None:
        cmd.extend(["--min-duration-s", str(float(min_duration_s))])
    if max_duration_s is not None:
        cmd.extend(["--max-duration-s", str(float(max_duration_s))])

    cmd.append("--run-eval" if with_eval else "--no-run-eval")
    if with_nlq:
        cmd.extend(["--run-nlq", "--nlq-mode", str(nlq_mode)])
    else:
        cmd.append("--no-run-nlq")

    if with_perception:
        cmd.extend(
            [
                "--run-perception",
                "--perception-backend",
                str(perception_backend),
                "--perception-fps",
                str(float(perception_fps)),
                "--perception-max-frames",
                str(int(perception_max_frames)),
            ]
        )
        if perception_strict:
            cmd.append("--perception-strict")
    else:
        cmd.append("--no-run-perception")

    if with_bye:
        cmd.append("--run-bye")
        cmd.extend(["--bye-video-mode", str(bye_video_mode)])
        cmd.append("--bye-collect-report" if with_bye_report else "--no-bye-collect-report")
        cmd.append("--bye-gate" if bye_gate else "--no-bye-gate")
        cmd.extend(["--max-bye-critical-fn", str(float(max_bye_critical_fn))])
        if bye_root:
            cmd.extend(["--bye-root", str(bye_root)])
        if bye_skip_regression:
            cmd.append("--bye-skip-regression")
        if bye_lint:
            cmd.extend(["--bye-lint", str(bye_lint)])
        if bye_report:
            cmd.extend(["--bye-report", str(bye_report)])
        if bye_regression:
            cmd.extend(["--bye-regression", str(bye_regression)])
        if strict:
            cmd.append("--bye-strict")
    else:
        cmd.append("--no-run-bye")
    return cmd


def _copy_snapshots(run_dir: Path, label: str, snapshots_root: Path) -> list[str]:
    out: list[str] = []
    target_root = snapshots_root / label
    target_root.mkdir(parents=True, exist_ok=True)
    hits = sorted(run_dir.rglob("snapshot.json"), key=lambda p: str(p).lower())
    if not hits:
        (target_root / "none.txt").write_text("no snapshot.json found\n", encoding="utf-8")
        return out
    for src in hits:
        rel = src.relative_to(run_dir)
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        out.append(str(dst))
    return out


def _pick_streaming_json(preferred_json_dir: Path, fallback_json_dir: Path, uids_file: Path | None) -> Path | None:
    def _scan(dir_path: Path) -> list[Path]:
        return sorted(dir_path.glob("*_v03_decisions.json"), key=lambda p: p.name.lower()) if dir_path.exists() else []

    preferred_hits = _scan(preferred_json_dir)
    fallback_hits = _scan(fallback_json_dir)
    if uids_file and uids_file.exists():
        requested_uids = [x.strip() for x in uids_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        for uid in requested_uids:
            for base_dir in (preferred_json_dir, fallback_json_dir):
                cand = base_dir / f"{uid}_v03_decisions.json"
                if cand.exists():
                    return cand
    if preferred_hits:
        return preferred_hits[0]
    if fallback_hits:
        return fallback_hits[0]
    return None


def _write_compare_readme(
    out_dir: Path,
    *,
    run_label: str,
    uids_file: Path | None,
    cmd_stub: list[str],
    cmd_real: list[str],
    cmd_bye_compare: list[str],
    cmd_bye_report_compare: list[str] | None,
    cmd_bye_budget: list[list[str]],
    cmd_nlq_budget: list[list[str]],
    cmd_streaming_budget: list[list[str]],
    cmd_streaming_chain_backoff: list[list[str]],
    cmd_reranker_sweep: list[list[str]],
    cmd_budget_recommend: list[list[str]],
    cmd_paper_ready: list[list[str]],
    fig_cmds: list[list[str]],
) -> None:
    lines = [
        "# AB v1.2 Compare",
        "",
        f"- run_label: `{run_label}`",
        f"- uids_file: `{uids_file}`",
        f"- run_stub_dir: `{out_dir / 'run_stub'}`",
        f"- run_real_dir: `{out_dir / 'run_real'}`",
        f"- compare_dir: `{out_dir / 'compare'}`",
        "",
        "## Commands",
        "",
        "```text",
        _render_cmd(cmd_stub),
        _render_cmd(cmd_real),
        _render_cmd(cmd_bye_compare),
    ]
    if cmd_bye_report_compare:
        lines.append(_render_cmd(cmd_bye_report_compare))
    for cmd in cmd_bye_budget:
        lines.append(_render_cmd(cmd))
    for cmd in cmd_nlq_budget:
        lines.append(_render_cmd(cmd))
    for cmd in cmd_streaming_budget:
        lines.append(_render_cmd(cmd))
    for cmd in cmd_streaming_chain_backoff:
        lines.append(_render_cmd(cmd))
    for cmd in cmd_reranker_sweep:
        lines.append(_render_cmd(cmd))
    for cmd in cmd_budget_recommend:
        lines.append(_render_cmd(cmd))
    for cmd in cmd_paper_ready:
        lines.append(_render_cmd(cmd))
    for cmd in fig_cmds:
        lines.append(_render_cmd(cmd))
    lines.extend(
        [
            "```",
            "",
            "## Outputs",
            "",
            f"- `{out_dir / 'run_stub' / 'summary.csv'}`",
            f"- `{out_dir / 'run_real' / 'summary.csv'}`",
            f"- `{out_dir / 'compare' / 'bye' / 'table_bye_compare.csv'}`",
            f"- `{out_dir / 'compare' / 'bye' / 'table_bye_compare.md'}`",
            f"- `{out_dir / 'compare' / 'bye' / 'compare_summary.json'}`",
            f"- `{out_dir / 'compare' / 'bye_report' / 'tables' / 'table_bye_report_compare.csv'}`",
            f"- `{out_dir / 'compare' / 'bye_report' / 'tables' / 'table_bye_report_compare.md'}`",
            f"- `{out_dir / 'compare' / 'bye_report' / 'figures' / 'fig_bye_critical_fn_delta.png'}`",
            f"- `{out_dir / 'compare' / 'bye_budget'}`",
            f"- `{out_dir / 'compare' / 'nlq_budget'}`",
            f"- `{out_dir / 'compare' / 'nlq_budget' / 'real' / 'aggregate' / 'table_lost_object_budget.csv'}`",
            f"- `{out_dir / 'compare' / 'streaming_budget'}`",
            f"- `{out_dir / 'compare' / 'streaming_chain_backoff'}`",
            f"- `{out_dir / 'compare' / 'reranker_sweep'}`",
            f"- `{out_dir / 'compare' / 'budget_recommend'}`",
            f"- `{out_dir / 'compare' / 'paper_ready'}`",
            f"- `{out_dir / 'compare' / 'commands.sh'}`",
            f"- `{out_dir / 'compare' / 'snapshots'}`",
            "",
            "## BYE Budget Compare",
            "",
            f"- table: `{out_dir / 'compare' / 'bye_budget' / 'compare' / 'tables' / 'table_budget_compare.md'}`",
            f"- curve: `{out_dir / 'compare' / 'bye_budget' / 'compare' / 'figures' / 'fig_bye_primary_vs_budget_seconds_compare.png'}`",
            f"- delta: `{out_dir / 'compare' / 'bye_budget' / 'compare' / 'figures' / 'fig_bye_primary_delta_vs_budget_seconds.png'}`",
            "",
            "## Budget Recommender",
            "",
            f"- stub table: `{out_dir / 'compare' / 'budget_recommend' / 'stub' / 'tables' / 'table_budget_recommend.md'}`",
            f"- real table: `{out_dir / 'compare' / 'budget_recommend' / 'real' / 'tables' / 'table_budget_recommend.md'}`",
            "- objective combines BYE and NLQ metrics with configurable weights and gate constraints.",
            "",
            "## Unified Budget Panel",
            "",
            f"- panel table: `{out_dir / 'compare' / 'paper_ready' / 'tables' / 'table_budget_panel.md'}`",
            f"- delta table: `{out_dir / 'compare' / 'paper_ready' / 'tables' / 'table_budget_panel_delta.md'}`",
            f"- primary curves: `{out_dir / 'compare' / 'paper_ready' / 'figures' / 'fig_budget_primary_vs_seconds_panel.png'}`",
            f"- NLQ safety curve: `{out_dir / 'compare' / 'nlq_budget' / 'real' / 'figures' / 'fig_nlq_critical_fn_rate_vs_budget_seconds.png'}`",
            f"- paper safety curve: `{out_dir / 'compare' / 'paper_ready' / 'figures' / 'fig_nlq_critical_fn_rate_vs_seconds.png'}`",
            "- nlq_budget now includes safety aggregation and failure attribution figures.",
            "- paper_ready auto carries NLQ safety curves when safety columns are present.",
            "",
            "## Reranker Sweep",
            "",
            f"- stub sweep: `{out_dir / 'compare' / 'reranker_sweep' / 'stub' / 'aggregate' / 'metrics_by_weights.csv'}`",
            f"- real sweep: `{out_dir / 'compare' / 'reranker_sweep' / 'real' / 'aggregate' / 'metrics_by_weights.csv'}`",
            "- includes decision-aligned score decomposition weights and strict+distractor objective.",
            "",
            "## Streaming Chain Backoff",
            "",
            f"- compare table: `{out_dir / 'compare' / 'streaming_chain_backoff' / 'compare' / 'tables' / 'table_streaming_chain_backoff_compare.csv'}`",
            f"- success curve: `{out_dir / 'compare' / 'streaming_chain_backoff' / 'compare' / 'figures' / 'fig_streaming_chain_backoff_success_vs_budget_seconds.png'}`",
            f"- delta figure: `{out_dir / 'compare' / 'streaming_chain_backoff' / 'compare' / 'figures' / 'fig_streaming_chain_backoff_delta.png'}`",
        ]
    )
    (out_dir / "compare" / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_nlq_budget_sweep = bool(args.with_nlq_budget_sweep) if args.with_nlq_budget_sweep is not None else bool(args.with_nlq)
    out_dir = Path(args.out_dir)
    run_stub = out_dir / "run_stub"
    run_real = out_dir / "run_real"
    compare_dir = out_dir / "compare"
    compare_bye_dir = compare_dir / "bye"
    compare_bye_report_dir = compare_dir / "bye_report"
    compare_bye_budget_stub = compare_dir / "bye_budget" / "stub"
    compare_bye_budget_real = compare_dir / "bye_budget" / "real"
    compare_bye_budget_compare = compare_dir / "bye_budget" / "compare"
    compare_nlq_budget_stub = compare_dir / "nlq_budget" / "stub"
    compare_nlq_budget_real = compare_dir / "nlq_budget" / "real"
    compare_streaming_budget_stub = compare_dir / "streaming_budget" / "stub"
    compare_streaming_budget_real = compare_dir / "streaming_budget" / "real"
    compare_streaming_chain_backoff = compare_dir / "streaming_chain_backoff"
    compare_reranker_sweep_stub = compare_dir / "reranker_sweep" / "stub"
    compare_reranker_sweep_real = compare_dir / "reranker_sweep" / "real"
    compare_budget_recommend_stub = compare_dir / "budget_recommend" / "stub"
    compare_budget_recommend_real = compare_dir / "budget_recommend" / "real"
    compare_budget_recommend_compare = compare_dir / "budget_recommend" / "compare"
    compare_paper_ready = compare_dir / "paper_ready"
    compare_snapshots = compare_dir / "snapshots"
    run_stub.mkdir(parents=True, exist_ok=True)
    run_real.mkdir(parents=True, exist_ok=True)
    compare_bye_dir.mkdir(parents=True, exist_ok=True)
    compare_bye_report_dir.mkdir(parents=True, exist_ok=True)
    compare_bye_budget_stub.mkdir(parents=True, exist_ok=True)
    compare_bye_budget_real.mkdir(parents=True, exist_ok=True)
    compare_bye_budget_compare.mkdir(parents=True, exist_ok=True)
    compare_nlq_budget_stub.mkdir(parents=True, exist_ok=True)
    compare_nlq_budget_real.mkdir(parents=True, exist_ok=True)
    compare_streaming_budget_stub.mkdir(parents=True, exist_ok=True)
    compare_streaming_budget_real.mkdir(parents=True, exist_ok=True)
    compare_streaming_chain_backoff.mkdir(parents=True, exist_ok=True)
    compare_reranker_sweep_stub.mkdir(parents=True, exist_ok=True)
    compare_reranker_sweep_real.mkdir(parents=True, exist_ok=True)
    compare_budget_recommend_stub.mkdir(parents=True, exist_ok=True)
    compare_budget_recommend_real.mkdir(parents=True, exist_ok=True)
    compare_budget_recommend_compare.mkdir(parents=True, exist_ok=True)
    compare_paper_ready.mkdir(parents=True, exist_ok=True)
    compare_snapshots.mkdir(parents=True, exist_ok=True)
    commands_file = compare_dir / "commands.sh"
    if not commands_file.exists():
        commands_file.write_text("#!/usr/bin/env text\n\n", encoding="utf-8")

    bye_root = str(args.bye_root) if args.bye_root else (str(Path.cwd()) if False else None)
    if args.with_bye and bye_root is None:
        env_value = os.environ.get("BYE_ROOT")
        if env_value:
            bye_root = env_value
    if args.with_bye and args.strict and not bye_root:
        print("error=with-bye strict mode requires --bye-root or BYE_ROOT env")
        return 2

    cmd_stub = _build_smoke_cmd(
        out_dir=run_stub,
        root=args.root,
        uids_file=args.uids_file,
        jobs=args.jobs,
        n=args.n,
        with_eval=args.with_eval,
        with_nlq=args.with_nlq,
        nlq_mode=args.nlq_mode,
        with_perception=args.with_perception,
        perception_backend=str(args.stub_perception_backend),
        perception_strict=bool(args.stub_perception_strict),
        perception_fps=args.perception_fps,
        perception_max_frames=args.perception_max_frames,
        with_bye=args.with_bye,
        with_bye_report=args.with_bye_report,
        bye_root=bye_root,
        bye_gate=args.bye_gate,
        max_bye_critical_fn=args.max_bye_critical_fn,
        bye_skip_regression=args.bye_skip_regression,
        bye_video_mode=args.bye_video_mode,
        bye_lint=args.bye_lint,
        bye_report=args.bye_report,
        bye_regression=args.bye_regression,
        strict=args.strict,
        prefer_short=args.prefer_short,
        prefer_long=args.prefer_long,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        probe_candidates=args.probe_candidates,
        min_size_bytes=args.min_size_bytes,
    )
    rc = _run(cmd_stub, cwd=ROOT, log_prefix=compare_dir / "run_stub", commands_file=commands_file)
    if rc != 0:
        return rc

    effective_uids_file: Path | None = Path(args.uids_file) if args.uids_file else None
    if effective_uids_file is None:
        summary_csv = run_stub / "summary.csv"
        if not summary_csv.exists():
            print("error=run_stub summary.csv missing; cannot derive reproducible uid set")
            return 5
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        used_uids = [str(r.get("video_uid", "")).strip() for r in rows if str(r.get("video_uid", "")).strip()]
        if not used_uids:
            print("error=no video_uid found in run_stub summary.csv")
            return 5
        effective_uids_file = compare_dir / "uids_used.txt"
        effective_uids_file.parent.mkdir(parents=True, exist_ok=True)
        effective_uids_file.write_text("\n".join(used_uids) + "\n", encoding="utf-8")

    cmd_real = _build_smoke_cmd(
        out_dir=run_real,
        root=args.root,
        uids_file=str(effective_uids_file) if effective_uids_file else None,
        jobs=args.jobs,
        n=args.n,
        with_eval=args.with_eval,
        with_nlq=args.with_nlq,
        nlq_mode=args.nlq_mode,
        with_perception=args.with_perception,
        perception_backend=str(args.real_perception_backend),
        perception_strict=bool(args.real_perception_strict),
        perception_fps=args.perception_fps,
        perception_max_frames=args.perception_max_frames,
        with_bye=args.with_bye,
        with_bye_report=args.with_bye_report,
        bye_root=bye_root,
        bye_gate=args.bye_gate,
        max_bye_critical_fn=args.max_bye_critical_fn,
        bye_skip_regression=args.bye_skip_regression,
        bye_video_mode=args.bye_video_mode,
        bye_lint=args.bye_lint,
        bye_report=args.bye_report,
        bye_regression=args.bye_regression,
        strict=args.strict,
        prefer_short=args.prefer_short,
        prefer_long=args.prefer_long,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        probe_candidates=args.probe_candidates,
        min_size_bytes=args.min_size_bytes,
    )
    rc = _run(cmd_real, cwd=ROOT, log_prefix=compare_dir / "run_real", commands_file=commands_file)
    if rc != 0:
        return rc

    cmd_bye_compare = [
        sys.executable,
        str(ROOT / "scripts" / "compare_bye_metrics.py"),
        "--run_a",
        str(run_stub),
        "--run_b",
        str(run_real),
        "--out_dir",
        str(compare_bye_dir),
        "--format",
        "md+csv",
    ]
    rc = _run(cmd_bye_compare, cwd=ROOT, log_prefix=compare_dir / "compare_bye", commands_file=commands_file)
    if rc != 0:
        return rc

    cmd_bye_report_compare: list[str] | None = None
    if args.with_bye and args.with_bye_report:
        cmd_bye_report_compare = [
            sys.executable,
            str(ROOT / "scripts" / "compare_bye_report_metrics.py"),
            "--a-dir",
            str(run_stub),
            "--b-dir",
            str(run_real),
            "--a-label",
            "stub",
            "--b-label",
            "real",
            "--format",
            "md+csv",
            "--out_dir",
            str(compare_bye_report_dir),
        ]
        rc = _run(
            cmd_bye_report_compare,
            cwd=ROOT,
            log_prefix=compare_dir / "compare_bye_report",
            commands_file=commands_file,
        )
        if rc != 0:
            return rc

    bye_budget_cmds: list[list[str]] = []
    bye_budget_compare_cmd: list[str] | None = None
    bye_budget_summary_path = compare_bye_budget_compare / "compare_summary.json"
    if args.with_bye_budget_sweep:
        if not args.with_bye:
            print("error=--with-bye-budget-sweep requires --with-bye")
            return 4
        if not bye_root and args.strict:
            print("error=BYE budget sweep strict mode requires --bye-root or BYE_ROOT env")
            return 4
        sweep_script = ROOT / "scripts" / "sweep_bye_budgets.py"
        common = [
            "--uids-file",
            str(effective_uids_file) if effective_uids_file else "",
            "--strict-uids",
            "--budgets",
            str(args.bye_budgets),
            "--primary-metric",
            str(args.bye_primary_metric),
            "--formats",
            "png,pdf",
        ]
        if bye_root:
            common.extend(["--bye-root", str(bye_root)])
        if args.bye_skip_regression:
            common.append("--skip-regression")
        cmd_stub_budgets = [
            sys.executable,
            str(sweep_script),
            "--pov-json-dir",
            str(run_stub / "json"),
            "--out-dir",
            str(compare_bye_budget_stub),
            *common,
        ]
        cmd_real_budgets = [
            sys.executable,
            str(sweep_script),
            "--pov-json-dir",
            str(run_real / "json"),
            "--out-dir",
            str(compare_bye_budget_real),
            *common,
        ]
        for cmd, lp in (
            (cmd_stub_budgets, compare_dir / "bye_budget_stub"),
            (cmd_real_budgets, compare_dir / "bye_budget_real"),
        ):
            rc = _run(cmd, cwd=ROOT, log_prefix=lp, commands_file=commands_file)
            if rc != 0:
                return rc
            bye_budget_cmds.append(cmd)

        bye_budget_compare_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "compare_bye_budget_sweeps.py"),
            "--a_dir",
            str(compare_bye_budget_stub),
            "--b_dir",
            str(compare_bye_budget_real),
            "--a_label",
            "stub",
            "--b_label",
            "real",
            "--primary-metric",
            str(args.bye_primary_metric),
            "--format",
            "md+csv",
            "--out_dir",
            str(compare_bye_budget_compare),
        ]
        rc = _run(
            bye_budget_compare_cmd,
            cwd=ROOT,
            log_prefix=compare_dir / "bye_budget_compare",
            commands_file=commands_file,
        )
        if rc != 0:
            return rc
        bye_budget_cmds.append(bye_budget_compare_cmd)

    nlq_budget_cmds: list[list[str]] = []
    if run_nlq_budget_sweep:
        nlq_sweep_script = ROOT / "scripts" / "sweep_nlq_budgets.py"
        common = [
            "--uids-file",
            str(effective_uids_file) if effective_uids_file else "",
            "--strict-uids",
            "--budgets",
            str(args.nlq_budgets),
            "--mode",
            str(args.nlq_mode),
            "--seed",
            "0",
            "--top-k",
            "6",
            "--no-allow-gt-fallback",
            "--hard-constraints",
            "--safety-report",
        ]
        if args.nlq_eval_script:
            common.extend(["--eval-script", str(args.nlq_eval_script)])

        cmd_stub_nlq = [
            sys.executable,
            str(nlq_sweep_script),
            "--json_dir",
            str(run_stub / "json"),
            "--index_dir",
            str(run_stub / "cache"),
            "--out_dir",
            str(compare_nlq_budget_stub),
            *common,
        ]
        cmd_real_nlq = [
            sys.executable,
            str(nlq_sweep_script),
            "--json_dir",
            str(run_real / "json"),
            "--index_dir",
            str(run_real / "cache"),
            "--out_dir",
            str(compare_nlq_budget_real),
            *common,
        ]
        for cmd, lp in (
            (cmd_stub_nlq, compare_dir / "nlq_budget_stub"),
            (cmd_real_nlq, compare_dir / "nlq_budget_real"),
        ):
            rc = _run(cmd, cwd=ROOT, log_prefix=lp, commands_file=commands_file)
            if rc != 0:
                return rc
            nlq_budget_cmds.append(cmd)

    streaming_budget_cmds: list[list[str]] = []
    if args.with_streaming_budget:
        streaming_sweep_script = ROOT / "scripts" / "sweep_streaming_budgets.py"
        common = [
            "--uids-file",
            str(effective_uids_file) if effective_uids_file else "",
            "--strict-uids",
            "--budgets",
            str(args.nlq_budgets),
            "--step-s",
            str(float(args.streaming_step_s)),
            "--mode",
            str(args.nlq_mode),
            "--policy",
            "fixed",
            "--formats",
            "png,pdf",
        ]
        common.append("--context-use-repo" if bool(args.with_repo) else "--no-context-use-repo")
        common.extend(["--repo-read-policy", str(args.repo_read_policy)])
        if args.repo_budget:
            common.extend(["--repo-budget", str(args.repo_budget)])
        cmd_stub_stream = [
            sys.executable,
            str(streaming_sweep_script),
            "--json_dir",
            str(run_stub / "json"),
            "--out_dir",
            str(compare_streaming_budget_stub),
            *common,
        ]
        cmd_real_stream = [
            sys.executable,
            str(streaming_sweep_script),
            "--json_dir",
            str(run_real / "json"),
            "--out_dir",
            str(compare_streaming_budget_real),
            *common,
        ]
        for cmd, lp in (
            (cmd_stub_stream, compare_dir / "streaming_budget_stub"),
            (cmd_real_stream, compare_dir / "streaming_budget_real"),
        ):
            rc = _run(cmd, cwd=ROOT, log_prefix=lp, commands_file=commands_file)
            if rc != 0:
                return rc
            streaming_budget_cmds.append(cmd)

    chain_backoff_budgets = str(args.streaming_chain_backoff_budgets).strip() if args.streaming_chain_backoff_budgets else str(args.nlq_budgets)
    chain_backoff_step_s = (
        float(args.streaming_chain_backoff_step_s)
        if args.streaming_chain_backoff_step_s is not None
        else float(args.streaming_step_s if args.streaming_step_s is not None else 8.0)
    )
    chain_backoff_seed = int(args.streaming_chain_backoff_seed) if args.streaming_chain_backoff_seed is not None else int(args.seed)
    streaming_chain_backoff_cmds: list[list[str]] = []
    if args.with_streaming_chain_backoff:
        chain_backoff_compare_script = ROOT / "scripts" / "run_streaming_chain_backoff_compare.py"
        chosen_json = _pick_streaming_json(run_real / "json", run_stub / "json", effective_uids_file)
        if chosen_json is None:
            print("error=with-streaming-chain-backoff enabled but no *_v03_decisions.json found in run_real/run_stub")
            return 6
        cmd_chain_backoff = [
            sys.executable,
            str(chain_backoff_compare_script),
            "--json",
            str(chosen_json),
            "--out_dir",
            str(compare_streaming_chain_backoff),
            "--budgets",
            str(chain_backoff_budgets),
            "--step-s",
            str(float(chain_backoff_step_s)),
            "--policies",
            str(args.streaming_chain_backoff_policies),
            "--seed",
            str(int(chain_backoff_seed)),
            "--mode",
            str(args.nlq_mode),
        ]
        rc = _run(
            cmd_chain_backoff,
            cwd=ROOT,
            log_prefix=compare_dir / "streaming_chain_backoff_compare",
            commands_file=commands_file,
        )
        if rc != 0:
            return rc
        streaming_chain_backoff_cmds.append(cmd_chain_backoff)

    reranker_sweep_cmds: list[list[str]] = []
    if args.with_reranker_sweep:
        sweep_script = ROOT / "scripts" / "sweep_reranker.py"
        common = [
            "--nlq-mode",
            str(args.nlq_mode),
            "--top-k",
            "6",
            "--seed",
            "0",
            "--search",
            "grid",
        ]
        if args.nlq_eval_script:
            common.extend(["--eval-script", str(args.nlq_eval_script)])
        if str(args.reranker_sweep_grid).strip():
            common.extend(["--grid", str(args.reranker_sweep_grid).strip()])
        cmd_stub_sweep = [
            sys.executable,
            str(sweep_script),
            "--run_dir",
            str(run_stub),
            "--out_dir",
            str(compare_reranker_sweep_stub),
            *common,
        ]
        cmd_real_sweep = [
            sys.executable,
            str(sweep_script),
            "--run_dir",
            str(run_real),
            "--out_dir",
            str(compare_reranker_sweep_real),
            *common,
        ]
        for cmd, lp in (
            (cmd_stub_sweep, compare_dir / "reranker_sweep_stub"),
            (cmd_real_sweep, compare_dir / "reranker_sweep_real"),
        ):
            rc = _run(cmd, cwd=ROOT, log_prefix=lp, commands_file=commands_file)
            if rc != 0:
                return rc
            reranker_sweep_cmds.append(cmd)

    budget_recommend_cmds: list[list[str]] = []
    if args.with_budget_recommend:
        if not args.with_bye_budget_sweep or not run_nlq_budget_sweep:
            print("error=--with-budget-recommend requires --with-bye-budget-sweep and --with-nlq-budget-sweep")
            return 6
        recommend_script = ROOT / "scripts" / "recommend_budget.py"
        cmd_stub_reco = [
            sys.executable,
            str(recommend_script),
            "--bye_dir",
            str(compare_bye_budget_stub),
            "--nlq_dir",
            str(compare_nlq_budget_stub),
            "--out_dir",
            str(compare_budget_recommend_stub),
            "--label",
            "stub",
            "--primary-bye-metric",
            str(args.bye_primary_metric),
            "--primary-nlq-metric",
            "nlq_full_hit_at_k_strict",
        ]
        cmd_real_reco = [
            sys.executable,
            str(recommend_script),
            "--bye_dir",
            str(compare_bye_budget_real),
            "--nlq_dir",
            str(compare_nlq_budget_real),
            "--out_dir",
            str(compare_budget_recommend_real),
            "--label",
            "real",
            "--primary-bye-metric",
            str(args.bye_primary_metric),
            "--primary-nlq-metric",
            "nlq_full_hit_at_k_strict",
        ]
        if args.budget_weights_json:
            cmd_stub_reco.extend(["--weights-json", str(args.budget_weights_json)])
            cmd_real_reco.extend(["--weights-json", str(args.budget_weights_json)])
        if args.budget_gates_json:
            cmd_stub_reco.extend(["--gates-json", str(args.budget_gates_json)])
            cmd_real_reco.extend(["--gates-json", str(args.budget_gates_json)])

        for cmd, lp in (
            (cmd_stub_reco, compare_dir / "budget_recommend_stub"),
            (cmd_real_reco, compare_dir / "budget_recommend_real"),
        ):
            rc = _run(cmd, cwd=ROOT, log_prefix=lp, commands_file=commands_file)
            if rc != 0:
                return rc
            budget_recommend_cmds.append(cmd)

    fig_cmds: list[list[str]] = []
    if args.with_figs:
        stub_cross = run_stub / "eval"
        real_cross = run_real / "eval"
        stub_nlq = run_stub / "nlq_summary_all.csv"
        real_nlq = run_real / "nlq_summary_all.csv"
        if not (stub_cross.exists() and real_cross.exists() and stub_nlq.exists() and real_nlq.exists()):
            print("error=with-figs requires eval outputs and nlq_summary_all.csv for both runs")
            return 3

        stub_fig_dir = compare_dir / "paper_figs_stub"
        real_fig_dir = compare_dir / "paper_figs_real"
        compare_fig_dir = compare_dir / "paper_figs_compare"
        cmd_stub_fig = [
            sys.executable,
            str(ROOT / "scripts" / "make_paper_figures.py"),
            "--cross_dir",
            str(stub_cross),
            "--nlq_csv",
            str(stub_nlq),
            "--out_dir",
            str(stub_fig_dir),
            "--macro_avg",
            "--label",
            "stub",
        ]
        cmd_real_fig = [
            sys.executable,
            str(ROOT / "scripts" / "make_paper_figures.py"),
            "--cross_dir",
            str(real_cross),
            "--nlq_csv",
            str(real_nlq),
            "--out_dir",
            str(real_fig_dir),
            "--macro_avg",
            "--label",
            "real",
        ]
        cmd_compare_fig = [
            sys.executable,
            str(ROOT / "scripts" / "make_paper_figures.py"),
            "--cross_dir",
            str(real_cross),
            "--nlq_csv",
            str(real_nlq),
            "--out_dir",
            str(compare_fig_dir),
            "--macro_avg",
            "--label",
            "real",
            "--compare_dir",
            str(stub_fig_dir),
            "--compare_label",
            "stub",
        ]
        for cmd, lp in (
            (cmd_stub_fig, compare_dir / "fig_stub"),
            (cmd_real_fig, compare_dir / "fig_real"),
            (cmd_compare_fig, compare_dir / "fig_compare"),
        ):
            rc = _run(cmd, cwd=ROOT, log_prefix=lp, commands_file=commands_file)
            if rc != 0:
                return rc
            fig_cmds.append(cmd)

    paper_ready_cmds: list[list[str]] = []
    if args.export_paper_ready:
        cmd_paper_ready = [
            sys.executable,
            str(ROOT / "scripts" / "export_paper_ready.py"),
            "--compare_dir",
            str(compare_dir),
            "--out_dir",
            str(compare_paper_ready),
            "--label_a",
            "stub",
            "--label_b",
            "real",
            "--format",
            str(args.paper_ready_format),
            "--with-figs",
        ]
        if args.with_bye and args.with_bye_report:
            cmd_paper_ready.extend(["--bye-report-compare-dir", str(compare_bye_report_dir)])
        if args.with_reranker_sweep:
            cmd_paper_ready.extend(["--reranker-sweep-dir", str(compare_reranker_sweep_real)])
        if run_nlq_budget_sweep:
            cmd_paper_ready.extend(["--lost-object-panel-dir", str(compare_nlq_budget_real / "aggregate")])
            if str(args.nlq_mode).strip().lower() == "hard_pseudo_chain":
                cmd_paper_ready.extend(["--chain-nlq-dir", str(compare_nlq_budget_real / "aggregate")])
        if args.with_streaming_chain_backoff:
            cmd_paper_ready.extend(
                ["--streaming-chain-backoff-compare-dir", str(compare_streaming_chain_backoff / "compare")]
            )
        rc = _run(cmd_paper_ready, cwd=ROOT, log_prefix=compare_dir / "paper_ready_export", commands_file=commands_file)
        if rc != 0:
            return rc
        paper_ready_cmds.append(cmd_paper_ready)

    _copy_snapshots(run_stub, "stub", compare_snapshots)
    _copy_snapshots(run_real, "real", compare_snapshots)
    compare_snapshot = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "root": str(args.root),
            "uids_file": str(effective_uids_file) if effective_uids_file else None,
            "seed": int(args.seed),
            "with_eval": bool(args.with_eval),
            "with_nlq": bool(args.with_nlq),
            "nlq_mode": str(args.nlq_mode),
            "with_streaming_budget": bool(args.with_streaming_budget),
            "streaming_step_s": float(args.streaming_step_s),
            "with_streaming_chain_backoff": bool(args.with_streaming_chain_backoff),
            "export_paper_ready": bool(args.export_paper_ready),
        },
        "outputs": {
            "run_stub": str(run_stub),
            "run_real": str(run_real),
            "compare_dir": str(compare_dir),
        },
        "streaming_chain_backoff": {
            "enabled": bool(args.with_streaming_chain_backoff),
            "dir": str(compare_streaming_chain_backoff),
            "policies": str(args.streaming_chain_backoff_policies),
            "budgets": str(chain_backoff_budgets),
            "step_s": float(chain_backoff_step_s),
            "seed": int(chain_backoff_seed),
        },
    }
    (compare_dir / "snapshot.json").write_text(json.dumps(compare_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_compare_readme(
        out_dir,
        run_label=str(args.run_label),
        uids_file=effective_uids_file,
        cmd_stub=cmd_stub,
        cmd_real=cmd_real,
        cmd_bye_compare=cmd_bye_compare,
        cmd_bye_report_compare=cmd_bye_report_compare,
        cmd_bye_budget=bye_budget_cmds,
        cmd_nlq_budget=nlq_budget_cmds,
        cmd_streaming_budget=streaming_budget_cmds,
        cmd_streaming_chain_backoff=streaming_chain_backoff_cmds,
        cmd_reranker_sweep=reranker_sweep_cmds,
        cmd_budget_recommend=budget_recommend_cmds,
        cmd_paper_ready=paper_ready_cmds,
        fig_cmds=fig_cmds,
    )

    print(f"saved_stub={run_stub}")
    print(f"saved_real={run_real}")
    print(f"saved_compare={compare_dir}")
    print(f"run_label={str(args.run_label)}")
    print(f"with_repo={str(bool(args.with_repo)).lower()}")
    print(f"repo_read_policy={str(args.repo_read_policy)}")
    print(f"with_bye_report={str(bool(args.with_bye_report)).lower()}")
    print(f"bye_gate={str(bool(args.bye_gate)).lower()}")
    if args.with_bye_budget_sweep:
        print(f"bye_budget_stub_saved={compare_bye_budget_stub}")
        print(f"bye_budget_real_saved={compare_bye_budget_real}")
        print(f"bye_budget_compare_saved={compare_bye_budget_compare}")
        if bye_budget_summary_path.exists():
            try:
                payload = json.loads(bye_budget_summary_path.read_text(encoding="utf-8"))
                print(f"budgets_matched={payload.get('budgets_matched')}")
            except Exception:
                pass
    if args.with_bye and args.with_bye_report:
        print(f"bye_report_saved={compare_bye_report_dir}")
    if run_nlq_budget_sweep:
        print(f"nlq_budget_stub_saved={compare_nlq_budget_stub}")
        print(f"nlq_budget_real_saved={compare_nlq_budget_real}")
    if args.with_streaming_budget:
        print(f"streaming_budget_stub_saved={compare_streaming_budget_stub}")
        print(f"streaming_budget_real_saved={compare_streaming_budget_real}")
    if args.with_streaming_chain_backoff:
        print(f"streaming_chain_backoff_saved={compare_streaming_chain_backoff}")
        print(f"streaming_chain_backoff_compare_saved={compare_streaming_chain_backoff / 'compare'}")
    if args.with_reranker_sweep:
        print(f"reranker_sweep_stub_saved={compare_reranker_sweep_stub}")
        print(f"reranker_sweep_real_saved={compare_reranker_sweep_real}")
    if args.with_budget_recommend:
        print(f"budget_recommend_stub_saved={compare_budget_recommend_stub}")
        print(f"budget_recommend_real_saved={compare_budget_recommend_real}")
        for label, path in (("stub", compare_budget_recommend_stub), ("real", compare_budget_recommend_real)):
            s = path / "recommend_summary.json"
            if s.exists():
                try:
                    payload = json.loads(s.read_text(encoding="utf-8"))
                    print(f"budgets_joined_{label}={payload.get('budgets_joined')}")
                    print(f"top1_{label}={payload.get('top1_budget_key')}")
                except Exception:
                    pass
    if args.export_paper_ready:
        print(f"paper_ready_saved={compare_paper_ready}")
        if args.with_streaming_chain_backoff:
            chain_fig = compare_paper_ready / "figures" / "fig_streaming_chain_backoff_success_vs_budget_seconds.png"
            print(f"paper_ready_chain_backoff_fig_exists={str(chain_fig.exists()).lower()}")
        panel_csv = compare_paper_ready / "tables" / "table_budget_panel.csv"
        if panel_csv.exists():
            try:
                with panel_csv.open("r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                keys = {
                    str(r.get("budget_key", "")).strip()
                    for r in rows
                    if str(r.get("budget_key", "")).strip()
                }
                print(f"budgets_matched={len(keys)}")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
