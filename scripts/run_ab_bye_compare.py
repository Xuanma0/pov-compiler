from __future__ import annotations

import argparse
import datetime as dt
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
    parser.add_argument("--uids-file", required=True, help="UID list file for reproducible AB")
    parser.add_argument("--out_dir", required=True, help="Output root")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--n", type=int, default=None)

    _parse_bool_with_neg(parser, "with-eval", default=False)
    _parse_bool_with_neg(parser, "with-nlq", default=False)
    parser.add_argument("--nlq-mode", choices=["mock", "pseudo_nlq", "hard_pseudo_nlq", "ego4d"], default="hard_pseudo_nlq")
    _parse_bool_with_neg(parser, "with-figs", default=False)

    _parse_bool_with_neg(parser, "with-perception", default=False)
    parser.add_argument("--stub-perception-backend", choices=["stub", "real"], default="stub")
    parser.add_argument("--real-perception-backend", choices=["stub", "real"], default="real")
    parser.add_argument("--perception-fps", type=float, default=5.0)
    parser.add_argument("--perception-max-frames", type=int, default=300)
    _parse_bool_with_neg(parser, "real-perception-strict", default=True)
    _parse_bool_with_neg(parser, "stub-perception-strict", default=False)

    _parse_bool_with_neg(parser, "with-bye", default=False)
    parser.add_argument("--bye-root", default=None)
    _parse_bool_with_neg(parser, "bye-skip-regression", default=True)
    parser.add_argument("--bye-video-mode", choices=["none", "copy", "link"], default="none")
    parser.add_argument("--bye-lint", default=None)
    parser.add_argument("--bye-report", default=None)
    parser.add_argument("--bye-regression", default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if BYE missing when --with-bye")

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
    uids_file: str,
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
    bye_root: str | None,
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
        "--uids-file",
        str(uids_file),
        "--jobs",
        str(int(jobs)),
        "--min-size-bytes",
        str(int(min_size_bytes)),
        "--probe-candidates",
        str(int(probe_candidates)),
        "--no-proxy",
        "--resume",
    ]
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


def _write_compare_readme(
    out_dir: Path,
    *,
    uids_file: Path,
    cmd_stub: list[str],
    cmd_real: list[str],
    cmd_bye_compare: list[str],
    fig_cmds: list[list[str]],
) -> None:
    lines = [
        "# AB v1.2 Compare",
        "",
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
            f"- `{out_dir / 'compare' / 'commands.sh'}`",
            f"- `{out_dir / 'compare' / 'snapshots'}`",
        ]
    )
    (out_dir / "compare" / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    run_stub = out_dir / "run_stub"
    run_real = out_dir / "run_real"
    compare_dir = out_dir / "compare"
    compare_bye_dir = compare_dir / "bye"
    compare_snapshots = compare_dir / "snapshots"
    run_stub.mkdir(parents=True, exist_ok=True)
    run_real.mkdir(parents=True, exist_ok=True)
    compare_bye_dir.mkdir(parents=True, exist_ok=True)
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
        bye_root=bye_root,
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

    cmd_real = _build_smoke_cmd(
        out_dir=run_real,
        root=args.root,
        uids_file=args.uids_file,
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
        bye_root=bye_root,
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

    _copy_snapshots(run_stub, "stub", compare_snapshots)
    _copy_snapshots(run_real, "real", compare_snapshots)
    _write_compare_readme(
        out_dir,
        uids_file=Path(args.uids_file),
        cmd_stub=cmd_stub,
        cmd_real=cmd_real,
        cmd_bye_compare=cmd_bye_compare,
        fig_cmds=fig_cmds,
    )

    print(f"saved_stub={run_stub}")
    print(f"saved_real={run_real}")
    print(f"saved_compare={compare_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
