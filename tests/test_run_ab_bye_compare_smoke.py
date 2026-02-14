from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_fake_tool(path: Path, *, writes_report: bool = False) -> None:
    lines = [
        "import argparse",
        "import json",
        "from pathlib import Path",
        "p=argparse.ArgumentParser()",
        "p.add_argument('--run-package','--run_package','--package',dest='run_package',default='')",
        "p.add_argument('--out-dir','--out_dir','--out',dest='out_dir',default='')",
        "p.add_argument('pos', nargs='*')",
        "a=p.parse_args()",
        "rp=a.run_package or (a.pos[0] if a.pos else '')",
        "od=a.out_dir or (a.pos[1] if len(a.pos)>1 else rp)",
        "print('OK', Path(__file__).name, rp, od)",
    ]
    if writes_report:
        lines.extend(
            [
                "out=Path(od)",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out/'report.json').write_text(json.dumps({'acc':0.75,'latency_ms':11.0}), encoding='utf-8')",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pipeline_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_perception_complete(run_dir: Path, uid: str) -> None:
    pdir = run_dir / "perception" / uid
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "perception.json").write_text(json.dumps({"video_id": uid}), encoding="utf-8")
    (pdir / "events_v0.json").write_text(json.dumps({"video_id": uid, "events_v0": []}), encoding="utf-8")
    (pdir / "report.md").write_text("# ok", encoding="utf-8")


def _write_fake_eval_script(path: Path) -> None:
    lines = [
        "import argparse, csv, json",
        "from pathlib import Path",
        "p=argparse.ArgumentParser()",
        "p.add_argument('--json', required=True)",
        "p.add_argument('--index', default='')",
        "p.add_argument('--out_dir', required=True)",
        "p.add_argument('--budget-max-total-s', type=float, default=60.0)",
        "p.add_argument('--budget-max-tokens', type=int, default=200)",
        "p.add_argument('--budget-max-decisions', type=int, default=12)",
        "p.add_argument('--mode', default='hard_pseudo_nlq')",
        "p.add_argument('--seed', type=int, default=0)",
        "p.add_argument('--top-k', type=int, default=6)",
        "p.add_argument('--allow-gt-fallback', action='store_true')",
        "p.add_argument('--no-allow-gt-fallback', action='store_true')",
        "p.add_argument('--hard-constraints', default='on')",
        "p.add_argument('--no-safety-gate', action='store_true')",
        "a=p.parse_args()",
        "out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)",
        "score=max(0.0, min(1.0, a.budget_max_total_s/100.0))",
        "rows=[{'variant':'full','hit_at_k_strict':score,'hit_at_1_strict':score*0.8,'top1_in_distractor':max(0.0,0.6-score*0.4),'mrr':score*0.9}]",
        "with (out/'nlq_results.csv').open('w', encoding='utf-8', newline='') as f:",
        " w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]",
        "(out/'nlq_summary.csv').write_text('query_type,variant,hit_at_k_strict\\n', encoding='utf-8')",
        "safety={'critical_fn_rate':max(0.0,0.5-score*0.3),'variant_stats':{'full':{'critical_fn_rate':max(0.0,0.5-score*0.3)}}}",
        "(out/'safety_report.json').write_text(json.dumps(safety), encoding='utf-8')",
        "(out/'nlq_report.md').write_text('# fake', encoding='utf-8')",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_run_ab_bye_compare_minimal(tmp_path: Path) -> None:
    root = tmp_path / "ego_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "demo.mp4").write_bytes(b"0")
    uid = hashlib.md5("demo.mp4".encode("utf-8")).hexdigest()
    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(uid + "\n", encoding="utf-8")

    out_dir = tmp_path / "ab_out"
    for run_name in ("run_stub", "run_real"):
        run_dir = out_dir / run_name
        _write_pipeline_json(run_dir / "json" / f"{uid}_v03_decisions.json", uid)
        (run_dir / "cache" / f"{uid}.index.npz").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "cache" / f"{uid}.index.npz").write_bytes(b"NPZ")
        (run_dir / "cache" / f"{uid}.index_meta.json").write_text("{}", encoding="utf-8")
        _write_perception_complete(run_dir, uid)

    fake_bye = tmp_path / "fake_bye"
    _write_fake_tool(fake_bye / "Gateway" / "scripts" / "lint_run_package.py")
    _write_fake_tool(fake_bye / "report_run.py", writes_report=True)
    _write_fake_tool(fake_bye / "run_regression_suite.py")
    fake_eval = tmp_path / "fake_eval_nlq.py"
    _write_fake_eval_script(fake_eval)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ab_bye_compare.py"),
        "--root",
        str(root),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--jobs",
        "1",
        "--with-bye",
        "--with-bye-budget-sweep",
        "--bye-budgets",
        "20/50/4,40/100/8",
        "--with-nlq-budget-sweep",
        "--nlq-budgets",
        "20/50/4,40/100/8",
        "--with-budget-recommend",
        "--with-streaming-budget",
        "--streaming-step-s",
        "8",
        "--export-paper-ready",
        "--with-reranker-sweep",
        "--reranker-sweep-grid",
        "w_trigger=0.2,0.8;w_action=0.3",
        "--nlq-eval-script",
        str(fake_eval),
        "--bye-root",
        str(fake_bye),
        "--bye-skip-regression",
        "--with-perception",
        "--stub-perception-backend",
        "stub",
        "--real-perception-backend",
        "stub",
        "--min-size-bytes",
        "0",
        "--probe-candidates",
        "0",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    compare_dir = out_dir / "compare"
    assert (compare_dir / "bye" / "table_bye_compare.csv").exists()
    assert (compare_dir / "bye" / "table_bye_compare.md").exists()
    assert (compare_dir / "bye" / "compare_summary.json").exists()
    assert (compare_dir / "bye_report" / "tables" / "table_bye_report_compare.csv").exists()
    assert (compare_dir / "bye_report" / "tables" / "table_bye_report_compare.md").exists()
    assert (compare_dir / "bye_report" / "figures" / "fig_bye_critical_fn_delta.png").exists()
    assert (compare_dir / "bye_report" / "figures" / "fig_bye_latency_delta.png").exists()
    assert (compare_dir / "bye_budget" / "stub" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "bye_budget" / "real" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "bye_budget" / "compare" / "tables" / "table_budget_compare.csv").exists()
    assert (compare_dir / "bye_budget" / "compare" / "tables" / "table_budget_compare.md").exists()
    assert (compare_dir / "bye_budget" / "compare" / "figures" / "fig_bye_primary_vs_budget_seconds_compare.png").exists()
    assert (compare_dir / "bye_budget" / "compare" / "figures" / "fig_bye_primary_delta_vs_budget_seconds.png").exists()
    assert (compare_dir / "nlq_budget" / "stub" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "nlq_budget" / "real" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "nlq_budget" / "stub" / "figures" / "fig_nlq_critical_fn_rate_vs_budget_seconds.png").exists()
    assert (compare_dir / "nlq_budget" / "real" / "figures" / "fig_nlq_critical_fn_rate_vs_budget_seconds.png").exists()
    assert (compare_dir / "streaming_budget" / "stub" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "streaming_budget" / "real" / "aggregate" / "metrics_by_budget.csv").exists()
    assert (compare_dir / "budget_recommend" / "stub" / "tables" / "table_budget_recommend.csv").exists()
    assert (compare_dir / "budget_recommend" / "real" / "tables" / "table_budget_recommend.csv").exists()
    assert (compare_dir / "paper_ready" / "tables" / "table_budget_panel.csv").exists()
    assert (compare_dir / "paper_ready" / "tables" / "table_budget_panel_delta.csv").exists()
    assert (compare_dir / "paper_ready" / "bye_report" / "table_bye_report_compare.csv").exists()
    assert (compare_dir / "paper_ready" / "report.md").exists()
    assert (compare_dir / "paper_ready" / "snapshot.json").exists()
    assert (compare_dir / "paper_ready" / "figures" / "fig_nlq_critical_fn_rate_vs_seconds.png").exists()
    assert (compare_dir / "reranker_sweep" / "stub" / "aggregate" / "metrics_by_weights.csv").exists()
    assert (compare_dir / "reranker_sweep" / "real" / "aggregate" / "metrics_by_weights.csv").exists()
    assert (compare_dir / "paper_ready" / "reranker_sweep" / "metrics_by_weights.csv").exists()
    assert (compare_dir / "commands.sh").exists()
    assert (compare_dir / "README.md").exists()


def test_run_ab_bye_compare_generates_uids_used_when_missing(tmp_path: Path) -> None:
    root = tmp_path / "ego_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "demo.mp4").write_bytes(b"0")
    uid = hashlib.md5("demo.mp4".encode("utf-8")).hexdigest()

    out_dir = tmp_path / "ab_out_no_uids"
    for run_name in ("run_stub", "run_real"):
        run_dir = out_dir / run_name
        _write_pipeline_json(run_dir / "json" / f"{uid}_v03_decisions.json", uid)
        (run_dir / "cache" / f"{uid}.index.npz").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "cache" / f"{uid}.index.npz").write_bytes(b"NPZ")
        (run_dir / "cache" / f"{uid}.index_meta.json").write_text("{}", encoding="utf-8")
        _write_perception_complete(run_dir, uid)

    fake_bye = tmp_path / "fake_bye"
    _write_fake_tool(fake_bye / "Gateway" / "scripts" / "lint_run_package.py")
    _write_fake_tool(fake_bye / "report_run.py", writes_report=True)
    _write_fake_tool(fake_bye / "run_regression_suite.py")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_ab_bye_compare.py"),
        "--root",
        str(root),
        "--out_dir",
        str(out_dir),
        "--n",
        "1",
        "--jobs",
        "1",
        "--with-bye",
        "--with-bye-budget-sweep",
        "--bye-budgets",
        "20/50/4",
        "--bye-root",
        str(fake_bye),
        "--bye-skip-regression",
        "--min-size-bytes",
        "0",
        "--probe-candidates",
        "0",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    compare_dir = out_dir / "compare"
    uids_used = compare_dir / "uids_used.txt"
    assert uids_used.exists()
    used_lines = [x.strip() for x in uids_used.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert used_lines
    commands_text = (compare_dir / "commands.sh").read_text(encoding="utf-8")
    assert "--uids-file" in commands_text
    assert str(uids_used) in commands_text
    assert "compare_bye_budget_sweeps.py" in commands_text
