from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_min_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fake_eval(path: Path) -> None:
    lines = [
        "import argparse, csv, json, re",
        "from pathlib import Path",
        "p=argparse.ArgumentParser()",
        "p.add_argument('--json', required=True)",
        "p.add_argument('--out_dir', required=True)",
        "p.add_argument('--rerank-cfg', required=True)",
        "p.add_argument('--index', default='')",
        "p.add_argument('--mode', default='hard_pseudo_nlq')",
        "p.add_argument('--seed', type=int, default=0)",
        "p.add_argument('--top-k', type=int, default=6)",
        "p.add_argument('--no-allow-gt-fallback', action='store_true')",
        "p.add_argument('--no-safety-gate', action='store_true')",
        "a=p.parse_args()",
        "cfg_text=Path(a.rerank_cfg).read_text(encoding='utf-8')",
        "m=re.search(r'w_trigger\\s*[:=]\\s*([0-9.]+)', cfg_text)",
        "w=float(m.group(1)) if m else 0.3",
        "mrr=max(0.0,min(1.0,0.2+0.6*w))",
        "dist=max(0.0,min(1.0,0.7-0.5*w))",
        "crit=max(0.0,min(1.0,0.5-0.4*w))",
        "out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)",
        "rows=[{'variant':'full','hit_at_k_strict':1.0 if mrr>0.4 else 0.0,'hit_at_1_strict':1.0 if mrr>0.6 else 0.0,'mrr':mrr,'top1_in_distractor':dist}]",
        "with (out/'nlq_results.csv').open('w', encoding='utf-8', newline='') as f:",
        " w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]",
        "(out/'nlq_summary.csv').write_text('variant,query_type,hit_at_k_strict\\nfull,hard_pseudo_decision,1\\n', encoding='utf-8')",
        "(out/'safety_report.json').write_text(json.dumps({'critical_fn_rate':crit,'reason_counts':{'budget_insufficient':1}}, ensure_ascii=False), encoding='utf-8')",
        "(out/'nlq_report.md').write_text('# fake', encoding='utf-8')",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sweep_reranker_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_min_json(run_dir / "json" / "vid_001_v03_decisions.json", "vid_001")
    eval_script = tmp_path / "fake_eval_nlq.py"
    _write_fake_eval(eval_script)
    out_dir = tmp_path / "sweep_out"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_reranker.py"),
        "--run_dir",
        str(run_dir),
        "--out_dir",
        str(out_dir),
        "--search",
        "grid",
        "--w-trigger-list",
        "0.1,0.9",
        "--w-action-list",
        "0.2",
        "--w-constraint-list",
        "0.1",
        "--w-outcome-list",
        "0.1",
        "--w-evidence-list",
        "0.1",
        "--w-semantic-list",
        "1.0",
        "--eval-script",
        str(eval_script),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    metrics_csv = out_dir / "aggregate" / "metrics_by_weights.csv"
    metrics_md = out_dir / "aggregate" / "metrics_by_weights.md"
    best_cfg = out_dir / "best_weights.yaml"
    best_report = out_dir / "best_report.md"
    snapshot = out_dir / "snapshot.json"
    fig1_png = out_dir / "figures" / "fig_objective_vs_weights_id.png"
    fig1_pdf = out_dir / "figures" / "fig_objective_vs_weights_id.pdf"
    fig2_png = out_dir / "figures" / "fig_tradeoff_strict_vs_distractor.png"
    fig2_pdf = out_dir / "figures" / "fig_tradeoff_strict_vs_distractor.pdf"

    assert metrics_csv.exists()
    assert metrics_md.exists()
    assert best_cfg.exists()
    assert best_report.exists()
    assert snapshot.exists()
    assert fig1_png.exists() and fig1_pdf.exists()
    assert fig2_png.exists() and fig2_pdf.exists()

    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2
    assert "objective" in rows[0]
