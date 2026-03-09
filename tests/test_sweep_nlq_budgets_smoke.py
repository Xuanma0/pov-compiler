from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_pov_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "events": [{"id": "event_0001", "t0": 0.0, "t1": 1.0, "scores": {}, "anchors": []}],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
        "rows=[",
        " {'variant':'full','hit_at_k_strict':score,'hit_at_1_strict':score*0.8,'top1_in_distractor':max(0.0,0.6-score*0.4),'mrr':score*0.9},",
        " {'variant':'highlights_only','hit_at_k_strict':score*0.8,'hit_at_1_strict':score*0.6,'top1_in_distractor':max(0.0,0.7-score*0.3),'mrr':score*0.7},",
        "]",
        "with (out/'nlq_results.csv').open('w', encoding='utf-8', newline='') as f:",
        " w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]",
        "with (out/'nlq_summary.csv').open('w', encoding='utf-8', newline='') as f:",
        " w=csv.DictWriter(f, fieldnames=['query_type','variant','hit_at_k_strict']); w.writeheader();",
        " w.writerow({'query_type':'hard_pseudo_token','variant':'full','hit_at_k_strict':score})",
        "safety={'critical_fn_rate':max(0.0,0.5-score*0.3),'variant_stats':{'full':{'critical_fn_rate':max(0.0,0.5-score*0.3)}}}",
        "(out/'safety_report.json').write_text(json.dumps(safety), encoding='utf-8')",
        "(out/'nlq_report.md').write_text('# fake', encoding='utf-8')",
        "print('saved_results', out/'nlq_results.csv')",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sweep_nlq_budgets_smoke(tmp_path: Path) -> None:
    json_dir = tmp_path / "json"
    index_dir = tmp_path / "cache"
    uid = "demo_uid"
    _write_pov_json(json_dir / f"{uid}_v03_decisions.json", uid)
    (index_dir / f"{uid}.index.npz").parent.mkdir(parents=True, exist_ok=True)
    (index_dir / f"{uid}.index.npz").write_bytes(b"NPZ")
    (index_dir / f"{uid}.index_meta.json").write_text("{}", encoding="utf-8")
    uids = tmp_path / "uids.txt"
    uids.write_text(uid + "\n", encoding="utf-8")
    fake_eval = tmp_path / "fake_eval_nlq.py"
    _write_fake_eval_script(fake_eval)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sweep_nlq_budgets.py"),
        "--json_dir",
        str(json_dir),
        "--index_dir",
        str(index_dir),
        "--uids-file",
        str(uids),
        "--out_dir",
        str(out_dir),
        "--budgets",
        "20/50/4,40/100/8,60/200/12",
        "--eval-script",
        str(fake_eval),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "selection_mode=uids_file" in result.stdout

    assert (out_dir / "aggregate" / "metrics_by_budget.csv").exists()
    assert (out_dir / "aggregate" / "metrics_by_budget.md").exists()
    assert (out_dir / "snapshot.json").exists()
    assert list((out_dir / "figures").glob("fig_nlq_quality_vs_budget_seconds.*"))
    assert list((out_dir / "figures").glob("fig_nlq_strict_vs_budget_seconds.*"))

