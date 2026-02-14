from __future__ import annotations

import csv
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
        "rows=[{'variant':'full','hit_at_k_strict':score,'hit_at_1_strict':score*0.8,'top1_in_distractor':max(0.0,0.6-score*0.4),'mrr':score*0.9}]",
        "with (out/'nlq_results.csv').open('w', encoding='utf-8', newline='') as f:",
        " w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]",
        "(out/'nlq_summary.csv').write_text('query_type,variant,hit_at_k_strict\\n', encoding='utf-8')",
        "den=10",
        "cnt=max(0, int(6-a.budget_max_total_s/20))",
        "safety={",
        " 'count_granularity':'row=(variant,budget,query)',",
        " 'critical_fn_denominator':den,",
        " 'critical_fn_count':cnt,",
        " 'critical_fn_rate':cnt/den,",
        " 'reason_counts':[",
        "   {'reason':'budget_insufficient','count':max(0,cnt-1)},",
        "   {'reason':'evidence_missing','count':1 if cnt>0 else 0},",
        "   {'reason':'constraints_over_filtered','count':0},",
        "   {'reason':'retrieval_distractor','count':0}",
        " ],",
        " 'variant_stats':{'full':{'critical_fn_denominator':den,'critical_fn_count':cnt,'critical_fn_rate':cnt/den}}",
        "}",
        "(out/'safety_report.json').write_text(json.dumps(safety), encoding='utf-8')",
        "(out/'nlq_report.md').write_text('# fake', encoding='utf-8')",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sweep_nlq_budgets_safety_smoke(tmp_path: Path) -> None:
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
        "20/50/4,40/100/8",
        "--eval-script",
        str(fake_eval),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "selection_mode=uids_file" in result.stdout

    metrics_csv = out_dir / "aggregate" / "metrics_by_budget.csv"
    assert metrics_csv.exists()
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    header = set(rows[0].keys())
    assert "safety_critical_fn_rate" in header
    assert "safety_reason_budget_insufficient_rate" in header
    assert "safety_reason_evidence_missing_rate" in header
    assert "safety_budget_insufficient_share" in header

    assert (out_dir / "figures" / "fig_nlq_critical_fn_rate_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_nlq_critical_fn_rate_vs_budget_seconds.pdf").exists()
    assert (out_dir / "figures" / "fig_nlq_failure_attribution_vs_budget_seconds.png").exists()
    assert (out_dir / "figures" / "fig_nlq_failure_attribution_vs_budget_seconds.pdf").exists()

    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    outputs = snapshot.get("outputs", {})
    assert isinstance(outputs.get("safety_figures"), list)
    assert outputs.get("safety_figures")
