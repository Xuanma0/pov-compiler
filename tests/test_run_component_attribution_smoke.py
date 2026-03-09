from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_pipeline_json(path: Path, video_id: str) -> None:
    payload = {
        "video_id": video_id,
        "meta": {"duration_s": 24.0, "fps": 30.0},
        "events": [
            {"id": "event_0001", "t0": 0.0, "t1": 10.0, "scores": {"boundary_conf": 0.7}, "anchors": []},
            {"id": "event_0002", "t0": 10.0, "t1": 24.0, "scores": {"boundary_conf": 0.6}, "anchors": []},
        ],
        "highlights": [],
        "decision_points": [],
        "token_codec": {"version": "0.2", "vocab": [], "tokens": []},
        "debug": {
            "signals": {
                "time": [0.0, 8.0, 16.0, 24.0],
                "motion_energy": [0.1, 0.3, 0.2, 0.25],
                "embed_dist": [0.1, 0.2, 0.15, 0.22],
                "boundary_score": [0.1, 0.5, 0.3, 0.4],
            }
        },
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
        "a=p.parse_args()",
        "out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)",
        "score=max(0.0, min(1.0, a.budget_max_total_s/100.0))",
        "rows=[{'variant':'full','hit_at_k_strict':score,'hit_at_1_strict':score*0.8,'top1_in_distractor':max(0.0,0.5-score*0.3),'mrr':score*0.9}]",
        "with (out/'nlq_results.csv').open('w', encoding='utf-8', newline='') as f:",
        " w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]",
        "(out/'nlq_summary.csv').write_text('query_type,variant,hit_at_k_strict\\n', encoding='utf-8')",
        "safety={'critical_fn_rate':max(0.0,0.4-score*0.2),'variant_stats':{'full':{'critical_fn_rate':max(0.0,0.4-score*0.2)}}}",
        "(out/'safety_report.json').write_text(json.dumps(safety), encoding='utf-8')",
        "(out/'nlq_report.md').write_text('# fake', encoding='utf-8')",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_run_component_attribution_smoke(tmp_path: Path) -> None:
    root = tmp_path / "ego_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "demo.mp4").write_bytes(b"0")
    uid = hashlib.md5("demo.mp4".encode("utf-8")).hexdigest()
    uids_file = tmp_path / "uids.txt"
    uids_file.write_text(uid + "\n", encoding="utf-8")

    out_dir = tmp_path / "component_out"
    for code in ("A", "B", "C", "D"):
        run_dir = out_dir / f"run_{code}"
        _write_pipeline_json(run_dir / "json" / f"{uid}_v03_decisions.json", uid)
        (run_dir / "cache" / f"{uid}.index.npz").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "cache" / f"{uid}.index.npz").write_bytes(b"NPZ")
        (run_dir / "cache" / f"{uid}.index_meta.json").write_text("{}", encoding="utf-8")
        (run_dir / "eval" / uid).mkdir(parents=True, exist_ok=True)
        (run_dir / "eval" / uid / "report.md").write_text("# ready", encoding="utf-8")

    fake_eval = tmp_path / "fake_eval_nlq.py"
    _write_fake_eval_script(fake_eval)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_component_attribution.py"),
        "--root",
        str(root),
        "--uids-file",
        str(uids_file),
        "--out_dir",
        str(out_dir),
        "--jobs",
        "1",
        "--budgets",
        "20/50/4,60/200/12",
        "--query",
        "anchor=turn_head top_k=6",
        "--query",
        "decision=ATTENTION_TURN_HEAD top_k=6",
        "--with-nlq",
        "--nlq-eval-script",
        str(fake_eval),
        "--no-with-perception",
        "--with-streaming-budget",
        "--streaming-step-s",
        "8",
        "--min-size-bytes",
        "0",
        "--probe-candidates",
        "0",
        "--max-uids",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    compare_dir = out_dir / "compare"
    assert (compare_dir / "tables" / "table_component_attribution.csv").exists()
    assert (compare_dir / "tables" / "table_component_attribution.md").exists()
    assert (compare_dir / "figures" / "fig_component_attribution_delta.png").exists()
    assert (compare_dir / "figures" / "fig_component_attribution_delta.pdf").exists()
    assert (compare_dir / "figures" / "fig_component_attribution_tradeoff.png").exists()
    assert (compare_dir / "figures" / "fig_component_attribution_tradeoff.pdf").exists()
    assert (compare_dir / "compare_summary.json").exists()
    assert (compare_dir / "snapshot.json").exists()
    assert (compare_dir / "commands.sh").exists()
    assert (compare_dir / "README.md").exists()

    rows = _read_csv_rows(compare_dir / "tables" / "table_component_attribution.csv")
    settings = [r.get("setting") for r in rows]
    assert settings == ["A", "B", "C", "D", "SYNERGY"]
