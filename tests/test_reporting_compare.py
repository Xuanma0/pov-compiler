from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _write_cross_pair(cross_dir: Path, *, hit_full: float, strict_full: float) -> None:
    cross_dir.mkdir(parents=True, exist_ok=True)
    overall = pd.DataFrame(
        [
            {
                "video_uid": "v1",
                "variant": "highlights_only",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 200.0,
                "budget_max_decisions": 12.0,
                "hit_at_k": 0.40,
                "mrr": 0.30,
                "coverage_ratio": 0.50,
                "compression_ratio": 2.0,
            },
            {
                "video_uid": "v1",
                "variant": "full",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 200.0,
                "budget_max_decisions": 12.0,
                "hit_at_k": hit_full,
                "mrr": 0.50,
                "coverage_ratio": 0.60,
                "compression_ratio": 2.5,
            },
        ]
    )
    by_type = pd.DataFrame(
        [
            {
                "video_uid": "v1",
                "query_type": "hard_pseudo_token",
                "variant": "highlights_only",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 200.0,
                "budget_max_decisions": 12.0,
                "hit_at_k": 0.30,
                "mrr": 0.20,
            },
            {
                "video_uid": "v1",
                "query_type": "hard_pseudo_token",
                "variant": "full",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 200.0,
                "budget_max_decisions": 12.0,
                "hit_at_k": hit_full,
                "mrr": 0.40,
            },
        ]
    )
    nlq = pd.DataFrame(
        [
            {
                "video_uid": "v1",
                "variant": "highlights_only",
                "query_type": "hard_pseudo_token",
                "duration_bucket": "short",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 200.0,
                "budget_max_decisions": 12.0,
                "num_queries": 10,
                "hit_at_k": 0.40,
                "hit_at_k_strict": 0.20,
                "hit_at_1_strict": 0.10,
                "mrr": 0.30,
                "top1_in_distractor_rate": 0.25,
            },
            {
                "video_uid": "v1",
                "variant": "full",
                "query_type": "hard_pseudo_token",
                "duration_bucket": "short",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 200.0,
                "budget_max_decisions": 12.0,
                "num_queries": 10,
                "hit_at_k": hit_full,
                "hit_at_k_strict": strict_full,
                "hit_at_1_strict": strict_full - 0.1,
                "mrr": 0.50,
                "top1_in_distractor_rate": 0.10,
            },
            {
                "video_uid": "v1",
                "variant": "full",
                "query_type": "hard_pseudo_decision",
                "duration_bucket": "medium",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 200.0,
                "budget_max_decisions": 12.0,
                "num_queries": 8,
                "hit_at_k": hit_full,
                "hit_at_k_strict": strict_full,
                "hit_at_1_strict": strict_full - 0.1,
                "mrr": 0.50,
                "top1_in_distractor_rate": 0.10,
            },
        ]
    )
    overall.to_csv(cross_dir / "results_overall.csv", index=False)
    by_type.to_csv(cross_dir / "results_by_query_type.csv", index=False)
    nlq.to_csv(cross_dir.parent / "nlq_summary_all.csv", index=False)


def test_make_paper_figures_compare_outputs(tmp_path: Path) -> None:
    main_dir = tmp_path / "main_run"
    compare_dir = tmp_path / "compare_run"
    main_cross = main_dir / "eval"
    compare_cross = compare_dir / "eval"
    _write_cross_pair(main_cross, hit_full=0.60, strict_full=0.50)
    _write_cross_pair(compare_cross, hit_full=0.40, strict_full=0.30)

    compare_snapshot = {
        "args": {
            "cross_dir": str(compare_cross),
            "nlq_csv": str(compare_dir / "nlq_summary_all.csv"),
            # Intentionally different to verify compare uses main selected budget.
            "budget_used": {"max_total_s": 20.0, "max_tokens": 50.0, "max_decisions": 4.0},
            "macro_avg": True,
        }
    }
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "snapshot.json").write_text(json.dumps(compare_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    out_dir = tmp_path / "out_compare"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_paper_figures.py"),
        "--cross_dir",
        str(main_cross),
        "--nlq_csv",
        str(main_dir / "nlq_summary_all.csv"),
        "--out_dir",
        str(out_dir),
        "--macro_avg",
        "--label",
        "real",
        "--compare_dir",
        str(compare_dir),
        "--compare_label",
        "stub",
        "--formats",
        "png",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout

    table_compare_md = out_dir / "tables" / "table_compare.md"
    assert table_compare_md.exists()
    text = table_compare_md.read_text(encoding="utf-8")
    assert "real_hit_at_k_strict" in text
    assert "stub_hit_at_k_strict" in text
    assert "delta_hit_at_k_strict" in text
    assert "0.2000" in text  # full: 0.50 - 0.30

    assert (out_dir / "figures" / "fig_compare_delta_strict.png").exists()
    assert (out_dir / "figures" / "fig_compare_by_query_type.png").exists()
    assert (out_dir / "figures" / "fig_compare_by_duration_bucket.png").exists()

    snapshot_out = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    used = snapshot_out["args"]["budget_used"]
    assert float(used["max_total_s"]) == 60.0
    assert float(used["max_tokens"]) == 200.0
    assert float(used["max_decisions"]) == 12.0
    assert snapshot_out["args"]["label"] == "real"
    assert snapshot_out["args"]["compare_label"] == "stub"
