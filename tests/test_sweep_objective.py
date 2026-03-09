from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.sweep_utils import compute_objective, rank_rows_by_metric


def test_compute_objective_combo_prefers_strict_and_low_fp() -> None:
    a = compute_objective(hit_at_k_strict=0.6, hit_at_1_strict=0.2, fp_rate=0.2, metric="objective_combo")
    b = compute_objective(hit_at_k_strict=0.6, hit_at_1_strict=0.2, fp_rate=0.5, metric="objective_combo")
    assert a > b


def test_rank_rows_by_metric_orders_correctly() -> None:
    rows = [
        {"cfg_name": "a", "objective": 0.30, "hit_at_k_strict": 0.4, "hit_at_1_strict": 0.1, "fp_rate": 0.3},
        {"cfg_name": "b", "objective": 0.45, "hit_at_k_strict": 0.5, "hit_at_1_strict": 0.2, "fp_rate": 0.2},
        {"cfg_name": "c", "objective": 0.20, "hit_at_k_strict": 0.2, "hit_at_1_strict": 0.1, "fp_rate": 0.1},
    ]

    ranked_obj = rank_rows_by_metric(rows, metric="objective_combo")
    assert ranked_obj[0]["cfg_name"] == "b"

    ranked_fp = rank_rows_by_metric(rows, metric="fp_rate")
    assert ranked_fp[0]["cfg_name"] == "c"
