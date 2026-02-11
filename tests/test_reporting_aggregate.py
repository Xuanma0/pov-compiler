from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.aggregate import (
    aggregate_over_videos,
    compute_deltas,
    pick_budget_slice,
)


def test_aggregate_over_videos_and_budget_slice_and_deltas() -> None:
    df = pd.DataFrame(
        [
            {
                "video_uid": "v1",
                "query_type": "pseudo_token",
                "variant": "highlights_only",
                "budget_max_total_s": 40.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.40,
                "mrr": 0.30,
            },
            {
                "video_uid": "v2",
                "query_type": "pseudo_token",
                "variant": "highlights_only",
                "budget_max_total_s": 40.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.20,
                "mrr": 0.10,
            },
            {
                "video_uid": "v1",
                "query_type": "pseudo_token",
                "variant": "full",
                "budget_max_total_s": 40.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.70,
                "mrr": 0.60,
            },
            {
                "video_uid": "v2",
                "query_type": "pseudo_token",
                "variant": "full",
                "budget_max_total_s": 40.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.50,
                "mrr": 0.40,
            },
            {
                "video_uid": "v1",
                "query_type": "pseudo_token",
                "variant": "highlights_plus_tokens",
                "budget_max_total_s": 40.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.60,
                "mrr": 0.50,
            },
            {
                "video_uid": "v2",
                "query_type": "pseudo_token",
                "variant": "highlights_plus_tokens",
                "budget_max_total_s": 40.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.40,
                "mrr": 0.30,
            },
            {
                "video_uid": "v1",
                "query_type": "pseudo_token",
                "variant": "full",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.90,
                "mrr": 0.80,
            },
            {
                "video_uid": "v2",
                "query_type": "pseudo_token",
                "variant": "full",
                "budget_max_total_s": 60.0,
                "budget_max_tokens": 100,
                "budget_max_decisions": 8,
                "hit_at_k": 0.70,
                "mrr": 0.60,
            },
        ]
    )

    budget_df = pick_budget_slice(
        df,
        budget={"max_total_s": 40.0, "max_tokens": 100, "max_decisions": 8},
    )
    assert len(budget_df) == 6

    macro = aggregate_over_videos(
        budget_df,
        group_cols=["variant"],
        metric_cols=["hit_at_k", "mrr"],
        agg="mean",
    )
    full_hit = float(macro.loc[macro["variant"] == "full", "hit_at_k"].iloc[0])
    full_mrr = float(macro.loc[macro["variant"] == "full", "mrr"].iloc[0])
    assert abs(full_hit - 0.60) < 1e-6
    assert abs(full_mrr - 0.50) < 1e-6

    by_type_macro = aggregate_over_videos(
        budget_df,
        group_cols=["query_type", "variant", "budget_max_total_s", "budget_max_tokens", "budget_max_decisions"],
        metric_cols=["hit_at_k", "mrr"],
        agg="mean",
    )
    deltas = compute_deltas(
        by_type_macro,
        baseline_variant="highlights_only",
        target_variants=["highlights_plus_tokens", "full"],
    )
    full_delta = deltas.loc[deltas["variant"] == "full", "delta_hit_at_k"].iloc[0]
    tokens_delta = deltas.loc[deltas["variant"] == "highlights_plus_tokens", "delta_hit_at_k"].iloc[0]
    assert abs(float(full_delta) - 0.30) < 1e-6
    assert abs(float(tokens_delta) - 0.20) < 1e-6

