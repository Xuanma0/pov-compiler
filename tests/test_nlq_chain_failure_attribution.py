from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.datasets import NLQSample
from pov_compiler.bench.nlq.evaluator import evaluate_nlq_samples
from pov_compiler.schemas import Event, Output, TokenCodec


def _output() -> Output:
    return Output(
        video_id="chain_failure_demo",
        meta={"duration_s": 20.0},
        events=[Event(id="event_0001", t0=0.0, t1=20.0)],
        token_codec=TokenCodec(version="0.2", vocab=[], tokens=[]),
    )


def test_chain_failure_reason_columns_exist() -> None:
    sample = NLQSample(
        qid="cnlq_fail_0001",
        query="anchor=not_exist top_k=6 then token=SCENE_CHANGE top_k=6",
        query_type="hard_pseudo_chain",
        gt_span=(1.0, 2.0),
        top_k=6,
        meta={"chain_meta": {"combo": "anchor_to_token", "derive": "time_only"}},
    )
    result = evaluate_nlq_samples(
        output=_output(),
        samples=[sample],
        budgets={"max_total_s": [20], "max_tokens": [50], "max_decisions": [4]},
        sweep=False,
        allow_gt_fallback=False,
        variants=["full"],
    )
    per_query_rows = list(result.get("per_query_rows", []))
    assert per_query_rows
    row = dict(per_query_rows[0])
    assert "chain_fail_reason" in row
    assert row["chain_fail_reason"] in {
        "step1_no_hit",
        "step2_no_hit",
        "constraints_over_filtered",
        "retrieval_distractor",
        "evidence_missing",
        "budget_insufficient",
        "other",
        "success",
    }
    assert "chain_derived_time_used" in row
    assert "chain_derived_place_used" in row
    assert "chain_derived_object_used" in row

    by_type_rows = list(result.get("by_query_type_rows", []))
    assert by_type_rows
    agg = dict(by_type_rows[0])
    assert "chain_fail_step1_no_hit_rate" in agg
    assert "chain_fail_constraints_over_filtered_rate" in agg
    assert "chain_derived_nonempty_rate" in agg
