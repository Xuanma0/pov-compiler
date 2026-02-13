from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.nlq.safety import SafetyGateConfig, build_safety_report


def _rows() -> list[dict[str, object]]:
    return [
        {
            "qid": "q1",
            "query_type": "hard_pseudo_anchor",
            "query": "when did I look around",
            "hit_at_k_strict": 0.0,
            "candidate_count": 2,
            "filtered_hits_before": 3,
            "filtered_hits_after": 0,
            "top1_in_distractor": 0.0,
            "budget_max_total_s": 60.0,
            "budget_max_tokens": 200,
            "budget_max_decisions": 12,
        },
        {
            "qid": "q2",
            "query_type": "hard_pseudo_decision",
            "query": "when did I pause",
            "hit_at_k_strict": 1.0,
            "candidate_count": 2,
            "filtered_hits_before": 2,
            "filtered_hits_after": 2,
            "top1_in_distractor": 0.0,
            "budget_max_total_s": 60.0,
            "budget_max_tokens": 200,
            "budget_max_decisions": 12,
        },
        {
            "qid": "q3",
            "query_type": "pseudo_time",
            "query": "time=10-20",
            "hit_at_k_strict": 0.0,
            "candidate_count": 1,
            "filtered_hits_before": 1,
            "filtered_hits_after": 1,
            "top1_in_distractor": 1.0,
            "budget_max_total_s": 60.0,
            "budget_max_tokens": 200,
            "budget_max_decisions": 12,
        },
    ]


def test_safety_gate_fails_on_critical_fn() -> None:
    report = build_safety_report(
        video_id="demo",
        per_query_rows=_rows(),
        gate_cfg=SafetyGateConfig(enabled=True, max_critical_fn=0),
    )
    assert report["critical_fn_count"] == 1
    assert report["pass_gate"] is False
    assert report["critical_failures"][0]["reason"] == "constraints_over_filtered"


def test_safety_report_has_required_fields() -> None:
    report = build_safety_report(
        video_id="demo",
        per_query_rows=_rows(),
        gate_cfg={"enabled": True, "max_critical_fn": 2},
    )
    for key in (
        "video_id",
        "strict_metric",
        "critical_query_types",
        "critical_fn_count",
        "pass_gate",
        "reason_counts",
        "critical_failures",
    ):
        assert key in report
    assert isinstance(report["critical_failures"], list)

