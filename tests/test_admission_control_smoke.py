from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pov_compiler.bench.reporting.admission_control import write_admission_outputs


def test_admission_control_smoke(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    (suite_dir / "manifest").mkdir(parents=True, exist_ok=True)
    (suite_dir / "result_health").mkdir(parents=True, exist_ok=True)
    (suite_dir / "result_diagnosis").mkdir(parents=True, exist_ok=True)
    (suite_dir / "provider_telemetry").mkdir(parents=True, exist_ok=True)
    (suite_dir / "delta_audit").mkdir(parents=True, exist_ok=True)

    (suite_dir / "manifest" / "experiment_manifest.yaml").write_text(
        "\n".join(
            [
                "suite_id: v1.47_main_real_admission",
                "admission_profile: main_real",
                "min_selected_uids: 2",
                "min_coverage_score_mean: 2.0",
                "min_significance_available_rate: 0.50",
                "max_no_data_rate: 0.50",
                "min_effect_size_nonzero_rate: 0.34",
                "max_provider_noise_parse_fail_rate: 0.10",
                "max_provider_noise_fallback_rate: 0.10",
                "min_usage_present_rate: 0.50",
                "allow_partial: true",
            ]
        ),
        encoding="utf-8",
    )
    (suite_dir / "result_health" / "snapshot.json").write_text(
        json.dumps(
            {
                "overall_no_data_reason_counts": {"ok": 3, "insufficient_pairs": 1},
                "gate": {
                    "gate_status": "ok",
                    "metrics": {
                        "selected_uids_count": 2,
                        "coverage_score_mean": 2.4,
                        "no_data_rate": 0.25,
                        "significance_available_rate": 0.50,
                        "effect_size_nonzero_rate": 0.25,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (suite_dir / "result_diagnosis" / "snapshot.json").write_text(
        json.dumps(
            {
                "selected_uids_count": 2,
                "coverage_score_stats": {"mean": 2.4},
                "significance_available_rate": 0.50,
                "effect_size_nonzero_rate": 0.25,
            }
        ),
        encoding="utf-8",
    )
    (suite_dir / "provider_telemetry" / "summary.json").write_text(
        json.dumps(
            {
                "availability": "ok",
                "usage_present_rate": 0.75,
                "structured_parse_fail_rate_mean": 0.12,
                "planner_fallback_rate": 0.04,
            }
        ),
        encoding="utf-8",
    )
    (suite_dir / "delta_audit" / "snapshot.json").write_text(
        json.dumps({"main_recommendation": "reduce_provider_noise"}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "admission_control"
    outputs = write_admission_outputs(suite_dir=suite_dir, out_dir=out_dir)
    assert outputs["admission_status"] == "partial"
    snapshot = json.loads((out_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot.get("admission_status") == "partial"
    assert "structured_parse_fail_rate_mean>0.10" in " ".join(snapshot.get("admission_fail_reasons", []))
    assert snapshot.get("delta_audit_summary", {}).get("main_recommendation") == "reduce_provider_noise"
