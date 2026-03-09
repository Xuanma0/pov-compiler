from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_telemetry_smoke(tmp_path: Path) -> None:
    paper_ready_dir = tmp_path / "paper_ready"
    (paper_ready_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "provider_telemetry").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (paper_ready_dir / "tables" / "table_budget_panel.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "figures" / "fig_budget_primary_vs_seconds_panel.png").write_bytes(b"PNG")
    (paper_ready_dir / "canonical" / "tables" / "Table_1.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "canonical" / "figures" / "Figure_2.png").write_bytes(b"PNG")
    (paper_ready_dir / "canonical" / "paper_map_resolved.json").write_text(
        json.dumps({"rows": [{"canonical_id": "Table 1", "canonical_relpath": "canonical/tables/Table_1.csv"}]}),
        encoding="utf-8",
    )
    (paper_ready_dir / "provider_telemetry" / "summary.json").write_text(
        json.dumps({"availability": "ok", "model_cost_known_rate": 1.0}),
        encoding="utf-8",
    )

    provider_telemetry_dir = tmp_path / "provider_telemetry"
    provider_telemetry_dir.mkdir(parents=True, exist_ok=True)
    (provider_telemetry_dir / "summary.json").write_text(
        json.dumps({"availability": "ok", "model_cost_known_rate": 1.0}),
        encoding="utf-8",
    )
    (provider_telemetry_dir / "by_variant.csv").write_text(
        "variant_label,provider,model,api_mode_used,usage_present,cost_known,calls_total\nreal,openai,gpt-4.1-mini,responses,False,False,0\n",
        encoding="utf-8",
    )

    result_diagnosis_dir = tmp_path / "result_diagnosis"
    (result_diagnosis_dir / "tables").mkdir(parents=True, exist_ok=True)
    (result_diagnosis_dir / "figures").mkdir(parents=True, exist_ok=True)
    (result_diagnosis_dir / "tables" / "table_result_diagnosis.csv").write_text("task\nnlq\n", encoding="utf-8")
    (result_diagnosis_dir / "report.md").write_text("# diagnosis report\n", encoding="utf-8")
    (result_diagnosis_dir / "snapshot.json").write_text(json.dumps({"diagnosis_recommendations": ["inspect telemetry"]}), encoding="utf-8")

    out_dir = tmp_path / "submission_pack"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_submission_pack.py"),
            "--paper-ready-dir",
            str(paper_ready_dir),
            "--out-dir",
            str(out_dir),
            "--result-diagnosis-dir",
            str(result_diagnosis_dir),
            "--provider-telemetry-dir",
            str(provider_telemetry_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "provider_telemetry" / "summary.json").exists()
    assert (out_dir / "paper_ready" / "provider_telemetry" / "summary.json").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Provider Noise" in readme_text
    assert "provider_telemetry/summary.json" in readme_text
