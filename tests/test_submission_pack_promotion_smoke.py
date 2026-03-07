from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submission_pack_promotion_smoke(tmp_path: Path) -> None:
    paper_ready_dir = tmp_path / "paper_ready"
    (paper_ready_dir / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "paper_map_resolved.json").write_text(
        json.dumps({"rows": [{"canonical_id": "Table 1", "canonical_relpath": "canonical/tables/Table_1.csv"}]}),
        encoding="utf-8",
    )
    (paper_ready_dir / "tables" / "table_budget_panel.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "figures" / "fig_budget_primary_vs_seconds_panel.png").write_bytes(b"PNG")
    (paper_ready_dir / "canonical" / "tables" / "Table_1.csv").write_text("task\nnlq\n", encoding="utf-8")
    (paper_ready_dir / "provider_normalization").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "provider_normalization" / "snapshot.json").write_text(
        json.dumps({"normalization_status": "partial"}),
        encoding="utf-8",
    )
    (paper_ready_dir / "query_promotion_pack").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "query_promotion_pack" / "report.md").write_text("# promotion pack\n", encoding="utf-8")
    (paper_ready_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (paper_ready_dir / "snapshot.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    provider_normalization_dir = tmp_path / "provider_normalization"
    (provider_normalization_dir / "tables").mkdir(parents=True, exist_ok=True)
    (provider_normalization_dir / "tables" / "table_provider_normalization.csv").write_text("provider\nfake\n", encoding="utf-8")
    (provider_normalization_dir / "report.md").write_text("# normalization\n", encoding="utf-8")
    (provider_normalization_dir / "snapshot.json").write_text(
        json.dumps({"normalization_status": "partial"}),
        encoding="utf-8",
    )

    query_promotion_pack_dir = tmp_path / "query_promotion_pack"
    (query_promotion_pack_dir / "query_pack").mkdir(parents=True, exist_ok=True)
    (query_promotion_pack_dir / "query_pack" / "promoted_queries.yaml").write_text("queries: []\n", encoding="utf-8")
    (query_promotion_pack_dir / "query_pack" / "analysis_only_queries.yaml").write_text("queries: []\n", encoding="utf-8")
    (query_promotion_pack_dir / "query_pack" / "promotion_summary.json").write_text(
        json.dumps({"promoted_count": 0}),
        encoding="utf-8",
    )
    (query_promotion_pack_dir / "report.md").write_text("# promotion\n", encoding="utf-8")

    out_dir = tmp_path / "submission_pack"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_submission_pack.py"),
            "--paper-ready-dir",
            str(paper_ready_dir),
            "--out-dir",
            str(out_dir),
            "--provider-normalization-dir",
            str(provider_normalization_dir),
            "--query-promotion-pack-dir",
            str(query_promotion_pack_dir),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert (out_dir / "provider_normalization" / "snapshot.json").exists()
    assert (out_dir / "query_promotion_pack" / "query_pack" / "promotion_summary.json").exists()
    assert (out_dir / "paper_ready" / "provider_normalization" / "snapshot.json").exists()
    assert (out_dir / "paper_ready" / "query_promotion_pack" / "report.md").exists()
    readme_text = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Normalized Telemetry" in readme_text
    assert "Query Promotion" in readme_text
