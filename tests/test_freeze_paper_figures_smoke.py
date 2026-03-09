from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_freeze_paper_figures_smoke(tmp_path: Path) -> None:
    paper_ready_dir = tmp_path / "paper_ready"
    (paper_ready_dir / "canonical" / "tables").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "canonical" / "figures").mkdir(parents=True, exist_ok=True)
    (paper_ready_dir / "manifest").mkdir(parents=True, exist_ok=True)

    (paper_ready_dir / "canonical" / "tables" / "Table_1.csv").write_text("task,budget_key\nnlq,20/50/4\n", encoding="utf-8")
    (paper_ready_dir / "canonical" / "tables" / "Table_1.md").write_text("# Table 1\n", encoding="utf-8")
    (paper_ready_dir / "canonical" / "figures" / "Figure_2.png").write_bytes(b"PNG")
    paper_map = tmp_path / "main_result_map_v1.yaml"
    paper_map.write_text(
        "\n".join(
            [
                "paper_map_id: main_result_map_v1",
                'paper_map_version: "1"',
                "description: freeze smoke",
                "entries:",
                "  - canonical_id: Table 1",
                "    kind: table",
                "    title: Main Results",
                "    sources:",
                "      - compare/tables/table_main_results.csv",
                "      - compare/tables/table_main_results.md",
                "  - canonical_id: Figure 2",
                "    kind: figure",
                "    title: Main Frontier",
                "    sources:",
                "      - compare/figures/fig_main_budget_frontier.png",
            ]
        ),
        encoding="utf-8",
    )
    (paper_ready_dir / "manifest" / "main_result_map_v1.yaml").write_text(paper_map.read_text(encoding="utf-8"), encoding="utf-8")

    out_dir = tmp_path / "paper_freeze"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "freeze_paper_figures.py"),
        "--paper-ready-dir",
        str(paper_ready_dir),
        "--paper-map",
        str(paper_map),
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "saved_freeze_manifest=" in proc.stdout
    assert (out_dir / "freeze_manifest.json").exists()
    assert (out_dir / "paper_artifacts_sha256.csv").exists()
    payload = json.loads((out_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert payload.get("paper_map_id") == "main_result_map_v1"
    assert int(payload.get("artifact_count", 0)) >= 3
