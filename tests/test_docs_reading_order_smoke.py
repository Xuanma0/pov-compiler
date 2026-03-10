from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docs_reading_order_smoke() -> None:
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_en = (ROOT / "docs" / "README.en.md").read_text(encoding="utf-8")
    docs_zh = (ROOT / "docs" / "README.zh-CN.md").read_text(encoding="utf-8")

    assert "docs/README.en.md" in root_readme
    assert "docs/README.zh-CN.md" in root_readme
    for required in (
        "current_mainline_status.md",
        "experiment_history.md",
        "project_overview.md",
        "repo_map.md",
        "development_workflow.md",
        "glossary.md",
    ):
        assert required in docs_en
        assert required in docs_zh
