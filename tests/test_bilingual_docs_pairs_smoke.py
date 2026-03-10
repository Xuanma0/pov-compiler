from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PAIRS = [
    (ROOT / "docs" / "README.en.md", ROOT / "docs" / "README.zh-CN.md"),
    (ROOT / "docs" / "en" / "project_overview.md", ROOT / "docs" / "zh-CN" / "project_overview.md"),
    (ROOT / "docs" / "en" / "repo_map.md", ROOT / "docs" / "zh-CN" / "repo_map.md"),
    (ROOT / "docs" / "en" / "experiment_history.md", ROOT / "docs" / "zh-CN" / "experiment_history.md"),
    (ROOT / "docs" / "en" / "current_mainline_status.md", ROOT / "docs" / "zh-CN" / "current_mainline_status.md"),
    (ROOT / "docs" / "en" / "development_workflow.md", ROOT / "docs" / "zh-CN" / "development_workflow.md"),
    (ROOT / "docs" / "en" / "glossary.md", ROOT / "docs" / "zh-CN" / "glossary.md"),
]


def test_bilingual_docs_pairs_smoke() -> None:
    for left, right in PAIRS:
        assert left.exists()
        assert right.exists()
        left_text = left.read_text(encoding="utf-8")
        right_text = right.read_text(encoding="utf-8")
        assert right.name in left_text
        assert left.name in right_text
