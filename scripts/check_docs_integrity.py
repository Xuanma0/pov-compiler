from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PAIR_MAP = [
    ("docs/README.en.md", "docs/README.zh-CN.md"),
    ("docs/en/project_overview.md", "docs/zh-CN/project_overview.md"),
    ("docs/en/repo_map.md", "docs/zh-CN/repo_map.md"),
    ("docs/en/experiment_history.md", "docs/zh-CN/experiment_history.md"),
    ("docs/en/current_mainline_status.md", "docs/zh-CN/current_mainline_status.md"),
    ("docs/en/development_workflow.md", "docs/zh-CN/development_workflow.md"),
    ("docs/en/glossary.md", "docs/zh-CN/glossary.md"),
]

REQUIRED_CANONICAL = [
    "docs/README.en.md",
    "docs/README.zh-CN.md",
    "docs/doc_inventory.md",
    "docs/doc_migration_map.md",
    "docs/doc_style_guide.md",
    "docs/archive/README.md",
]

PAPER_READY_REQUIRED = [
    "persistent_memory_main_compare",
    "persistent_memory_main_decision",
    "mainline_admission_cleanup",
    "harder_sample_contract",
    "mainline_admission_closure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check canonical docs, bilingual pairs, and reading-order integrity.")
    parser.add_argument("--docs-root", default=str(ROOT / "docs"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--paper-ready", default=str(ROOT / "data" / "outputs" / "v160_mainline_cleanup" / "paper_ready"))
    parser.add_argument("--submission-pack", default=str(ROOT / "data" / "outputs" / "v160_mainline_cleanup" / "submission_pack"))
    return parser.parse_args()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    args = parse_args()
    readme_path = Path(args.readme)
    paper_ready = Path(args.paper_ready)
    submission_pack = Path(args.submission_pack)

    missing: list[str] = []
    pair_errors: list[str] = []
    reading_order_errors: list[str] = []

    for rel in REQUIRED_CANONICAL:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)

    for left_rel, right_rel in PAIR_MAP:
        left = ROOT / left_rel
        right = ROOT / right_rel
        if not left.exists():
            missing.append(left_rel)
            continue
        if not right.exists():
            missing.append(right_rel)
            continue
        left_text = _read(left)
        right_text = _read(right)
        if Path(right_rel).name not in left_text:
            pair_errors.append(left_rel)
        if Path(left_rel).name not in right_text:
            pair_errors.append(right_rel)

    if readme_path.exists():
        readme_text = _read(readme_path)
        for required in ("docs/README.en.md", "docs/README.zh-CN.md", "current_mainline_status.md"):
            if required not in readme_text:
                reading_order_errors.append(f"README.md missing {required}")
    else:
        missing.append(str(readme_path))

    docs_entry = ROOT / "docs" / "README.en.md"
    if docs_entry.exists():
        docs_entry_text = _read(docs_entry)
        for required in (
            "current_mainline_status.md",
            "experiment_history.md",
            "project_overview.md",
            "repo_map.md",
            "development_workflow.md",
            "glossary.md",
        ):
            if required not in docs_entry_text:
                reading_order_errors.append(f"docs/README.en.md missing {required}")

    paper_ready_status = "skipped"
    if paper_ready.exists():
        paper_ready_status = "ok"
        for panel in PAPER_READY_REQUIRED:
            if not (paper_ready / panel).exists():
                missing.append(str((paper_ready / panel).relative_to(ROOT)))
                paper_ready_status = "missing_panels"
        report = paper_ready / "report.md"
        if report.exists():
            report_text = _read(report)
            for required in (
                "Persistent Memory Main Compare",
                "Persistent Memory Main Decision",
                "Mainline Admission Cleanup",
                "Harder Sample Contract",
                "Mainline Admission Closure",
            ):
                if required not in report_text:
                    reading_order_errors.append(f"{report.relative_to(ROOT)} missing {required}")
        else:
            missing.append(str(report.relative_to(ROOT)))
            paper_ready_status = "missing_report"

    submission_pack_status = "skipped"
    if submission_pack.exists():
        submission_pack_status = "ok"
        readme = submission_pack / "README.md"
        if readme.exists():
            text = _read(readme)
            for required in (
                "persistent_memory_main_compare/",
                "persistent_memory_main_decision/",
                "mainline_admission_cleanup/",
                "harder_sample_contract/",
                "mainline_admission_closure/",
            ):
                if required not in text:
                    reading_order_errors.append(f"{readme.relative_to(ROOT)} missing {required}")
            if "None" in text:
                reading_order_errors.append(f"{readme.relative_to(ROOT)} contains None")
        else:
            missing.append(str(readme.relative_to(ROOT)))
            submission_pack_status = "missing_readme"

    status = "ok" if not missing and not pair_errors and not reading_order_errors else "issues_found"
    print(f"required_docs={len(REQUIRED_CANONICAL)}")
    print(f"bilingual_pairs={len(PAIR_MAP)}")
    print(f"missing_items={len(missing)}")
    print(f"pair_errors={len(pair_errors)}")
    print(f"reading_order_errors={len(reading_order_errors)}")
    print(f"paper_ready_status={paper_ready_status}")
    print(f"submission_pack_status={submission_pack_status}")
    print(f"status={status}")
    if missing:
        print(f"missing_list={missing}")
    if pair_errors:
        print(f"pair_error_list={pair_errors}")
    if reading_order_errors:
        print(f"reading_order_error_list={reading_order_errors}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
