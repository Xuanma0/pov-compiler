# Documentation Style Guide

## Purpose

Define the repository-wide rules for Markdown and text documentation after the v1.61 cleanup.

## Encoding

- Use UTF-8 for all text docs.
- Do not keep obvious mojibake, replacement characters, or broken BOM artifacts.
- If the original wording cannot be fully recovered, preserve a conservative repair and record it in `docs/doc_migration_map.md`.

## Canonical vs Archive

- `canonical`: current authoritative docs for external or onboarding reading.
- `current`: still relevant reference docs, but not the primary entry.
- `archive`: historical evidence retained for version evolution and research traceability.
- `duplicate`: overlapping docs that should eventually merge into a canonical target.
- `remove_candidate`: low-value docs that can be removed after migration is confirmed.
- `broken_encoding`: docs that require encoding repair before they can stay current.

## Bilingual Rules

- Canonical docs should appear as English and Simplified Chinese pairs whenever practical.
- Each paired doc should link to its counterpart in the first line.
- Keep section order stable between the two languages.
- If one side must be shorter, do not change the conclusion or status wording.

## Reading Order

Repository-level reading order:

1. `docs/en/current_mainline_status.md` or `docs/zh-CN/current_mainline_status.md`
2. `docs/en/experiment_history.md` or `docs/zh-CN/experiment_history.md`
3. `docs/en/project_overview.md` or `docs/zh-CN/project_overview.md`
4. `docs/en/repo_map.md` or `docs/zh-CN/repo_map.md`
5. `docs/en/development_workflow.md` or `docs/zh-CN/development_workflow.md`
6. `docs/en/glossary.md` or `docs/zh-CN/glossary.md`

Generated evidence reading order:

1. `persistent_memory_main_compare/`
2. `persistent_memory_main_decision/`
3. `mainline_admission_cleanup/`
4. `harder_sample_contract/`
5. `mainline_admission_closure/`

## Status Wording

Current canonical wording must match the v1.60 evidence:

- `persistent memory mainline`: promoted
- `mainline admission closure`: `partial_but_harder`
- `sample_contract_status`: `adequate`
- `large_sample_claim_status`: `supported`
- residual caveat: provider semantics gap

Do not rewrite these conclusions without new evidence.
